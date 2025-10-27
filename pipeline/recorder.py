import threading,queue
from typing import Optional
import numpy as np, sounddevice as sd 

from config import CHUNK_DURATION,SAMPLE_RATE

import math
try:
    import webrtcvad
    _HAVE_VAD = True
except Exception:
    _HAVE_VAD = False


def _resample_linear_mono_int16(x_int16: bytes, src_sr: int, dst_sr: int) -> np.ndarray:
    """ linear resample to dst_sr """
    if src_sr == dst_sr or not x_int16:
        return np.frombuffer(x_int16, dtype=np.int16)
    src = np.frombuffer(x_int16, dtype=np.int16).astype(np.float32)
    ratio = dst_sr / float(src_sr)
    xi = np.arange(0, src.shape[0], 1.0 / ratio, dtype=np.float32)
    if xi.size == 0:
        return np.zeros(0, dtype=np.int16)
    x = np.arange(src.shape[0], dtype=np.float32)
    out = np.interp(xi, x, src).astype(np.int16)
    return out

class VADGate:
    """WebRTC VAD-based start/stop gate over 20 ms 16 kHz frames."""
    def __init__(self, aggressiveness: int = 2, trigger_ms: int = 80, release_ms: int = 240):
        if not _HAVE_VAD:
            raise RuntimeError("webrtcvad not installed; pip install webrtcvad")
        self.vad = webrtcvad.Vad(int(aggressiveness))
        self.frame_ms = 20
        self.fs = 16000
        self._need = (self.fs * self.frame_ms) // 1000  # samples per 20 ms
        self._buf = bytearray()
        self._voiced_count_needed = max(1, trigger_ms // self.frame_ms)
        self._unvoiced_count_needed = max(1, release_ms // self.frame_ms)
        self._voiced_run = 0
        self._unvoiced_run = 0
        self.active = False  

    def push_pcm16_44k1(self, pcm_bytes_44100: bytes):
        """Feed 44.1 kHz mono int16 bytes; internally resamples to 16 kHz 20 ms frames."""
        # Resample chunk to 16 kHz
        pcm16_16k = _resample_linear_mono_int16(pcm_bytes_44100, 44100, self.fs).tobytes()
        if not pcm16_16k:
            return None

        # Append and slice into 20 ms frames
        self._buf.extend(pcm16_16k)
        events = []
        while len(self._buf) >= self._need * 2:  # 2 bytes per sample
            frame = bytes(self._buf[: self._need * 2])
            del self._buf[: self._need * 2]
            is_voiced = self.vad.is_speech(frame, self.fs)

            if is_voiced:
                self._voiced_run += 1
                self._unvoiced_run = 0
            else:
                self._unvoiced_run += 1
                self._voiced_run = 0

            was_active = self.active
            if not self.active and self._voiced_run >= self._voiced_count_needed:
                self.active = True
                events.append(("voice_start",))
            elif self.active and self._unvoiced_run >= self._unvoiced_count_needed:
                self.active = False
                events.append(("voice_end",))

            

        return events  




class StreamingRecorder:
    """Capture microphone input continuously and expose fixed-size chunks."""

    def __init__(self, chunk_duration: float = CHUNK_DURATION, sample_rate: int = SAMPLE_RATE):
        self.chunk_duration = float(chunk_duration)
        self.sample_rate = int(sample_rate)
        self.chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.recording = False
        self._thread: Optional[threading.Thread] = None

        self.use_vad = True
        self.vad = VADGate(aggressiveness=2, trigger_ms=80, release_ms=240) if _HAVE_VAD else None


    def start(self) -> None:
        """Start capturing microphone audio in a background thread."""

        if self.recording:
            return
        self.recording = True
        self._thread = threading.Thread(target=self._record_loop, name="StreamingRecorder", daemon=True)
        self._thread.start()

    def _record_loop(self) -> None:
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        with sd.InputStream(device=1,samplerate=self.sample_rate, channels=1, dtype="int16") as stream:
            while self.recording:
                audio_chunk, _ = stream.read(chunk_samples)
                # Copy to detach from PortAudio's buffers
                self.chunk_queue.put(audio_chunk.copy())

    def get_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self) -> None:
        """Remove any queued audio chunks without blocking."""

        try:
            while True:
                self.chunk_queue.get_nowait()
        except queue.Empty:
            return

    def stop(self) -> None:
        """Signal the recorder to stop and wait for the background thread."""

        self.recording = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

