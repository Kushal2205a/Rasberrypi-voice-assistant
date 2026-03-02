from __future__ import annotations

# NOTE: ALSA hw devices allow only ONE open stream. Opening a second stream — even milliseconds
# after closing the first — raises "Device unavailable [-9985]". This class keeps ONE stream open
# for the entire process lifetime and switches routing between wake-word and session modes via a flag.

import math
import threading
import time
from typing import TYPE_CHECKING, Optional
from collections import deque
import numpy as np
import sounddevice as sd

try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

if TYPE_CHECKING:
    from wake_word import WakeWordDetector
    from recorder import StreamingRecorder


class SharedMicStream:

    WAKE    = "wake"
    SESSION = "session"

    _OWW_SR = 16000  # OpenWakeWord always expects 16 kHz

    def __init__(
        self,
        device:         Optional[object] = None,
        sample_rate:    int              = 44100,
        chunk_duration: float            = 0.6,
        oww_chunk_ms:   int              = 80,
    ) -> None:
        self._device       = device
        self._sample_rate  = int(sample_rate)
        self._chunk_dur    = float(chunk_duration)

        self._oww_chunk_samples_16k = int(self._OWW_SR * oww_chunk_ms / 1000)
        self._oww_chunk_samples_nat = int(self._sample_rate * oww_chunk_ms / 1000)
        self._oww_buf               = bytearray()
        self._oww_buf_lock          = threading.Lock()

        self._wake_raw_q: deque = deque(maxlen=128)
        self._wake_thread: Optional[threading.Thread] = None

        g = math.gcd(self._sample_rate, self._OWW_SR)
        self._rs_up   = self._OWW_SR     // g
        self._rs_down = self._sample_rate // g

        self._mode      = self.WAKE
        self._mode_lock = threading.Lock()

        self._detector: Optional[WakeWordDetector] = None
        self._recorder: Optional[StreamingRecorder] = None

        self._stream:  Optional[sd.InputStream] = None
        self._running  = False
        self._cb_blocksize = max(256, int(self._sample_rate * 0.10))

        self._last_status_log = 0.0

    def attach_detector(self, detector: "WakeWordDetector") -> None:
        self._detector = detector

    def attach_recorder(self, recorder: "StreamingRecorder") -> None:
        self._recorder = recorder

    def set_mode(self, mode: str) -> None:
        if mode not in (self.WAKE, self.SESSION):
            raise ValueError(f"mode must be 'wake' or 'session', got {mode!r}")
        with self._mode_lock:
            self._mode = mode
            # NOTE: flush both buffers so stale audio from the previous mode doesn't bleed through
            self._oww_buf.clear()
            self._wake_raw_q.clear()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._wake_thread = threading.Thread(
            target=self._wake_process_loop,
            name="SharedMicWakeProc",
            daemon=True,
        )
        self._wake_thread.start()
        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._cb_blocksize,
            latency="high",
            callback=self._callback,
        )
        self._stream.start()
        print(f"[SharedMic] Stream open — device={self._device!r}, "
              f"{self._sample_rate} Hz, mode={self._mode}")

    def stop(self) -> None:
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._wake_thread:
            self._wake_thread.join(timeout=2.0)
            self._wake_thread = None
        print("[SharedMic] Stream closed.")

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            now = time.time()
            if now - self._last_status_log >= 1.0:
                print(f"[SharedMic] Audio status: {status}")
                self._last_status_log = now

        with self._mode_lock:
            mode = self._mode

        raw = indata.tobytes()

        if mode == self.SESSION:
            self._feed_recorder(raw)
        else:
            # NOTE: enqueue raw bytes for background thread — no work done in the PortAudio callback
            self._wake_raw_q.append(raw)

    def _feed_recorder(self, raw: bytes) -> None:
        if self._recorder is None:
            return
        chunk = np.frombuffer(raw, dtype=np.int16)
        self._recorder.inject_audio(chunk)

    def _wake_process_loop(self) -> None:
        oww_chunk_bytes_nat = self._oww_chunk_samples_nat * 2

        while self._running:
            if not self._wake_raw_q:
                time.sleep(0.005)
                continue

            raw = self._wake_raw_q.popleft()

            with self._mode_lock:
                mode = self._mode
            if mode != self.WAKE:
                continue

            if self._detector is None:
                continue

            with self._oww_buf_lock:
                self._oww_buf.extend(raw)
                while len(self._oww_buf) >= oww_chunk_bytes_nat:
                    chunk_nat = bytes(self._oww_buf[:oww_chunk_bytes_nat])
                    del self._oww_buf[:oww_chunk_bytes_nat]

                    src = np.frombuffer(chunk_nat, dtype=np.int16).astype(np.float32)
                    if _HAVE_SCIPY:
                        dst = resample_poly(src, self._rs_up, self._rs_down)
                    else:
                        ratio = self._OWW_SR / float(self._sample_rate)
                        xi = np.arange(0, len(src), 1.0 / ratio, dtype=np.float32)
                        dst = np.interp(np.arange(len(src), dtype=np.float32), xi, src) \
                            if xi.size else src
                    chunk_16k = np.clip(dst, -32768, 32767).astype(np.int16)

                    try:
                        self._detector.push_audio(chunk_16k)
                    except Exception:
                        pass