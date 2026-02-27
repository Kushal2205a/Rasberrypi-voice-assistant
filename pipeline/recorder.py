import threading
import queue
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from config import CHUNK_DURATION, SAMPLE_RATE

try:
    from config import INPUT_DEVICE
except Exception:
    INPUT_DEVICE = None


class StreamingRecorder:
    """
    Capture mic input and emit fixed-size int16 mono chunks.

    Two operating modes:
      Normal mode  — opens its own sd.InputStream (original behaviour).
      Push mode    — SharedMicStream writes directly into chunk_queue via
                     inject_audio(); no sd.InputStream is opened here.
                     Activated by passing push_mode=True.
    """

    def __init__(
        self,
        chunk_duration: float        = CHUNK_DURATION,
        sample_rate:    int          = SAMPLE_RATE,
        input_device:   Optional[int] = INPUT_DEVICE,
        latency:        str          = "high",
        push_mode:      bool         = False,
    ) -> None:
        self.chunk_duration = float(chunk_duration)
        self.sample_rate    = int(sample_rate)
        self.input_device   = input_device
        self.latency        = latency
        self._push_mode     = bool(push_mode)

        self.chunk_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=64)

        self.recording = False
        self._thread: Optional[threading.Thread] = None

        self._buf      = bytearray()
        self._buf_lock = threading.Lock()

        self._cb_blocksize    = max(256, int(self.sample_rate * min(0.10, self.chunk_duration)))
        self._last_status_log = 0.0

    def start(self) -> None:
        self.recording = True   # always re-enable (may have been stopped by previous session)
        if self._push_mode:
            # SharedMicStream feeds us; no audio thread needed
            return
        if self._thread and self._thread.is_alive():
            return  # already running in normal mode
        self._thread = threading.Thread(
            target=self._record_loop,
            name="StreamingRecorder",
            daemon=True,
        )
        self._thread.start()

    def inject_audio(self, indata: np.ndarray) -> None:
        """
        Called by SharedMicStream in SESSION mode to push raw audio in.
        Replicates the PortAudio callback logic without opening a stream.
        """
        if not self.recording:
            return

        chunk_samples = int(self.chunk_duration * self.sample_rate)
        chunk_bytes   = chunk_samples * 2

        b = np.ascontiguousarray(indata, dtype=np.int16).tobytes()
        with self._buf_lock:
            self._buf.extend(b)
            while len(self._buf) >= chunk_bytes:
                payload = bytes(self._buf[:chunk_bytes])
                del self._buf[:chunk_bytes]
                try:
                    self.chunk_queue.put_nowait(payload)
                except queue.Full:
                    try:
                        self.chunk_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.chunk_queue.put_nowait(payload)
                    except queue.Full:
                        pass

    def _record_loop(self) -> None:
        """Used only in normal (non-push) mode."""
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        chunk_bytes   = chunk_samples * 2

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                now = time.time()
                if now - self._last_status_log >= 0.5:
                    print(f"[REC] Audio status: {status}")
                    self._last_status_log = now
            if not self.recording:
                return
            b = indata.tobytes()
            with self._buf_lock:
                self._buf.extend(b)
                while len(self._buf) >= chunk_bytes:
                    payload = bytes(self._buf[:chunk_bytes])
                    del self._buf[:chunk_bytes]
                    try:
                        self.chunk_queue.put_nowait(payload)
                    except queue.Full:
                        try:
                            self.chunk_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.chunk_queue.put_nowait(payload)
                        except queue.Full:
                            pass

        with sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._cb_blocksize,
            latency=self.latency,
            callback=callback,
        ):
            while self.recording:
                time.sleep(0.05)

    def get_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            payload = self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return np.frombuffer(payload, dtype=np.int16).reshape(-1, 1)

    def clear_queue(self) -> None:
        try:
            while True:
                self.chunk_queue.get_nowait()
        except queue.Empty:
            pass
        with self._buf_lock:
            self._buf.clear()

    def stop(self) -> None:
        self.recording = False
        if self._push_mode:
            # In push mode the SharedMicStream keeps the real audio stream open.
            # Just mark as not recording; start() will re-enable it next session.
            return
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None