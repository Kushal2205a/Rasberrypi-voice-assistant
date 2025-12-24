import threading
import queue
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from config import CHUNK_DURATION, SAMPLE_RATE

try:
    from config import INPUT_DEVICE  # optional
except Exception:
    INPUT_DEVICE = None


class StreamingRecorder:
    """
    Capture mic input continuously and emit fixed-size int16 mono chunks.

    Notes for Pi:
      - Use a PortAudio callback so we don't miss audio when other threads are busy.
      - Keep the callback *very* light (no numpy work) to reduce "input overflow".
    """

    def __init__(
        self,
        chunk_duration: float = CHUNK_DURATION,
        sample_rate: int = SAMPLE_RATE,
        input_device: Optional[int] = INPUT_DEVICE,
        latency: str = "high",
    ) -> None:
        self.chunk_duration = float(chunk_duration)
        self.sample_rate = int(sample_rate)
        self.input_device = input_device
        self.latency = latency

        # Queue holds raw bytes; conversion to numpy happens in get_chunk().
        self.chunk_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=64)

        self.recording = False
        self._thread: Optional[threading.Thread] = None

        self._buf = bytearray()
        self._buf_lock = threading.Lock()

        # Larger blocks => fewer callbacks => fewer overflows on a busy Pi.
        # 100ms is a good compromise for command + conversation.
        self._cb_blocksize = max(256, int(self.sample_rate * min(0.10, self.chunk_duration)))

        self._last_status_log = 0.0

    def start(self) -> None:
        if self.recording:
            return
        self.recording = True
        self._thread = threading.Thread(target=self._record_loop, name="StreamingRecorder", daemon=True)
        self._thread.start()

    def _record_loop(self) -> None:
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        chunk_bytes = chunk_samples * 2  # int16 mono

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                now = time.time()
                if now - self._last_status_log >= 0.5:
                    print(f"[REC] Audio status: {status}")
                    self._last_status_log = now

            if not self.recording:
                return

            b = indata.tobytes()  # int16 mono
            with self._buf_lock:
                self._buf.extend(b)

                # Cut full chunks out of the buffer and enqueue them.
                while len(self._buf) >= chunk_bytes:
                    payload = bytes(self._buf[:chunk_bytes])
                    del self._buf[:chunk_bytes]

                    # Never block the audio callback; drop oldest if the queue is full.
                    try:
                        self.chunk_queue.put_nowait(payload)
                    except queue.Full:
                        try:
                            _ = self.chunk_queue.get_nowait()
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

        # Convert raw bytes -> (N, 1) int16
        chunk = np.frombuffer(payload, dtype=np.int16).reshape(-1, 1)
        return chunk

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
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
