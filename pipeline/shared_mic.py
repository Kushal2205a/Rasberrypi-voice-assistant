"""
shared_mic.py — Single sounddevice InputStream shared between wake-word detection
and the session recorder.

WHY THIS EXISTS
---------------
ALSA hw devices (e.g. hw:3,0 / USB PnP mic) allow only ONE open stream at a
time.  Opening a second PortAudio stream while the first is still alive —
even milliseconds after closing it — raises "Device unavailable [-9985]".

The fix: keep ONE stream open for the entire lifetime of the process and
switch its routing between two modes:

  WAKE mode    → downsample 44100→16kHz, feed OpenWakeWord model
  SESSION mode → pass raw int16 chunks to StreamingRecorder.chunk_queue

Mode switching is a single flag flip — no stream teardown, no ALSA close,
no race condition.
"""

from __future__ import annotations

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
    """
    One sounddevice InputStream, two routing modes.

    Parameters
    ----------
    device        : sounddevice device index (int) or name.  None = default.
    sample_rate   : Native capture rate matching the recorder (e.g. 44100).
    chunk_duration: Recorder chunk size in seconds (e.g. 0.6).
    oww_chunk_ms  : OWW inference chunk in ms — 80 ms is optimal for OWW.
    """

    WAKE    = "wake"
    SESSION = "session"

    _OWW_SR = 16000   # OpenWakeWord always expects 16 kHz

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

        # --- OWW geometry (at 16 kHz) ---
        self._oww_chunk_samples_16k = int(self._OWW_SR * oww_chunk_ms / 1000)
        # How many native samples correspond to one 80-ms OWW chunk
        self._oww_chunk_samples_nat = int(self._sample_rate * oww_chunk_ms / 1000)
        self._oww_buf               = bytearray()
        self._oww_buf_lock          = threading.Lock()

        # Raw audio queue for wake processing (filled by callback, drained by thread)
        self._wake_raw_q: deque = deque(maxlen=128)
        self._wake_thread: Optional[threading.Thread] = None

        # --- resampling ratio (for wake mode) ---
        g = math.gcd(self._sample_rate, self._OWW_SR)
        self._rs_up   = self._OWW_SR     // g
        self._rs_down = self._sample_rate // g

        # --- routing ---
        self._mode      = self.WAKE
        self._mode_lock = threading.Lock()

        self._detector: Optional[WakeWordDetector] = None
        self._recorder: Optional[StreamingRecorder] = None

        # --- stream lifecycle ---
        self._stream:  Optional[sd.InputStream] = None
        self._running  = False
        self._cb_blocksize = max(256, int(self._sample_rate * 0.10))

        self._last_status_log = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach_detector(self, detector: "WakeWordDetector") -> None:
        self._detector = detector

    def attach_recorder(self, recorder: "StreamingRecorder") -> None:
        self._recorder = recorder

    def set_mode(self, mode: str) -> None:
        """Switch routing.  Thread-safe, instantaneous."""
        if mode not in (self.WAKE, self.SESSION):
            raise ValueError(f"mode must be 'wake' or 'session', got {mode!r}")
        with self._mode_lock:
            self._mode = mode
            # Flush buffers so stale audio doesn't bleed between modes
            self._oww_buf.clear()
            self._wake_raw_q.clear()  # discard any audio queued before the switch

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

    # ------------------------------------------------------------------
    # Internal callback (runs in PortAudio thread — keep it fast)
    # ------------------------------------------------------------------

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
            # Enqueue raw bytes for background wake processing thread (no work in callback)
            self._wake_raw_q.append(raw)

    # ------------------------------------------------------------------
    # Recorder feeding
    # ------------------------------------------------------------------

    def _feed_recorder(self, raw: bytes) -> None:
        if self._recorder is None:
            return
        chunk = np.frombuffer(raw, dtype=np.int16)
        self._recorder.inject_audio(chunk)

    # ------------------------------------------------------------------
    # Wake-word feeding — background thread (resampling off the hot path)
    # ------------------------------------------------------------------

    def _wake_process_loop(self) -> None:
        """Drain raw audio queue, resample to 16 kHz, push to OWW detector."""
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