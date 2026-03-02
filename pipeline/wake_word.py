from __future__ import annotations

import threading
import time
import os
import collections
from typing import Callable, Optional, List

import numpy as np

try:
    from openwakeword.model import Model as OWWModel
    _HAVE_OWW = True
except ImportError:
    _HAVE_OWW = False

try:
    import sounddevice as sd
    _HAVE_SD = True
except ImportError:
    _HAVE_SD = False


DEFAULT_KEYWORD   = os.getenv("WAKE_WORD", "hey_jarvis")
DEFAULT_THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.5"))
TRIGGER_FRAMES    = int(os.getenv("WAKE_TRIGGER_FRAMES", "1"))

BUILTIN_KEYWORDS: List[str] = [
    "hey_jarvis",
    "alexa",
    "hey_mycroft",
    "hey_rhasspy",
    "ok_nabu",
    "hey_mr_robot",
]


class WakeWordDetector:

    def __init__(
        self,
        keyword:   str                          = DEFAULT_KEYWORD,
        threshold: float                        = DEFAULT_THRESHOLD,
        on_wake:   Optional[Callable[[], None]] = None,
        cooldown:  float                        = 3.0,
        device:    Optional[object]             = None,   # ignored — SharedMicStream controls the device
        chunk_ms:  int                          = 80,     # ignored — kept for API compatibility
    ) -> None:
        if not _HAVE_OWW:
            raise RuntimeError(
                "openwakeword is not installed.\n"
                "Run:  pip install openwakeword --break-system-packages\n"
                "Then: python -c \"import openwakeword; "
                "openwakeword.utils.download_models()\""
            )

        self._keyword   = keyword
        self._threshold = float(threshold)
        self._on_wake   = on_wake
        self._cooldown  = float(cooldown)

        self._model:       Optional[OWWModel] = None
        self._running      = False
        self._last_trigger = 0.0
        self._model_lock   = threading.Lock()

        self._score_window: collections.deque = collections.deque(
            maxlen=max(1, TRIGGER_FRAMES)
        )

        self._audio_q:    collections.deque = collections.deque(maxlen=64)
        self._audio_lock  = threading.Lock()
        self._proc_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._model   = self._load_model()
        self._running = True
        self._proc_thread = threading.Thread(
            target=self._process_loop,
            name="WakeWordProcessor",
            daemon=True,
        )
        self._proc_thread.start()
        print(f"[WakeWord] Listening for '{self._keyword}' "
              f"(threshold={self._threshold:.2f}) ...")

    def stop(self) -> None:
        self._running = False
        if self._proc_thread:
            self._proc_thread.join(timeout=2.0)
            self._proc_thread = None
        with self._model_lock:
            self._model = None
        print("[WakeWord] Stopped.")

    def push_audio(self, chunk_16k: np.ndarray) -> None:
        if not self._running:
            return
        with self._audio_lock:
            self._audio_q.append(chunk_16k)

    def reset_state(self) -> None:
        with self._audio_lock:
            self._audio_q.clear()

        with self._model_lock:
            if self._model is not None:
                try:
                    self._model.reset()  # NOTE: clears internal mel-spectrogram buffer; older OWW may not have this
                except Exception:
                    pass

        self._score_window.clear()
        self._last_trigger = time.time()
        print("[WakeWord] State reset — ready for next session.")

    @staticmethod
    def list_devices() -> None:
        if _HAVE_SD:
            print(sd.query_devices())
        else:
            print("sounddevice not installed.")

    @staticmethod
    def available_models() -> List[str]:
        try:
            import openwakeword as _oww
            return list(getattr(_oww, "MODELS", {}).keys())
        except Exception:
            return BUILTIN_KEYWORDS

    def _load_model(self) -> OWWModel:
        if os.path.isfile(self._keyword):
            print(f"[WakeWord] Loading custom model: {self._keyword}")
        else:
            print(f"[WakeWord] Loading model '{self._keyword}' "
                  f"(auto-downloading if needed)...")
        model = OWWModel(
            wakeword_models=[self._keyword],
            inference_framework="onnx",
        )
        print("[WakeWord] Model ready.")
        return model

    def _process_loop(self) -> None:
        while self._running:
            chunk = None
            with self._audio_lock:
                if self._audio_q:
                    chunk = self._audio_q.popleft()

            if chunk is None:
                time.sleep(0.005)
                continue

            with self._model_lock:
                model = self._model
            if model is None:
                continue

            try:
                prediction = model.predict(chunk)
                score = float(max(prediction.values())) if prediction else 0.0
            except Exception:
                continue

            self._score_window.append(score >= self._threshold)

            if (all(self._score_window)
                    and len(self._score_window) == self._score_window.maxlen):
                now = time.time()
                if now - self._last_trigger >= self._cooldown:
                    self._last_trigger = now
                    self._score_window.clear()
                    print(f"[WakeWord] '{self._keyword}' detected! "
                          f"(score={score:.2f})")
                    if callable(self._on_wake):
                        try:
                            self._on_wake()
                        except Exception as exc:
                            print(f"[WakeWord] on_wake error: {exc}")