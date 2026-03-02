from __future__ import annotations

from typing import Optional, Any, Dict, List
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
import json
import math

import numpy as np

try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    import webrtcvad
    _HAVE_VAD = True
except Exception:
    _HAVE_VAD = False

try:
    from vosk import Model as VoskModel, KaldiRecognizer
except Exception as e:
    VoskModel = None
    KaldiRecognizer = None
    _vosk_import_err = e

from pathlib import Path
from config import SAMPLE_RATE
try:
    from config import VOSK_MODEL_PATH
except Exception:
    VOSK_MODEL_PATH = Path.home() / "models" / "vosk-model-en-in-0.5"

# NOTE: singleton — shared across all PersistentVoskSTT instances to avoid repeated disk loads
_SHARED_MODEL = None
_SHARED_LOCK = threading.Lock()


def _get_vosk_model(model_path: Path) -> VoskModel:
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        with _SHARED_LOCK:
            if _SHARED_MODEL is None:
                _SHARED_MODEL = VoskModel(str(model_path))
                print(f"[STT][VOSK] Loaded model at {model_path}")
    return _SHARED_MODEL


@dataclass
class _State:
    recognizer: KaldiRecognizer
    stable_text: str = ""
    last_partial: str = ""


class PersistentVoskSTT:
    IS_STREAMING = True

    def __init__(
        self,
        num_workers: int = 2,
        sample_rate: int = SAMPLE_RATE,
        emit_partials: bool = True,
        vad_aggressiveness: int = 2,
        min_partial_interval_ms: int = 120,
        model_path: Optional[Path] = None,
        **_ignore: Any,
    ) -> None:
        if VoskModel is None or KaldiRecognizer is None:
            raise RuntimeError(f"vosk is not installed: {_vosk_import_err}")

        self.executor = ThreadPoolExecutor(max_workers=max(2, num_workers))

        self.sample_rate = int(sample_rate)
        self._target_sr = 16000

        # NOTE: pre-computed rational resample ratio avoids repeated gcd calls per chunk
        g = math.gcd(self.sample_rate, self._target_sr)
        self._rs_up   = self._target_sr // g
        self._rs_down = self.sample_rate // g

        self.emit_partials = bool(emit_partials)
        self.finalizing = False

        self._model = _get_vosk_model(Path(model_path or VOSK_MODEL_PATH))
        self._state = _State(KaldiRecognizer(self._model, self._target_sr))

        self._vad = (
            webrtcvad.Vad(vad_aggressiveness)
            if (_HAVE_VAD and vad_aggressiveness and vad_aggressiveness > 0)
            else None
        )
        self._frame_ms = 20
        self._bytes_per_sample = 2

        self._last_partial_time = 0.0
        self._min_partial_interval = (min_partial_interval_ms or 0) / 1000.0

        # NOTE: single-flight gate — KaldiRecognizer is not thread-safe
        self._lock = threading.Lock()
        self._inflight = False

        self._pending = bytearray()

        self._vad_in_speech = False
        self._vad_speech_frames = 0
        self._vad_silence_frames = 0

        self._vad_preroll = bytearray()
        self._vad_preroll_ms = 200
        self._vad_preroll_frames = max(0, int(self._vad_preroll_ms / self._frame_ms))

        self._vad_end_silence_ms = 700
        self._vad_end_silence_frames = max(1, int(self._vad_end_silence_ms / self._frame_ms))

        self._vad_min_speech_ms = 60
        self._vad_min_speech_frames = max(1, int(self._vad_min_speech_ms / self._frame_ms))

    def empty_future(self, chunk_id: int) -> Future:
        f: Future = Future()
        f.set_result({"chunk_id": chunk_id, "text": "", "is_final": False})
        return f

    def _resample_to_16k(self, audio_bytes: bytes) -> bytes:
        if not audio_bytes:
            return b""
        if self.sample_rate == self._target_sr:
            return audio_bytes

        src = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        if _HAVE_SCIPY:
            dst = resample_poly(src, self._rs_up, self._rs_down)
        else:
            ratio = self._target_sr / float(self.sample_rate)
            x = np.arange(src.shape[0], dtype=np.float32)
            xi = np.arange(0, src.shape[0], 1.0 / ratio, dtype=np.float32)
            if xi.size == 0:
                return b""
            dst = np.interp(xi, x, src)

        dst = np.clip(dst, -32768.0, 32767.0).astype(np.int16)
        return dst.tobytes()

    def decode_grammar(self, audio_chunk: np.ndarray, phrases: List[str]) -> str:
        if audio_chunk is None or audio_chunk.size == 0 or not phrases:
            return ""
        audio_bytes = np.ascontiguousarray(audio_chunk, dtype=np.int16).tobytes()
        audio_16k = self._resample_to_16k(audio_bytes)
        if not audio_16k:
            return ""

        grammar = json.dumps(phrases)
        try:
            recog = KaldiRecognizer(self._model, self._target_sr, grammar)
        except TypeError:
            recog = KaldiRecognizer(self._model, self._target_sr)
            try:
                recog.SetGrammar(grammar)
            except Exception:
                pass

        recog.AcceptWaveform(audio_16k)
        try:
            res = json.loads(recog.FinalResult() or "{}")
        except Exception:
            return ""
        return (res.get("text") or "").strip()

    def _feed_and_read(self, audio_16k: bytes) -> Dict[str, Any]:
        recog = self._state.recognizer
        frame_bytes = int(self._target_sr * (self._frame_ms / 1000.0)) * self._bytes_per_sample
        had_final = False

        for i in range(0, len(audio_16k), frame_bytes):
            frame = audio_16k[i:i + frame_bytes]
            if len(frame) < frame_bytes:
                break
            if recog.AcceptWaveform(frame):
                had_final = True

        if had_final:
            res = json.loads(recog.Result() or "{}")
            seg = (res.get("text") or "").strip()
            if seg:
                if self._state.stable_text:
                    self._state.stable_text += " "
                self._state.stable_text += seg
                had_final = True

        partial_out = ""
        if not had_final and self.emit_partials:
            now = time.time()
            if (now - self._last_partial_time) >= self._min_partial_interval:
                pres = json.loads(recog.PartialResult() or "{}")
                partial = (pres.get("partial") or "").strip()
                full = (self._state.stable_text + (" " + partial if partial else "")).strip()
                if full and full != self._state.last_partial:
                    partial_out = full
                    self._state.last_partial = full
                    self._last_partial_time = now

        return {"partial_text": partial_out, "had_final": had_final}

    def _partial_worker(self, chunk_id: int, audio_bytes: bytes) -> Dict[str, Any]:
        try:
            audio_16k = self._resample_to_16k(audio_bytes)
            out = self._feed_and_read(audio_16k)

            if out["had_final"]:
                return {"chunk_id": chunk_id, "text": self._state.stable_text, "is_final": False}
            if out["partial_text"]:
                return {"chunk_id": chunk_id, "text": out["partial_text"], "is_final": False}
            return {"chunk_id": chunk_id, "text": "", "is_final": False}
        finally:
            with self._lock:
                self._inflight = False

    def _finalize_worker(self, chunk_id: int, mark_final: bool) -> Dict[str, Any]:
        try:
            with self._lock:
                leftover = bytes(self._pending)
                self._pending.clear()

            if leftover:
                try:
                    audio_16k = self._resample_to_16k(leftover)
                    self._feed_and_read(audio_16k)
                except Exception as e:
                    print(f"[STT][VOSK] Error feeding leftover audio during finalize: {e}")

            recog = self._state.recognizer
            try:
                fres = json.loads(recog.FinalResult() or "{}")
            except Exception:
                fres = {}

            final_seg = (fres.get("text") or "").strip()
            text = self._state.stable_text
            if final_seg:
                text = (text + " " + final_seg).strip() if text else final_seg

            self._state = _State(KaldiRecognizer(self._model, self._target_sr))
            return {"chunk_id": chunk_id, "text": text, "is_final": bool(mark_final)}
        finally:
            with self._lock:
                self.finalizing = False
                self._inflight = False

    def submit_chunk(self, audio_chunk: np.ndarray, chunk_id: int) -> Future:
        audio_bytes = np.ascontiguousarray(audio_chunk, dtype=np.int16).tobytes()
        with self._lock:
            if self.finalizing or self._inflight:
                self._pending.extend(audio_bytes)
                return self.empty_future(chunk_id)

            if self._pending:
                audio_bytes = bytes(self._pending) + audio_bytes
                self._pending.clear()

            self._inflight = True

        return self.executor.submit(self._partial_worker, chunk_id, audio_bytes)

    def finalize(self, chunk_id: int, mark_final: bool = True) -> Optional[Future]:
        with self._lock:
            self.finalizing = True
        return self.executor.submit(self._finalize_worker, chunk_id, mark_final)

    def reset(self) -> None:
        with self._lock:
            self._state = _State(KaldiRecognizer(self._model, self._target_sr))
            self._inflight = False
            self.finalizing = False
            self._pending.clear()
            self._last_partial_time = 0.0

    def shutdown(self) -> None:
        # NOTE: recycles pool so instance can be reused across sessions without a full rebuild
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass
        self.executor = ThreadPoolExecutor(max_workers=max(2, 2))
        self.reset()