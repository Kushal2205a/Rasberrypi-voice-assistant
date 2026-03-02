"""
stt_whisper_server.py — whisper.cpp HTTP server STT adapter.

WHY THIS IS FASTER THAN THE SUBPROCESS APPROACH
------------------------------------------------
stt_whisper_cpp.py spawns a new whisper-cli process on every request.
That means: fork + exec + load model from disk + inference + exit = ~2-4s.

In server mode whisper.cpp loads the model ONCE and keeps it in RAM.
Each request is just: send WAV over HTTP + inference = ~300-600ms for tiny.en.

SETUP (one-time, run in terminal)
----------------------------------
1. Check if the server binary exists:
       ls ~/whisper.cpp/build/bin/

   You need either 'whisper-server' or 'server'. If missing, rebuild:
       cd ~/whisper.cpp
       cmake -B build -DWHISPER_BUILD_SERVER=ON
       cmake --build build --config Release -j4

2. Start the server (run this in a separate terminal or as a systemd service):
       ~/whisper.cpp/build/bin/whisper-server \
           -m ~/whisper.cpp/models/ggml-tiny.en.bin \
           -t 3 --port 8080 --host 127.0.0.1

   Or with the base.en model for better accuracy:
       ~/whisper.cpp/build/bin/whisper-server \
           -m ~/whisper.cpp/models/ggml-base.en.bin \
           -t 3 --port 8080 --host 127.0.0.1

3. Run the assistant:
       STT_BACKEND=whisper_server python main.py

AUTOSTART (optional — add to /etc/rc.local or a systemd unit)
--------------------------------------------------------------
       nohup ~/whisper.cpp/build/bin/whisper-server \
           -m ~/whisper.cpp/models/ggml-tiny.en.bin \
           -t 3 --port 8080 --host 127.0.0.1 \
           > /tmp/whisper-server.log 2>&1 &
"""

from __future__ import annotations

import io
import math
import threading
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import deque
import numpy as np
import requests

try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from config import SAMPLE_RATE, WHISPER_SERVER_URL


class WhisperServerSTT:
    """
    Drop-in replacement for PersistentVoskSTT using the whisper.cpp HTTP server.

    IS_STREAMING = False — audio accumulates via submit_chunk(), then
    finalize() sends the whole buffer to the server in one shot.
    """

    IS_STREAMING = False

    def __init__(
        self,
        server_url:  str = WHISPER_SERVER_URL,
        sample_rate: int = SAMPLE_RATE,
        language:    str = "en",
        temperature: float = 0.0,
        num_workers: int   = 2,    # compat, ignored
        **_ignore: Any,
    ) -> None:
        self._url        = server_url.rstrip("/") + "/inference"
        self._language   = language
        self._temperature = float(temperature)

        self.sample_rate = int(sample_rate)
        self._target_sr  = 16000

        g = math.gcd(self.sample_rate, self._target_sr)
        self._rs_up   = self._target_sr  // g
        self._rs_down = self.sample_rate // g

        self._buf      = bytearray()
        self._buf_lock = threading.Lock()
        self._lock     = threading.Lock()
        self.finalizing = False
        self.executor   = ThreadPoolExecutor(max_workers=2)

        # Test connectivity
        self._check_server()

    def _check_server(self) -> None:
        base = self._url.rsplit("/inference", 1)[0]
        try:
            r = requests.get(base, timeout=2)
            print(f"[STT][whisper-server] Connected at {base}")
        except Exception:
            print(
                f"[STT][whisper-server] WARNING: server not reachable at {base}\n"
                f"  Start it with:\n"
                f"  ~/whisper.cpp/build/bin/whisper-server "
                f"-m ~/whisper.cpp/models/ggml-tiny.en.bin "
                f"-t 3 --port 8080 --host 127.0.0.1"
            )

    # ------------------------------------------------------------------
    # Orchestrator interface
    # ------------------------------------------------------------------

    def empty_future(self, chunk_id: int) -> Future:
        f: Future = Future()
        f.set_result({"chunk_id": chunk_id, "text": "", "is_final": False})
        return f

    def submit_chunk(self, audio_chunk: np.ndarray, chunk_id: int) -> Future:
        audio_bytes = np.ascontiguousarray(audio_chunk, dtype=np.int16).tobytes()
        with self._buf_lock:
            self._buf.extend(audio_bytes)
        return self.empty_future(chunk_id)

    def finalize(self, chunk_id: int, mark_final: bool = True) -> Future:
        with self._lock:
            self.finalizing = True
        return self.executor.submit(self._transcribe_worker, chunk_id, mark_final)

    def reset(self) -> None:
        with self._buf_lock:
            self._buf.clear()
        with self._lock:
            self.finalizing = False

    def shutdown(self) -> None:
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.reset()

    def decode_grammar(self, audio_chunk: np.ndarray, phrases: List[str]) -> str:
        if audio_chunk is None or audio_chunk.size == 0:
            return ""
        audio_bytes = np.ascontiguousarray(audio_chunk, dtype=np.int16).tobytes()
        prompt = ", ".join(phrases)
        return self._call_server(audio_bytes, initial_prompt=prompt) or ""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resample(self, audio_bytes: bytes) -> bytes:
        if not audio_bytes:
            return b""
        if self.sample_rate == self._target_sr:
            return audio_bytes
        src = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if _HAVE_SCIPY:
            dst = resample_poly(src, self._rs_up, self._rs_down)
        else:
            ratio = self._target_sr / float(self.sample_rate)
            xi = np.arange(0, len(src), 1.0 / ratio, dtype=np.float32)
            dst = np.interp(np.arange(len(src), dtype=np.float32), xi, src).astype(np.float32)
        return np.clip(dst, -32768, 32767).astype(np.int16).tobytes()

    def _make_wav(self, pcm_16k: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._target_sr)
            wf.writeframes(pcm_16k)
        return buf.getvalue()

    def _call_server(
        self,
        audio_bytes: bytes,
        initial_prompt: Optional[str] = None,
    ) -> str:
        pcm_16k  = self._resample(audio_bytes)
        if not pcm_16k:
            return ""
        wav_data = self._make_wav(pcm_16k)

        data = {
            "temperature":     str(self._temperature),
            "temperature_inc": "0.0",
            "response_format": "json",
        }
        if self._language:
            data["language"] = self._language
        if initial_prompt:
            data["prompt"] = initial_prompt

        try:
            resp = requests.post(
                self._url,
                files={"file": ("audio.wav", wav_data, "audio/wav")},
                data=data,
                timeout=15,
            )
            resp.raise_for_status()
            j = resp.json()

            # whisper.cpp server returns {"text": "..."}
            text = (j.get("text") or "").strip()

            # Strip leading/trailing whitespace artifacts whisper sometimes adds
            text = text.strip("() \n")

            # Filter common silence hallucinations
            hallucinations = {
                "thank you", "thanks for watching", "thanks",
                "you", "bye", "bye-bye", ".", "[blank_audio]", "(blank audio)",
            }
            if text.lower().rstrip(".!?,") in hallucinations:
                return ""

            if text:
                print(f"[STT][whisper-server] → '{text}'")
            return text

        except requests.exceptions.ConnectionError:
            print("[STT][whisper-server] Connection refused — is the server running?")
            return ""
        except Exception as exc:
            print(f"[STT][whisper-server] Error: {exc}")
            return ""

    def _transcribe_worker(self, chunk_id: int, mark_final: bool) -> Dict[str, Any]:
        try:
            with self._buf_lock:
                audio_bytes = bytes(self._buf)
                self._buf.clear()
            text = self._call_server(audio_bytes) if audio_bytes else ""
            return {"chunk_id": chunk_id, "text": text, "is_final": bool(mark_final)}
        finally:
            with self._lock:
                self.finalizing = False