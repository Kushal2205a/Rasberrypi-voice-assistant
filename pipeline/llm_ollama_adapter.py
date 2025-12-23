import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Dict, Callable

import requests

from config import MODEL


class OllamaStreamingLLM:
    """
    Streaming adapter compatible with your existing API:
      - set_stream_callback(fn)
      - set_stream_context(ts)
      - process_incremental(text, is_final) -> Optional[Future[str]]
      - shutdown()
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        keep_alive: Optional[str] = None,
        num_ctx: int = 256,
        num_thread: int = 4,
        history_max_turns: int = 2,
    ) -> None:
        self.model = (model or os.getenv("OLLAMA_MODEL", MODEL)).strip()
        self.host = (host or os.getenv("OLLAMA_HOST", "http://0.0.0.0:11434")).rstrip("/")
        self.keep_alive = keep_alive or os.getenv("LLM_KEEP_ALIVE", "30m")

        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", str(num_ctx)))
        self.num_thread = int(os.getenv("OLLAMA_NUM_THREAD", str(num_thread)))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "32"))  # cap for speed

        self.chat_url = f"{self.host}/api/chat"

        self._history: List[Dict[str, str]] = []
        self._history_max_turns = int(history_max_turns)

        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._shutdown = False

        self._stream_cb: Optional[Callable[[str], None]] = None
        self._stream_ref_ts: Optional[float] = None

    # ------------------- Public controls -------------------

    def set_stream_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        self._stream_cb = cb

    def set_stream_context(self, ref_ts: Optional[float]) -> None:
        self._stream_ref_ts = ref_ts

    def get_model(self) -> str:
        with self._lock:
            return self.model

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def set_model(self, model: str, reset_history: bool = True) -> None:
        model = (model or "").strip()
        if not model:
            return
        with self._lock:
            self.model = model
            if reset_history:
                self._history.clear()

    # ------------------- Internals -------------------

    def _trim_history(self) -> None:
        max_msgs = 2 * self._history_max_turns
        if self._history_max_turns <= 0:
            self._history.clear()
        elif len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def _chat_stream(self, prompt: str) -> str:
        with self._lock:
            model = self.model
            messages = list(self._history) + [{"role": "user", "content": prompt}]

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_thread": self.num_thread,
                "num_predict": self.num_predict,
            },
        }

        acc: List[str] = []
        with requests.post(self.chat_url, json=payload, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                msg = (data.get("message") or {}).get("content", "")
                if msg:
                    acc.append(msg)
                    cb = self._stream_cb
                    if cb:
                        try:
                            cb(msg)
                        except Exception:
                            pass
                if data.get("done"):
                    break

        content = "".join(acc).strip()

        with self._lock:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": content})
            self._trim_history()

        return content

    # ------------------- Existing pipeline API -------------------

    def process_incremental(self, text: str, is_final: bool = False) -> Optional[Future]:
        if not is_final or self._shutdown or getattr(self._pool, "_shutdown", False):
            return None
        return self._pool.submit(self._chat_stream, text)

    def shutdown(self) -> None:
        self._shutdown = True
        try:
            self._pool.shutdown(wait=False)
        except Exception:
            pass
