# pipeline/llm_ollama_adapter.py
import os, json, requests
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Dict, Callable
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
        self.model = model or os.getenv("OLLAMA_MODEL", MODEL)
        self.host = (host or os.getenv("OLLAMA_HOST", "http://0.0.0.0:11434")).rstrip("/")
        self.keep_alive = keep_alive or os.getenv("LLM_KEEP_ALIVE", "30m")
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", num_ctx))
        self.num_thread = int(os.getenv("OLLAMA_NUM_THREAD", num_thread))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "32"))  # cap for speed
        self.chat_url = f"{self.host}/api/chat"

        self._history: List[Dict[str, str]] = []
        self._history_max_turns = history_max_turns
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._shutdown = False

        self._stream_cb: Optional[Callable[[str], None]] = None
        self._stream_ref_ts: Optional[float] = None

    def set_stream_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        self._stream_cb = cb

    def set_stream_context(self, ref_ts: Optional[float]) -> None:
        self._stream_ref_ts = ref_ts  # just stored for metrics on your side

    def _trim_history(self) -> None:
        max_msgs = 2 * self._history_max_turns
        if self._history_max_turns <= 0:
            self._history.clear()
        elif len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def _chat_stream(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": self._history + [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {
                "keep_alive": self.keep_alive,
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
                    if self._stream_cb:
                        try:
                            self._stream_cb(msg)
                        except Exception:
                            pass
                if data.get("done"):
                    break

        content = "".join(acc).strip()
        self._history.append({"role": "user", "content": prompt})
        self._history.append({"role": "assistant", "content": content})
        self._trim_history()
        return content

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
