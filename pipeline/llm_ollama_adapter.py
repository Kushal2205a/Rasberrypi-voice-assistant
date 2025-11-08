# pipeline/llm_ollama_adapter.py
import os
import requests
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Dict

class OllamaStreamingLLM:
    """
    Adapter that matches your existing StreamingLLM API:
      - process_incremental(text, is_final) -> Optional[Future[str]]
      - shutdown()
    It calls Ollama's /api/chat (non-streaming per request) so your pipeline doesn't change.
    """
    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        keep_alive: Optional[str] = None,
        num_ctx: int = 512,
        num_thread: int = 4,
        history_max_turns: int = 6,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.keep_alive = keep_alive or os.getenv("LLM_KEEP_ALIVE", "30m")
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", num_ctx))
        self.num_thread = int(os.getenv("OLLAMA_NUM_THREAD", num_thread))
        self.chat_url = f"{self.host}/api/chat"

        # Keep minimal history so follow-ups feel coherent (fits your pipeline)
        self._history: List[Dict[str, str]] = []
        self._history_max_turns = history_max_turns

        self._pool = ThreadPoolExecutor(max_workers=1)

    def _trim_history(self) -> None:
        # Keep most recent 2*history_max_turns messages (user+assistant pairs)
        if self._history_max_turns <= 0:
            self._history.clear()
            return
        # messages are a flat list; 2 entries per turn
        max_msgs = 2 * self._history_max_turns
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def _chat_once(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": self._history + [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "keep_alive": self.keep_alive,
                "num_ctx": self.num_ctx,
                "num_thread": self.num_thread,
            },
        }
        r = requests.post(self.chat_url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json() if r.content else {}
        content = (data.get("message") or {}).get("content", "") if isinstance(data, dict) else ""
        # update short conversation history
        self._history.append({"role": "user", "content": prompt})
        self._history.append({"role": "assistant", "content": content})
        self._trim_history()
        return content.strip()

    def process_incremental(self, text: str, is_final: bool = False) -> Optional[Future]:
        # Only hit the model when utterance is final (matches your current LLM timing)
        if not is_final:
            return None
        return self._pool.submit(self._chat_once, text)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
