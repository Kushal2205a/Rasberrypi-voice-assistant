# pipeline/llm_ollama.py
import json
import os
import requests
from typing import Generator, Iterable, Optional

DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "10m")  # keep model warm between calls

class OllamaLLM:
    """
    Minimal streaming chat client for Ollama.
    Use .stream(prompt) to yield tokens as soon as they arrive.
    """
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self.model = model or DEFAULT_MODEL
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.chat_url = f"{self.host}/api/chat"

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Streams incremental tokens from Ollama as they are generated.
        Yields small string chunks; concatenate them for the full text.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"keep_alive": KEEP_ALIVE},
        }
        with requests.post(self.chat_url, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                # Each line is a JSON object with partial text under message.content
                data = json.loads(line.decode("utf-8"))
                msg = data.get("message", {})
                piece = msg.get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break

    def generate(self, prompt: str) -> str:
        """
        Non-streaming convenience method if you ever need the whole answer.
        """
        buf = []
        for tok in self.stream(prompt):
            buf.append(tok)
        return "".join(buf)
