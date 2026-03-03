import os
import re
import json
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Dict, Callable

import requests

from config import MODEL

# ---------------------------------------------------------------------------
# Complexity tiers — tuned for Pi latency vs. response completeness trade-off
# ---------------------------------------------------------------------------
#
# SIMPLE  : one-word / one-number answers, yes/no, single facts
# MEDIUM  : a sentence or two, definitions, short how-tos
# COMPLEX : multi-step explanations, lists, code, comparisons
# MAX cap : hard ceiling so runaway prompts never stall the Pi
#
_PREDICT_SIMPLE  = int(os.getenv("OLLAMA_NUM_PREDICT_SIMPLE",  "48"))
_PREDICT_MEDIUM  = int(os.getenv("OLLAMA_NUM_PREDICT_MEDIUM", "120"))
_PREDICT_COMPLEX = int(os.getenv("OLLAMA_NUM_PREDICT_COMPLEX", "280"))
_PREDICT_MAX     = int(os.getenv("OLLAMA_NUM_PREDICT_MAX",     "400"))

# ---------------------------------------------------------------------------
# Pattern sets — ordered from most to least specific
# ---------------------------------------------------------------------------

# Prompts that almost always need only a word, number, or single sentence
_SIMPLE_PATTERNS = re.compile(
    r"\b(what(?:'s| is) the (?:time|date|day|weather|capital|population|age|name)|"
    r"who (?:is|was|are) (?:the )?(?:president|ceo|founder|author|inventor)|"
    r"how (?:many|much|old|tall|far|long)|"
    r"(?:yes or no|true or false)|"
    r"(?:define|spell|translate)\b|"
    r"(?:what does \w+ stand for)|"
    r"(?:convert \d))",
    re.IGNORECASE,
)

# Open-ended / creative prompts — short phrasing but demand a *long* response.
# These must be checked BEFORE the word-count fallback, which would wrongly
# classify "tell me a story" (5 words) as SIMPLE.
_CREATIVE_PATTERNS = re.compile(
    r"\b("
    # storytelling
    r"tell (?:me )?(?:a |an )?(?:story|tale|joke|riddle|poem|fable|legend|myth)|"
    r"make up (?:a |an )?(?:story|tale|character|world)|"
    r"write (?:me )?(?:a |an )?(?:story|poem|song|essay|letter|speech|script|haiku)|"
    r"(?:compose|create|generate|give me) (?:a |an )?(?:story|poem|song|rap|essay|speech)|"
    # open-ended "can you …" / "could you …" with creative verb
    r"(?:can|could) you (?:tell|write|make|describe|explain|give)|"
    # roleplay / pretend
    r"(?:pretend|act|roleplay|imagine|suppose)\b|"
    # continuation
    r"(?:continue|finish|complete) (?:the )?(?:story|sentence|poem)"
    r")",
    re.IGNORECASE,
)

# Prompts that expect a multi-sentence structured response
_COMPLEX_PATTERNS = re.compile(
    r"\b(explain|describe|compare|contrast|summarize|list|enumerate|"
    r"write|generate|create|draft|code|implement|step[s]? (?:to|for)|"
    r"how (?:do|does|did|can|should|would)|"
    r"what (?:are|were) (?:the )?(?:reasons|benefits|drawbacks|pros|cons|steps|ways)|"
    r"give me (?:an? )?(?:example|overview|breakdown|summary))",
    re.IGNORECASE,
)


def _estimate_num_predict(prompt: str) -> int:
    """
    Heuristically estimate a good num_predict cap for this prompt.

    Priority order:
      1. Creative / open-ended intent  → COMPLEX  (catches "tell me a story")
      2. Known-simple factual patterns → SIMPLE
      3. Known-complex structural cues → MEDIUM or COMPLEX by word count
      4. Word-count fallback           → SIMPLE / MEDIUM / COMPLEX
      5. Hard clamp to _PREDICT_MAX
    """
    stripped = prompt.strip()
    words    = stripped.split()
    n_words  = len(words)

    # 1. Creative intent always gets the full budget — short phrasing, long output
    if _CREATIVE_PATTERNS.search(stripped):
        return min(_PREDICT_COMPLEX, _PREDICT_MAX)

    # 2. Structured / explanatory request — checked BEFORE the short-question
    #    shortcut so "How do I make pasta?" isn't collapsed to SIMPLE.
    if _COMPLEX_PATTERNS.search(stripped):
        budget = _PREDICT_COMPLEX if n_words > 10 else _PREDICT_MEDIUM
        return min(budget, _PREDICT_MAX)

    # 3. Factual one-liners
    if _SIMPLE_PATTERNS.search(stripped):
        return _PREDICT_SIMPLE

    # Very short question with no remaining complex signal → quick fact
    if n_words <= 6 and stripped.endswith("?"):
        return _PREDICT_SIMPLE

    # 4. Word-count fallback
    if n_words <= 8:
        return _PREDICT_SIMPLE
    if n_words <= 20:
        return _PREDICT_MEDIUM
    return min(_PREDICT_COMPLEX, _PREDICT_MAX)


class OllamaStreamingLLM:

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

        # NOTE: num_predict is now computed per-request via _estimate_num_predict().
        # This env-var acts as a hard override when set (e.g. for testing).
        self._fixed_num_predict: Optional[int] = (
            int(os.getenv("OLLAMA_NUM_PREDICT"))
            if os.getenv("OLLAMA_NUM_PREDICT")
            else None
        )

        self.chat_url = f"{self.host}/api/chat"

        self._history: List[Dict[str, str]] = []
        self._history_max_turns = int(history_max_turns)

        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._shutdown = False

        self._stream_cb: Optional[Callable[[str], None]] = None
        self._stream_ref_ts: Optional[float] = None

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

    def _trim_history(self) -> None:
        max_msgs = 2 * self._history_max_turns
        if self._history_max_turns <= 0:
            self._history.clear()
        elif len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def _chat_stream(self, prompt: str) -> str:
        # Resolve num_predict: fixed override > per-request estimate
        num_predict = (
            self._fixed_num_predict
            if self._fixed_num_predict is not None
            else _estimate_num_predict(prompt)
        )

        with self._lock:
            model = self.model
            messages = list(self._history) + [{"role": "user", "content": prompt}]

        print(f"[LLM] num_predict={num_predict} for prompt ({len(prompt.split())} words)")

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_thread": self.num_thread,
                "num_predict": num_predict,
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

    def process_incremental(self, text: str, is_final: bool = False) -> Optional[Future]:
        if not is_final or self._shutdown or getattr(self._pool, "_shutdown", False):
            return None
        return self._pool.submit(self._chat_stream, text)

    def shutdown(self) -> None:
        # NOTE: recycles the pool rather than destroying it so history/model are preserved across sessions
        try:
            self._pool.shutdown(wait=False)
        except Exception:
            pass
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._shutdown = False