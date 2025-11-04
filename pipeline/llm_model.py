import time
import subprocess, shlex, re, os
from pathlib import Path

from Resource_Sampler import ResourceSampler
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Optional, List, Dict


def _parse_llm_timing(raw_text):
    """Best-effort extraction of an inference time from llama.cpp output."""
    if not raw_text:
        return None

    patterns = [
        r"inference(?:_| |-)?time[:=]?\s*([0-9]*\.?[0-9]+)\s*(ms|s)\b",
        r"model(?:_| |-)?time[:=]?\s*([0-9]*\.?[0-9]+)\s*(ms|s)\b",
        r"total(?:_| |-)?time[:=]?\s*([0-9]*\.?[0-9]+)\s*(ms|s)\b",
        r"elapsed[:=]?\s*([0-9]*\.?[0-9]+)\s*(ms|s)\b",
        r"([0-9]*\.?[0-9]+)\s*ms\s*per\s*token",
        r"([0-9]*\.?[0-9]+)\s*ms\b",
        r"([0-9]*\.?[0-9]+)\s*s\b",
    ]

    for pat in patterns:
        for match in re.finditer(pat, raw_text, flags=re.IGNORECASE):
            num = match.group(1)
            unit = match.group(2) if match.lastindex and match.lastindex >= 2 else None
            try:
                val = float(num)
                if unit and unit.lower().startswith("ms"):
                    val = val / 1000.0
                if val > 0:
                    return val
            except Exception:
                continue

    return None


# small TTS wrapper

def speak_text_timed(text: str, cmd=None):
    """
    Very small wrapper that runs a TTS command and returns elapsed seconds.
    Default uses `espeak text`. If you use a different TTS, pass `cmd` as a list,
    e.g. ['tts-cli', '--out', '/tmp/out.wav', text] or similar.
    """
    text = (text or "").strip()
    if not text:
        return 0.0
    if cmd is None:
        # default simple espeak call; replace if you need a different TTS
        cmd = ["espeak", text]
    else:
        # allow passing a format-string or a list; convert format-string to shell-safe list
        if isinstance(cmd, str):
            # user provided something like "mytts --text '{}'"
            safe = cmd.format(shlex.quote(text))
            cmd = shlex.split(safe)
        elif isinstance(cmd, (list, tuple)):
            # if list contains {} we substitute with text
            cmd = [c.format(text) if ("{}" in c or "{text}" in c) else c for c in cmd]
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        # swallow TTS errors but still return elapsed
        pass
    return time.time() - t0

def llama110(prompt_text: str,
             llama_cli_path: str = None,
             model_path: str = None,
             n_predict: int = 256,
             threads: int = 4,
             temperature: float = 0.7,
             sampler: ResourceSampler = None,
             tts_after: bool = False,
             tts_cmd = None,
             timeout_seconds: int = 240):
    

    # sensible defaults (do not overwrite your other model; use explicit path if you have one)
    if llama_cli_path is None:
        llama_cli_path = str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli")
    if model_path is None:
        model_path = str(Path.home() / "Downloads" / "llama2.c-stories110M-pruned50.Q3_K_M.gguf")

    exe = Path(llama_cli_path)
    model = Path(model_path)

    result = {
        "generated": "",
        "model_reply_time": 0.0,
        "model_inference_time": None,
        "tokens": 0,
        "tts_time": 0.0,
        "resource": {"peak_rss":0, "avg_rss":0, "peak_cpu":0.0, "avg_cpu":0.0, "samples":0},
        "raw_stdout": None,
        "raw_stderr": None,
    }

    if sampler is None:
        sampler = ResourceSampler(interval=0.05)  # local no-op if psutil missing

    if not exe.exists():
        raise FileNotFoundError(f"llama-cli binary not found at: {exe}")

    if not model.exists():
        raise FileNotFoundError(f"llama model not found at: {model}")

    # Build command (adjust flags to your llama-cli flavor if different)
    cmd = [
        str(exe),
        "-m", str(model),
        "-p", prompt_text,
        "-n", str(n_predict),
        "-t", str(threads),
        "--temp", str(temperature),
        "-no-cnv",       
        "--single-turn",
    ]

    # start sampling and call subprocess
    sampler.start()
    t_before = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        t_after = time.time()
    except subprocess.TimeoutExpired as e:
        t_after = time.time()
        sampler.stop()
        # return partial info
        result.update({
            "generated": "",
            "model_reply_time": t_after - t_before,
            "model_inference_time": None,
            "tokens": 0,
            "raw_stdout": getattr(e, "stdout", None) or "",
            "raw_stderr": getattr(e, "stderr", None) or "TIMEOUT",
        })
        return result

    sampler.stop()

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    raw_combined = stdout + ("\n" + stderr if stderr else "")
    result["raw_stdout"] = stdout
    result["raw_stderr"] = stderr

    # wall-clock subprocess time:
    model_reply_time = t_after - t_before
    result["model_reply_time"] = model_reply_time

    # Heuristic: if CLI echoes the prompt, remove the prompt from output to isolate generation
    generated = stdout
    if prompt_text and prompt_text.strip() and prompt_text.strip() in stdout:
        parts = stdout.split(prompt_text.strip())
        if len(parts) > 1:
            generated = parts[-1].strip()

    # Fallback: if stdout empty but stderr holds text
    if not generated and stderr:
        generated = stderr.strip()

    result["generated"] = generated
    result["tokens"] = len(generated.split())

    
    # Parse by looking for numbers followed by 'ms' or 's' near keywords.
    inference_time = _parse_llm_timing(raw_combined)

    result["model_inference_time"] = inference_time

    # Resource summary
    result["resource"] = sampler.summary()

    # Optionally run TTS and measure
    if tts_after:
        tts_elapsed = speak_text_timed(result["generated"], cmd=tts_cmd)
        result["tts_time"] = tts_elapsed

    return result

class StreamingLLM:
    """Accumulate recognized text and asynchronously invoke llama110 when ready."""

    def __init__(self, llama_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.context_buffer: List[str] = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.llama_kwargs = llama_kwargs or {}

        self.llama_kwargs.setdefault("n_predict", 12)
        self.llama_kwargs.setdefault("threads", os.cpu_count() or 4)
        self.llama_kwargs.setdefault("temperature", 0.6)
        self._closed = False

    def process_incremental(self, text_chunk: str, is_final: bool = False) -> Optional[Future]:
        if getattr(self, "_closed", False):
            return None
        chunk = (text_chunk or "").strip()
        if chunk:
            self.context_buffer.append(chunk)

        if not self.context_buffer and not chunk and not is_final:
            return None

        should_generate = False
        if is_final:
            should_generate = bool(self.context_buffer)
        elif self._should_respond():
            should_generate = True

        if not should_generate:
            return None

        prompt_text = " ".join(self.context_buffer).strip()
        self.context_buffer.clear()
        if not prompt_text:
            return None

        return self.executor.submit(self._generate_response, prompt_text)

    def _should_respond(self) -> bool:
        current = " ".join(self.context_buffer)
        if any(marker in current for marker in [".", "?", "!"]):
            return True
        return len(current.split()) >=6

    def _generate_response(self, text: str) -> str:
        prompt = f"Answer concisely: {text}".strip()
        call_kwargs = {
            "llama_cli_path": self.llama_kwargs.get("llama_cli_path"),
            "model_path": self.llama_kwargs.get("model_path"),

            "n_predict": self.llama_kwargs.get("n_predict", 256),
            "threads": self.llama_kwargs.get("threads", os.cpu_count() or 4),
            "temperature": self.llama_kwargs.get("temperature", 0.7),

            "sampler": self.llama_kwargs.get("sampler"),
            "tts_after": False,
            "tts_cmd": None,
            "timeout_seconds": self.llama_kwargs.get("timeout_seconds", 240),
        }

        try:
            result = llama110(prompt_text=prompt, **call_kwargs)
        except FileNotFoundError as exc:
            print(f"[LLM] {exc}")
            return "I could not load the local LLM."
        except Exception as exc:
            print(f"[LLM] Error invoking llama110: {exc}")
            return "I encountered an error while thinking about that."

        response = (result or {}).get("generated", "")
        if not response:
            fallback = (result or {}).get("raw_stdout") or ""
            response = fallback.strip()

        return self._clean_response(response)

    @staticmethod
    def _clean_response(response: str) -> str:
        text = (response or "").strip()
        leading_quotes = ('"', "'", "`", "â€œ", "â€", "â€˜", "â€™")
        while text and text[0] in leading_quotes:
            text = text[1:].lstrip()
        if text.startswith("?"):
            # Strip leading question mark artifacts such as ?" or ? 'Hello'
            trimmed = text[1:].lstrip()
            if trimmed and trimmed[0].isalpha():
                text = trimmed
        if text.startswith("\"") and len(text) > 1:
            text = text[1:].lstrip()
        return text


    def shutdown(self) -> None:
        self._closed = True
        self.executor.shutdown(wait=False)

