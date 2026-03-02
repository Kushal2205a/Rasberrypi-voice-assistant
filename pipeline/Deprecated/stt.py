from typing import Optional, Any,  Dict, List, Optional
import shutil, threading, numpy as np
import tempfile,subprocess,wave, contextlib
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
import io, json, requests,time,os


from config import SAMPLE_RATE, WHISPER_EXE, WHISPER_MODEL





# Parallel Speech-to-Text (whisper.cpp)

class ParallelSTT:

    """Process audio chunks asynchronously using the whisper.cpp CLI."""

    def __init__(
        self,
        num_workers: int = 2,

        sample_rate: int = SAMPLE_RATE,
        whisper_exe: Path = WHISPER_EXE,
        whisper_model: Path = WHISPER_MODEL,
        whisper_threads: Optional[int] = None,
        emit_partials: bool = False,
    ) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max(1, num_workers))
        self.sample_rate = int(sample_rate)
        self.whisper_exe = Path(whisper_exe)
        self.whisper_model = Path(whisper_model)
        self.whisper_threads = int(whisper_threads) if whisper_threads else None
        self.emit_partials = bool(emit_partials)

        self._chunks: List[bytes] = []
        self._transcript_lock = threading.Lock()
        self._emitted_transcript = ""
        self._last_partial = ""
        
        ramdir = "/dev/shm"
        mkdtemp_kwargs = {"prefix":"pipeline_stt_"}
        if os.path.isdir(ramdir):
            mkdtemp_kwargs["dir"] = ramdir
        self._temp_dir = Path(tempfile.mkdtemp(**mkdtemp_kwargs))

        if not self.whisper_exe.exists():
            raise FileNotFoundError(f"Whisper binary not found: {self.whisper_exe}")
        if not self.whisper_model.exists():
            raise FileNotFoundError(f"Whisper model not found: {self.whisper_model}")

    # --------------------- Whisper helpers -----------------------
    
    def empty_future(self, chunk_id: int) -> Future:
        return self.executor.submit(lambda cid=chunk_id: {"chunk_id": cid, "text": "", "is_final": False})

    def _write_wav(self, audio_bytes: bytes, prefix: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix=f"{prefix}_", dir=self._temp_dir, delete=False
        )
        tmp_path = Path(tmp.name)
        tmp.close()
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_bytes)
        return tmp_path

    def _run_whisper(self, wav_path: Path, timeout: int = 60) -> str:
        cmd = [
            str(self.whisper_exe),
            "-m",
            str(self.whisper_model),
            "-f",
            str(wav_path),
            "--no-prints",
            "--output-txt",
        ]
        if self.whisper_threads:
            cmd += ["-t", str(self.whisper_threads)]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print("[STT][Whisper] Transcription timed out")
            return ""
        except Exception as exc:
            print(f"[STT][Whisper] Error running whisper.cpp: {exc}")
            return ""

        text = ""
        candidates = [wav_path.with_suffix(".txt"), Path(str(wav_path) + ".txt")]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as exc:
                print(f"[STT][Whisper] Failed reading {candidate.name}: {exc}")
                text = ""
            finally:
                candidate.unlink(missing_ok=True)
            if text:
                break

        return text

    def _process_chunk_whisper(self, audio_bytes: bytes, chunk_id: int) -> Dict[str, Any]:
        wav_path = self._write_wav(audio_bytes, f"chunk{chunk_id}")
        try:
            text = self._run_whisper(wav_path, timeout=45)
        finally:
            wav_path.unlink(missing_ok=True)

        text = (text or "").strip()
        if not text:
            return {"chunk_id": chunk_id, "text": "", "is_final": False}

        with self._transcript_lock:
            if text.lower() == self._last_partial.lower():
                return {"chunk_id": chunk_id, "text": "", "is_final": False}
            self._last_partial = text
            self._emitted_transcript = (f"{self._emitted_transcript} {text}").strip()

        return {"chunk_id": chunk_id, "text": text, "is_final": False}

    def _finalize_whisper(self, chunk_id: int, mark_final: bool) -> Dict[str, Any]:
        with self._transcript_lock:
            chunks = list(self._chunks)
            emitted = self._emitted_transcript.strip()
            self._last_partial = ""
            self._chunks.clear()

        if not chunks:
            return {"chunk_id": chunk_id, "text": "", "is_final": True}

        wav_path = self._write_wav(b"".join(chunks), f"session{chunk_id}")
        try:
            full_text = (self._run_whisper(wav_path, timeout=120) or "").strip()
        finally:
            wav_path.unlink(missing_ok=True)

        new_text = full_text
        if emitted and full_text.lower().startswith(emitted.lower()):
            new_text = full_text[len(emitted) :].strip()

        with self._transcript_lock:
            self._emitted_transcript = full_text

        return {"chunk_id": chunk_id, "text": new_text, "is_final": mark_final}

    def _empty_chunk(self, chunk_id: int) -> Dict[str, Any]:
        return {"chunk_id": chunk_id, "text": "", "is_final": False}


    # -------------------------- Public API -----------------------

    def submit_chunk(self, audio_chunk: np.ndarray, chunk_id: int) -> Future:

        audio_bytes = np.ascontiguousarray(audio_chunk, dtype=np.int16).tobytes()
        with self._transcript_lock:
            self._chunks.append(audio_bytes)

        if not self.emit_partials:
            return self.executor.submit(self._empty_chunk, chunk_id)
        return self.executor.submit(self._process_chunk_whisper, audio_bytes, chunk_id)

    def finalize(self, chunk_id: int, mark_final: bool = True) -> Optional[Future]:
        return self.executor.submit(self._finalize_whisper, chunk_id, mark_final)

    def reset(self) -> None:
        with self._transcript_lock:
            self._chunks.clear()
            self._emitted_transcript = ""
            self._last_partial = ""

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
        with contextlib.suppress(Exception):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

# --- HTTP (whisper.cpp --server) client ---
import io, json, requests

class ParallelSTTHTTP:
    """
    Talk to a persistent whisper.cpp server over HTTP.
    Each submit_chunk sends ~0.5s WAV; server returns JSON {text: "..."} quickly.
    finalize() sends the session buffer for a full decode to get any tail text.
    """
    def __init__(self, num_workers: int = 2, sample_rate: int = SAMPLE_RATE,
                 server_url: str = "http://127.0.0.1:8080", emit_partials: bool = True):
        self.executor = ThreadPoolExecutor(max_workers=max(1, num_workers))
        self.sample_rate = int(sample_rate)
        self.server_url = server_url.rstrip("/")
        self.emit_partials = bool(emit_partials)

        self._chunks: List[bytes] = []
        self._transcript_lock = threading.Lock()
        self._emitted_transcript = ""
        self._last_partial = ""
        self._rolling: bytearray = bytearray()
        self._rolling_ms = 600 
        self._last_http_fail = 0.0


        # quick health check (non-fatal)
        try:
            requests.get(self.server_url, timeout=0.5)
        except Exception:
            pass
        
    def empty_future(self, chunk_id: int) -> Future:
        return self.executor.submit(lambda cid=chunk_id: {"chunk_id": cid, "text": "", "is_final": False})

    def _wav_bytes(self, audio_bytes: bytes) -> bytes:
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_bytes)
        return bio.getvalue()

    def _post_audio(self, wav_bytes: bytes, timeout: float = 8.0) -> str:
        # Typical server accepts multipart with field name 'audio'
        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}   # <— changed key to "file"
        data = {"response_format": "json",
                "temparature":"0.2",
                "audio_format":"wav"}  
        try:
            r = requests.post(f"{self.server_url}/inference",
                  files=files,
                  data=data,   # <-- add this
                  timeout=timeout)

            r.raise_for_status()
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            # try common keys
            text = (data.get("text") or data.get("transcription") or "").strip()
            if not text and isinstance(data.get("segments"), list):
                text = " ".join(seg.get("text","") for seg in data["segments"]).strip()
            return text
        except requests.Timeout:
            print("Timeout HTTP")
            self._last_http_fail = time.time()
            return ""
        except Exception as e:
            print(f"HTTP Error {e}")
            self._last_http_fail = time.time()
            return ""

    def _process_chunk_http(self, audio_bytes: bytes, chunk_id: int) -> Dict[str, Any]:
        # --- build a WAV from the rolling tail (~1.2 s), not just this chunk
        with self._transcript_lock:
            rolling = bytes(self._rolling)

        # safety: if rolling is very short/silent, pad to ~200 ms
        min_ms = 200.0
        min_bytes = int(self.sample_rate * 2 * (min_ms / 1000.0))
        if len(rolling) < min_bytes:
            rolling = rolling + b"\x00" * (min_bytes - len(rolling))

        wav = self._wav_bytes(rolling)
        text = (self._post_audio(wav, timeout=20.0) or "").strip()  # keep tight timeout

        if not text:
            # optional fallback: try just the chunk once (rarely needed)
            wav_single = self._wav_bytes(audio_bytes)
            text = (self._post_audio(wav_single, timeout=30.0) or "").strip()

        return {"chunk_id": chunk_id, "text": text, "is_final": False}


    def _finalize_http(self, chunk_id: int, mark_final: bool) -> Dict[str, Any]:
        # Join buffered PCM and send once more for a full pass (to catch trailing words)
        with self._transcript_lock:
            chunks = list(self._chunks)
            emitted = self._emitted_transcript.strip()
            self._last_partial = ""
            self._chunks.clear()

        if not chunks:
            return {"chunk_id": chunk_id, "text": "", "is_final": True}

        wav = self._wav_bytes(b"".join(chunks))
        full_text = (self._post_audio(wav, timeout=30.0) or "").strip()

        new_text = full_text
        if emitted and full_text.lower().startswith(emitted.lower()):
            new_text = full_text[len(emitted):].strip()

        with self._transcript_lock:
            self._emitted_transcript = full_text

        return {"chunk_id": chunk_id, "text": new_text, "is_final": mark_final}

    # Public API compatible with your orchestrator
    def submit_chunk(self, audio_chunk: np.ndarray, chunk_id: int) -> Future:
        audio_bytes = np.ascontiguousarray(audio_chunk, dtype=np.int16).tobytes()
        with self._transcript_lock:
            self._chunks.append(audio_bytes)
            # update rolling buffer (PCM int16 mono)
            max_bytes = int(self.sample_rate * 2 * (self._rolling_ms / 1000.0))
            self._rolling.extend(audio_bytes)
            if len(self._rolling) > max_bytes:
                # keep tail only
                self._rolling = self._rolling[-max_bytes:]
              

            if not self.emit_partials:
                return self.executor.submit(lambda cid=chunk_id: {"chunk_id": cid, "text": "", "is_final": False})
            
            # back off for 2s if server is timing out/absent
            if (time.time() - getattr(self, "_last_http_fail", 0.0)) < 2.0:
                return self.executor.submit(lambda cid=chunk_id: {"chunk_id": cid, "text": "", "is_final": False})


            return self.executor.submit(self._process_chunk_http, audio_bytes, chunk_id)

    def finalize(self, chunk_id: int, mark_final: bool = True) -> Optional[Future]:
        return self.executor.submit(self._finalize_http, chunk_id, mark_final)

    def reset(self) -> None:
        with self._transcript_lock:
            self._chunks.clear()
            self._emitted_transcript = ""
            self._last_partial = ""

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
