
from pathlib import Path
from typing import Optional, Any, Dict
import os, tempfile, wave, subprocess, shutil

from config import SAMPLE_RATE, WHISPER_EXE, WHISPER_MODEL, PIPER_MODEL_PATH, DEFAULT_SILENCE_THRESHOLD, DEFAULT_SILENCE_TIMEOUT




class ModelPreloader:
    """Utility helpers to warm up the local models so first inference is faster."""

    @staticmethod

    def warmup_whisper(
        
        whisper_exe: Path = WHISPER_EXE,
        whisper_model: Path = WHISPER_MODEL,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        print("[WARMUP] Priming whisper.cpp...")
        exe = Path(whisper_exe)
        model = Path(whisper_model)
        if not exe.exists():
            print(f"[WARMUP] Whisper binary missing at {exe}")
            return
        if not model.exists():
            print(f"[WARMUP] Whisper model missing at {model}")
            return

        tmp_dir = Path(tempfile.mkdtemp(prefix="whisper_warmup_"))
        wav_path = tmp_dir / "warmup.wav"
        success = True
        try:
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(b"\x00" * sample_rate)
            cmd = [
                str(exe),
                "-m",
                str(model),
                "-f",
                str(wav_path),
                "--no-prints",
                "--output-txt",
                "-t",
                "1",
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
        except Exception as exc:
            print(f"[WARMUP] Whisper warm-up failed: {exc}")
            success = False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if success:
            print("[WARMUP] whisper.cpp ready")


    @staticmethod
    def warmup_llama(llama_kwargs: Optional[Dict[str, Any]] = None) -> None:
        print("[WARMUP] Warming up llama110...")
        kwargs = llama_kwargs or {}
        try:
            from llm_model import llama110
            llama110(
                prompt_text="Hello",
                llama_cli_path=kwargs.get("llama_cli_path"),
                model_path=kwargs.get("model_path"),
                n_predict=96,
                threads=kwargs.get("threads", os.cpu_count() or 4),
                temperature=kwargs.get("temperature", 0.5),
                sampler=kwargs.get("sampler"),
                timeout_seconds=kwargs.get("timeout_seconds", 120),
            )
        except Exception as exc:
            print(f"[WARMUP] llama110 warm-up failed: {exc}")
        else:
            print("[WARMUP] llama110 ready")

    @staticmethod
    def warmup_piper(model_path: Path = PIPER_MODEL_PATH) -> None:
        print("[WARMUP] Warming up Piper...")
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"[WARMUP] Piper model missing at {model_path}")
            return
        cmd = ["piper", "-m", str(model_path), "--output_file", "/tmp/piper_warmup.wav"]
        try:
            subprocess.run(cmd, input="Warm up".encode("utf-8"), check=True, timeout=10)
        except Exception as exc:
            print(f"[WARMUP] Piper warm-up failed: {exc}")
        else:
            Path("/tmp/piper_warmup.wav").unlink(missing_ok=True)
            print("[WARMUP] Piper ready")
