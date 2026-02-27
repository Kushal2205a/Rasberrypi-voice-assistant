from pathlib import Path
import os

PROJECT_DIR = Path.cwd()
RECORDED_WAV = PROJECT_DIR / "recorded.wav"

# Mic capture rate (your mic is 44.1kHz)
SAMPLE_RATE = 44100


INPUT_DEVICE = 1

MODEL_LITE = os.getenv("MODEL_LITE", "qwen:0.5b")
MODEL_PRO  = os.getenv("MODEL_PRO",  "tinyllama:latest")
MODEL = os.getenv("OLLAMA_MODEL", MODEL_LITE)

SWITCH_CODEWORD = os.getenv("SWITCH_CODEWORD", "hello").strip().lower()
SWITCH_REQUIRE_CODEWORD = os.getenv("SWITCH_REQUIRE_CODEWORD", "0").strip() == "1"

CHUNK_DURATION = 0.6

DEFAULT_SILENCE_TIMEOUT = 1.2      # seconds
DEFAULT_SILENCE_THRESHOLD = 600.0      # RMS amplitude threshold

WHISPER_EXE = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = Path.home() / "whisper.cpp" / "models" / "ggml-tiny.bin"  

VOSK_MODEL_PATH = Path.home() / "models" / "vosk-model-en-in-0.5"

PIPER_MODEL_PATH = Path.home() / "Rasberrypi-voice-assistant" / "voices" / "en_US-amy-medium.onnx"
WHISPER_SERVER_URL = "http://127.0.0.1:8080"
