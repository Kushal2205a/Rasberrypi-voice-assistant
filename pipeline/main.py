
from pathlib import Path
import os, argparse, RPi.GPIO as GPIO,time

from config import (
    CHUNK_DURATION,
    SAMPLE_RATE,
    WHISPER_EXE,
    WHISPER_MODEL,
    PIPER_MODEL_PATH,
    DEFAULT_SILENCE_THRESHOLD,
    DEFAULT_SILENCE_TIMEOUT,
    MODEL_LITE,
    MODEL_PRO,
    SWITCH_CODEWORD,
    SWITCH_REQUIRE_CODEWORD,
)

from orchestrator import ParallelVoiceAssistant


BUTTON_PIN = 17  
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming voice assistant pipeline")

    parser.add_argument("--duration", type=float, default=15.0, help="How long to run the streaming demo")
    parser.add_argument("--warmup", action="store_true", help="Run model warm-up steps before streaming")
    parser.add_argument("--piper-model", type=Path, default=PIPER_MODEL_PATH, help="Path to Piper .onnx model")
    parser.add_argument(
        "--whisper-cli",
        type=Path,
        default=WHISPER_EXE,
        help="Path to whisper.cpp CLI binary",
    )
    parser.add_argument(
        "--whisper-model",
        type=Path,
        default=WHISPER_MODEL,
        help="Path to whisper.cpp model",
    )
    parser.add_argument(
        "--whisper-threads",
        type=int,
        default=os.cpu_count() or 1,
        help="Threads to dedicate to whisper.cpp",
    )
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="Threads to pass to llama110")
    parser.add_argument("--n-predict", type=int, default=256, help="Tokens to generate with llama110")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for llama110")
    parser.add_argument("--llama-cli", type=Path, default=None, help="Optional override for llama-cli path")
    parser.add_argument("--llama-model", type=Path, default=None, help="Optional override for llama model path")

    parser.add_argument("--output-device", type=str, default=None, help="sounddevice output (index or name) for Piper playback")
    parser.add_argument("--playback-cmd", nargs="+", default=None, help="Fallback playback command for Piper audio")
    parser.add_argument(
        "--force-subprocess-playback",
        action="store_true",

        help="Always use the playback command for Piper audio (default)",
    )
    parser.add_argument(
        "--direct-playback",
        action="store_true",
        help="Play Piper audio through sounddevice instead of piping to the CLI player",
    )

    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=DEFAULT_SILENCE_TIMEOUT,
        help="Seconds of silence before automatically stopping the recorder",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=DEFAULT_SILENCE_THRESHOLD,
        help="RMS amplitude threshold (int16) to treat chunks as silence",
    )

    parser.add_argument(
        "--enable-stt-partials",
        action="store_true",
        help="Run whisper.cpp on each chunk for incremental transcripts",
        default = True 
    )
    
    parser.add_argument("--whisper-server", type=str, default=None, help="URL of whisper.cpp server, e.g. http://127.0.0.1:8080")

    parser.add_argument("--model-lite", type=str, default=MODEL_LITE, help="Ollama model tag for Lite mode")
    parser.add_argument("--model-pro", type=str, default=MODEL_PRO, help="Ollama model tag for Pro mode")
    parser.add_argument("--switch-codeword", type=str, default=SWITCH_CODEWORD, help="Codeword for switching (placeholder ok)")
    parser.add_argument(
        "--switch-require-codeword",
        action="store_true",
        default=SWITCH_REQUIRE_CODEWORD,
        help="Require codeword for switching (otherwise 'switch ... pro/lite' works too)",
    )
    parser.add_argument(
        "--switch-keep-history",
        action="store_true",
        help="Do NOT reset chat history when switching models (default resets)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    llama_kwargs = {
        "threads": args.threads,
        "n_predict": args.n_predict,
        "temperature": args.temperature,
    }
    if args.llama_cli:
        llama_kwargs["llama_cli_path"] = str(args.llama_cli)
    if args.llama_model:
        llama_kwargs["model_path"] = str(args.llama_model)


    if args.warmup:
        ModelPreloader.warmup_whisper(args.whisper_cli, args.whisper_model)
        ModelPreloader.warmup_llama(llama_kwargs)
        ModelPreloader.warmup_piper(args.piper_model)

    use_subprocess_playback = True
    if args.direct_playback:
        use_subprocess_playback = False
    elif args.force_subprocess_playback:
        use_subprocess_playback = True


    assistant = ParallelVoiceAssistant(
        chunk_duration=CHUNK_DURATION,
        sample_rate=SAMPLE_RATE,
        stt_workers=2,

        whisper_exe=args.whisper_cli,
        whisper_model=args.whisper_model,
        whisper_threads=args.whisper_threads,
        emit_stt_partials=args.enable_stt_partials,
        piper_model_path=args.piper_model,
        llama_kwargs=llama_kwargs,

        output_device=args.output_device,
        playback_cmd=args.playback_cmd,
        use_subprocess_playback=use_subprocess_playback,

        silence_timeout=args.silence_timeout,
        whisper_server=args.whisper_server,
        silence_threshold=args.silence_threshold,
        model_lite=args.model_lite,
        model_pro=args.model_pro,
        switch_codeword=args.switch_codeword,
        switch_require_codeword=args.switch_require_codeword,
        switch_reset_history=(not args.switch_keep_history),
    )
    max_duration = args.duration if args.duration and args.duration > 0 else None
    assistant.run(duration=max_duration)


try:
    print("Press the button to start the assistant.")
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            print("Button pressed — running session.")
            main()
            print("Session ended. Waiting for next press.")
            while GPIO.input(BUTTON_PIN) == GPIO.LOW:  # wait for release
                time.sleep(0.1)
        time.sleep(0.05)
except KeyboardInterrupt:
    GPIO.cleanup()
