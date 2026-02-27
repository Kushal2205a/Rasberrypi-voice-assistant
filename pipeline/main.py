from pathlib import Path
import os, argparse, time, threading

try:
    import RPi.GPIO as GPIO
    _HAVE_GPIO = True
except ImportError:
    _HAVE_GPIO = False

from config import (
    CHUNK_DURATION,
    SAMPLE_RATE,
    INPUT_DEVICE,
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
from wake_word import WakeWordDetector, DEFAULT_KEYWORD, DEFAULT_THRESHOLD, BUILTIN_KEYWORDS
from shared_mic import SharedMicStream

BUTTON_PIN = 17
if _HAVE_GPIO:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming voice assistant pipeline")

    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--warmup", action="store_true")

    # Wake word
    parser.add_argument("--wake-word", type=str, default=DEFAULT_KEYWORD,
                        help=f"Wake word model name or .onnx path. Built-ins: {BUILTIN_KEYWORDS}")
    parser.add_argument("--wake-threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="OWW confidence threshold 0.0-1.0 (default 0.5)")
    parser.add_argument("--wake-cooldown", type=float, default=3.0)
    parser.add_argument("--no-wake-word", action="store_true",
                        help="Disable wake word; use button fallback")
    parser.add_argument("--list-devices", action="store_true")

    # Mic device (shared between wake word and recorder)
    parser.add_argument("--mic-device", default=None,
                        help="sounddevice device index or name for the mic "
                             "(default: uses INPUT_DEVICE from config.py)")

    # STT
    parser.add_argument("--piper-model", type=Path, default=PIPER_MODEL_PATH)
    parser.add_argument("--whisper-cli", type=Path, default=WHISPER_EXE)
    parser.add_argument("--whisper-model", type=Path, default=WHISPER_MODEL)
    parser.add_argument("--whisper-threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--whisper-server", type=str, default=None)
    parser.add_argument("--enable-stt-partials", action="store_true", default=True)

    # LLM
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--llama-cli", type=Path, default=None)
    parser.add_argument("--llama-model", type=Path, default=None)
    parser.add_argument("--model-lite", type=str, default=MODEL_LITE)
    parser.add_argument("--model-pro", type=str, default=MODEL_PRO)
    parser.add_argument("--switch-codeword", type=str, default=SWITCH_CODEWORD)
    parser.add_argument("--switch-require-codeword", action="store_true",
                        default=SWITCH_REQUIRE_CODEWORD)
    parser.add_argument("--switch-keep-history", action="store_true")

    # TTS
    parser.add_argument("--output-device", type=str, default=None)
    parser.add_argument("--playback-cmd", nargs="+", default=None)
    parser.add_argument("--force-subprocess-playback", action="store_true")
    parser.add_argument("--direct-playback", action="store_true")

    # Silence
    parser.add_argument("--silence-timeout", type=float, default=DEFAULT_SILENCE_TIMEOUT)
    parser.add_argument("--silence-threshold", type=float, default=DEFAULT_SILENCE_THRESHOLD)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(mic_device_arg):
    """Return the device to use: CLI arg > config INPUT_DEVICE > None."""
    if mic_device_arg is not None:
        try:
            return int(mic_device_arg)
        except (ValueError, TypeError):
            return mic_device_arg
    if INPUT_DEVICE is not None:
        return INPUT_DEVICE
    return None


def _build_assistant(args, recorder) -> ParallelVoiceAssistant:
    llama_kwargs: dict = {
        "threads": args.threads,
        "n_predict": args.n_predict,
        "temperature": args.temperature,
    }
    if args.llama_cli:
        llama_kwargs["llama_cli_path"] = str(args.llama_cli)
    if args.llama_model:
        llama_kwargs["model_path"] = str(args.llama_model)

    use_subprocess = True
    if args.direct_playback:
        use_subprocess = False

    return ParallelVoiceAssistant(
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
        use_subprocess_playback=use_subprocess,
        silence_timeout=args.silence_timeout,
        whisper_server=args.whisper_server,
        silence_threshold=args.silence_threshold,
        model_lite=args.model_lite,
        model_pro=args.model_pro,
        switch_codeword=args.switch_codeword,
        switch_require_codeword=args.switch_require_codeword,
        switch_reset_history=(not args.switch_keep_history),
        _recorder=recorder,   # inject pre-built recorder
    )


def run_session(assistant, duration):
    assistant.run(duration=duration if duration and duration > 0 else None)


# ---------------------------------------------------------------------------
# Wake-word mode  (SharedMicStream — single ALSA open)
# ---------------------------------------------------------------------------

def main_wake_word(args: argparse.Namespace) -> None:
    """
    One sounddevice stream stays open the whole time.
    In WAKE mode  → audio goes to OWW detector.
    In SESSION mode → audio goes to the recorder's queue.
    No open/close/reopen — no ALSA conflict.
    """
    device = _resolve_device(args.mic_device)

    # Single shared mic stream
    mic = SharedMicStream(
        device=device,
        sample_rate=SAMPLE_RATE,
        chunk_duration=CHUNK_DURATION,
    )

    # Recorder in push mode (no own InputStream) — reused across sessions
    from recorder import StreamingRecorder
    recorder = StreamingRecorder(
        chunk_duration=CHUNK_DURATION,
        sample_rate=SAMPLE_RATE,
        input_device=device,
        push_mode=True,
    )
    # Start recorder once so recording=True persists across sessions
    recorder.start()

    _wake_event     = threading.Event()
    _session_active = threading.Event()

    def on_wake():
        if _session_active.is_set():
            print("[WakeWord] Session already active — ignoring.")
            return
        print("[WakeWord] Wake word detected — starting session.")
        _wake_event.set()

    detector = WakeWordDetector(
        keyword=args.wake_word,
        threshold=args.wake_threshold,
        on_wake=on_wake,
        cooldown=args.wake_cooldown,
    )

    mic.attach_detector(detector)
    mic.attach_recorder(recorder)

    def session_runner():
        while True:
            _wake_event.wait()
            _wake_event.clear()
            _session_active.set()
            assistant = None
            try:
                # Switch mic routing to recorder — instant, no stream restart
                mic.set_mode(SharedMicStream.SESSION)
                recorder.clear_queue()

                # Build a fresh assistant each session (STT/LLM executors can't be reused after shutdown)
                assistant = _build_assistant(args, recorder)

                # Audio cue
                try:
                    assistant.tts.start_playback()
                    fut = assistant.tts.generate_and_queue("I'm listening.", 0)
                    if fut:
                        fut.result(timeout=5)
                except Exception:
                    pass

                run_session(assistant, args.duration)
                print("[WakeWord] Session ended — back to listening.")
            except Exception as exc:
                print(f"[Session] Error: {exc}")
            finally:
                if assistant is not None:
                    try:
                        assistant.llm.shutdown()
                    except Exception:
                        pass
                # Reset OWW state BEFORE switching mode so no stale audio triggers
                detector.reset_state()
                # Switch back to wake mode — instant
                mic.set_mode(SharedMicStream.WAKE)
                _session_active.clear()

    sess_thread = threading.Thread(target=session_runner, name="SessionRunner", daemon=True)
    sess_thread.start()

    try:
        mic.start()
        detector.start()
        print("Press Ctrl-C to quit.\n")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")
    finally:
        detector.stop()
        mic.stop()


# ---------------------------------------------------------------------------
# Button fallback mode
# ---------------------------------------------------------------------------

def main_button(args: argparse.Namespace) -> None:
    if not _HAVE_GPIO:
        print("[MAIN] No GPIO — running a single session.")
        from recorder import StreamingRecorder
        recorder = StreamingRecorder(chunk_duration=CHUNK_DURATION, sample_rate=SAMPLE_RATE)
        assistant = _build_assistant(args, recorder)
        run_session(assistant, args.duration)
        return

    from recorder import StreamingRecorder
    recorder  = StreamingRecorder(chunk_duration=CHUNK_DURATION, sample_rate=SAMPLE_RATE)
    assistant = _build_assistant(args, recorder)
    try:
        print("Press the button to start the assistant.")
        while True:
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                print("Button pressed — running session.")
                run_session(assistant, args.duration)
                print("Session ended. Waiting for next press.")
                while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    time.sleep(0.1)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    if args.list_devices:
        from wake_word import WakeWordDetector
        print("Available audio input devices:")
        WakeWordDetector.list_devices()
        return

    if args.no_wake_word:
        main_button(args)
    else:
        main_wake_word(args)


if __name__ == "__main__":
    main()