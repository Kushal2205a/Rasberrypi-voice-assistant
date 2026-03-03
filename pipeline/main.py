from pathlib import Path
import os, argparse, time, threading, subprocess
import numpy as np

try:
    import RPi.GPIO as GPIO
    _HAVE_GPIO = True
except ImportError:
    _HAVE_GPIO = False

try:
    import sounddevice as sd
    _HAVE_SD = True
except ImportError:
    _HAVE_SD = False

from config import (
    CHUNK_DURATION,
    SAMPLE_RATE,
    INPUT_DEVICE,
    WHISPER_EXE,
    WHISPER_MODEL,
    PIPER_EXE,
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


def _bell_strike(freq: float = 1047.0, dur: float = 0.55,
                 sr: int = 22050, amp: float = 0.45) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    # NOTE: real bells have inharmonic partials, not integer multiples — these ratios produce a chime, not a buzz
    partials = [
        (1.00, 1.00),
        (2.76, 0.50),
        (5.40, 0.25),
        (8.93, 0.12),
    ]
    wave = np.zeros(len(t), dtype=np.float64)
    for ratio, gain in partials:
        decay = np.exp(-t * (5.0 + ratio * 1.5))  # higher partials decay faster — matches real bell physics
        wave += gain * decay * np.sin(2 * np.pi * freq * ratio * t)

    wave = wave / (np.max(np.abs(wave)) + 1e-9) * amp

    attack = max(1, int(sr * 0.004))  # 4 ms hard attack feels like a physical strike
    wave[:attack] *= np.linspace(0.0, 1.0, attack)

    return wave.astype(np.float32)


def _play_audio(wave: np.ndarray, sr: int = 22050, output_device=None) -> None:
    if not _HAVE_SD:
        return
    try:
        sd.play(wave, samplerate=sr, device=output_device, blocking=True)
    except Exception:
        pass


def _play_startup_chime(output_device=None) -> None:
    sr      = 22050
    strike1 = _bell_strike(freq=880.0, dur=0.55, sr=sr)
    gap     = np.zeros(int(sr * 0.08), dtype=np.float32)
    strike2 = _bell_strike(freq=880.0, dur=0.55, sr=sr)
    chime   = np.concatenate([strike1, gap, strike2])
    _play_audio(chime, sr=sr, output_device=output_device)


def _play_ready_chime(output_device=None) -> None:
    # NOTE: ascending minor third (C6 → Eb6) plays BEFORE mic switches to SESSION — cannot be mistaken for speech
    sr      = 22050
    strike1 = _bell_strike(freq=1047.0, dur=0.55, sr=sr)
    gap     = np.zeros(int(sr * 0.10), dtype=np.float32)
    strike2 = _bell_strike(freq=1245.0, dur=0.60, sr=sr)
    chime   = np.concatenate([strike1, gap, strike2])
    _play_audio(chime, sr=sr, output_device=output_device)


def _speak_via_piper(text: str, piper_model: Path, output_device=None) -> None:
    try:
        piper_cmd = [str(PIPER_EXE), "--model", str(piper_model), "--output_raw", "--quiet"]
        aplay_cmd = ["aplay", "-t", "raw", "-f", "S16_LE", "-r", "22050", "-c", "1"]
        if output_device:
            aplay_cmd += ["-D", str(output_device)]
        aplay_cmd.append("-")

        piper_proc = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        aplay_proc = subprocess.Popen(
            aplay_cmd,
            stdin=piper_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        piper_proc.stdin.write(text.encode("utf-8"))
        piper_proc.stdin.close()
        piper_proc.wait(timeout=12)
        aplay_proc.wait(timeout=12)
    except Exception:
        pass


def _prewarm_vosk(model_path=None) -> None:
    # NOTE: populates the _SHARED_MODEL singleton so the first session starts instantly
    try:
        from stt_vosk import _get_vosk_model
        from config import VOSK_MODEL_PATH
        path = model_path or VOSK_MODEL_PATH
        print(f"[WARMUP] Loading Vosk model from {path} ...")
        t0 = time.time()
        _get_vosk_model(path)
        print(f"[WARMUP] Vosk ready in {time.time() - t0:.1f}s")
    except Exception as exc:
        print(f"[WARMUP] Vosk pre-warm failed (non-fatal): {exc}")


def _run_startup(args) -> None:
    print("[STARTUP] Starting up Peppo...")

    piper_ok = Path(args.piper_model).exists()
    if piper_ok:
        ann_thread = threading.Thread(
            target=_speak_via_piper,
            args=("Hi, I'm Peppo. The models are warming up, please wait.", args.piper_model),
            kwargs={"output_device": args.output_device},
            daemon=True,
        )
        ann_thread.start()
    else:
        ann_thread = None
        print("[STARTUP] Piper model not found — skipping voice announcement.")

    _prewarm_vosk()

    if ann_thread is not None:
        ann_thread.join(timeout=14)

    # NOTE: ready chime is intentionally omitted here — it plays in session_runner
    # immediately before mic switches to SESSION mode, so it always means "I'm listening now".
    print("[STARTUP] Peppo is ready.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Peppo voice assistant")

    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--warmup", action="store_true")

    parser.add_argument("--wake-word", type=str, default=DEFAULT_KEYWORD,
                        help=f"Built-ins: {BUILTIN_KEYWORDS}")
    parser.add_argument("--wake-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--wake-cooldown", type=float, default=3.0)
    parser.add_argument("--no-wake-word", action="store_true",
                        help="Disable OWW; button-only mode")
    parser.add_argument("--list-devices", action="store_true")

    parser.add_argument("--mic-device", default=None)

    parser.add_argument("--piper-model", type=Path, default=PIPER_MODEL_PATH)
    parser.add_argument("--whisper-cli", type=Path, default=WHISPER_EXE)
    parser.add_argument("--whisper-model", type=Path, default=WHISPER_MODEL)
    parser.add_argument("--whisper-threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--whisper-server", type=str, default=None)
    parser.add_argument("--enable-stt-partials", action="store_true", default=True)

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

    parser.add_argument("--output-device", type=str, default=None)
    parser.add_argument("--playback-cmd", nargs="+", default=None)
    parser.add_argument("--force-subprocess-playback", action="store_true")
    parser.add_argument("--direct-playback", action="store_true")

    parser.add_argument("--silence-timeout", type=float, default=DEFAULT_SILENCE_TIMEOUT)
    parser.add_argument("--silence-threshold", type=float, default=DEFAULT_SILENCE_THRESHOLD)

    return parser.parse_args()


def _resolve_device(mic_device_arg):
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
        use_subprocess_playback=(not args.direct_playback),
        silence_timeout=args.silence_timeout,
        whisper_server=args.whisper_server,
        silence_threshold=args.silence_threshold,
        model_lite=args.model_lite,
        model_pro=args.model_pro,
        switch_codeword=args.switch_codeword,
        switch_require_codeword=args.switch_require_codeword,
        switch_reset_history=(not args.switch_keep_history),
        _recorder=recorder,
    )


def run_session(assistant, duration):
    assistant.run(duration=duration if duration and duration > 0 else None)


def main_wake_word(args: argparse.Namespace) -> None:
    device = _resolve_device(args.mic_device)

    mic = SharedMicStream(
        device=device,
        sample_rate=SAMPLE_RATE,
        chunk_duration=CHUNK_DURATION,
    )

    from recorder import StreamingRecorder
    recorder = StreamingRecorder(
        chunk_duration=CHUNK_DURATION,
        sample_rate=SAMPLE_RATE,
        input_device=device,
        push_mode=True,
    )
    recorder.start()

    _wake_event     = threading.Event()
    _session_active = threading.Event()

    def _trigger(source: str):
        if _session_active.is_set():
            print(f"[{source}] Session already active — ignoring.")
            return
        print(f"[{source}] Triggered — starting session.")
        _wake_event.set()

    detector = None
    if not args.no_wake_word:
        detector = WakeWordDetector(
            keyword=args.wake_word,
            threshold=args.wake_threshold,
            on_wake=lambda: _trigger("WakeWord"),
            cooldown=args.wake_cooldown,
        )
        mic.attach_detector(detector)

    mic.attach_recorder(recorder)

    def _button_loop():
        if not _HAVE_GPIO:
            return
        print(f"[Button] Monitoring GPIO pin {BUTTON_PIN}.")
        try:
            _was_pressed = False
            while True:
                if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    if not _was_pressed:
                        _was_pressed = True
                        _trigger("Button")
                else:
                    _was_pressed = False
                time.sleep(0.05)
        except Exception as exc:
            print(f"[Button] Thread error: {exc}")

    btn_thread = threading.Thread(target=_button_loop, name="ButtonMonitor", daemon=True)
    btn_thread.start()

    def session_runner():
        while True:
            _wake_event.wait()
            _wake_event.clear()
            _session_active.set()
            assistant = None
            try:
                assistant = _build_assistant(args, recorder)

                # NOTE: chime plays while mic is still in WAKE mode — cannot be captured as speech input
                _play_ready_chime(output_device=args.output_device)

                mic.set_mode(SharedMicStream.SESSION)
                recorder.clear_queue()

                run_session(assistant, args.duration)
                print("[Session] Ended — back to listening.")

            except Exception as exc:
                print(f"[Session] Error: {exc}")
            finally:
                if assistant is not None:
                    try:
                        assistant.llm.shutdown()
                    except Exception:
                        pass
                if detector is not None:
                    detector.reset_state()
                mic.set_mode(SharedMicStream.WAKE)
                _session_active.clear()

    sess_thread = threading.Thread(target=session_runner, name="SessionRunner", daemon=True)
    sess_thread.start()

    try:
        mic.start()
        if detector is not None:
            detector.start()
        mode_desc = ("wake word" if not args.no_wake_word else "")
        if _HAVE_GPIO:
            mode_desc = (mode_desc + " + button").lstrip(" + ")
        print(f"[MAIN] Peppo active — trigger: {mode_desc}.  Ctrl-C to quit.\n")
        _play_startup_chime(output_device=args.output_device)
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted.")
    finally:
        if detector is not None:
            detector.stop()
        mic.stop()
        if _HAVE_GPIO:
            GPIO.cleanup()


def main():
    args = _parse_args()

    if args.list_devices:
        print("Available audio input devices:")
        WakeWordDetector.list_devices()
        return

    _run_startup(args)
    main_wake_word(args)


if __name__ == "__main__":
    main()