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
# Audio cue helpers
# ---------------------------------------------------------------------------

def _bell_strike(freq: float = 1047.0, dur: float = 0.55,
                 sr: int = 22050, amp: float = 0.45) -> np.ndarray:
    """
    Synthesise a single bell strike.

    A real bell doesn't ring at integer multiples of its fundamental —
    it has *inharmonic* partials.  Using those ratios (2.76×, 5.40×, 8.93×)
    plus per-partial exponential decay is what makes this sound like a
    chime rather than a buzzer.
    """
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    # (frequency_ratio, relative_gain)
    partials = [
        (1.00, 1.00),   # fundamental
        (2.76, 0.50),   # inharmonic 2nd partial
        (5.40, 0.25),   # 3rd partial
        (8.93, 0.12),   # 4th partial — adds that glassy shimmer
    ]
    wave = np.zeros(len(t), dtype=np.float64)
    for ratio, gain in partials:
        # Higher partials decay faster — matches real bell physics
        decay = np.exp(-t * (5.0 + ratio * 1.5))
        wave += gain * decay * np.sin(2 * np.pi * freq * ratio * t)

    # Normalise then scale to desired amplitude
    wave = wave / (np.max(np.abs(wave)) + 1e-9) * amp

    # 4 ms hard attack (feels like a physical strike, not a fade-in)
    attack = max(1, int(sr * 0.004))
    wave[:attack] *= np.linspace(0.0, 1.0, attack)

    return wave.astype(np.float32)


def _play_audio(wave: np.ndarray, sr: int = 22050, output_device=None) -> None:
    """Blocking playback. Fails silently — never crash over a missing beep."""
    if not _HAVE_SD:
        return
    try:
        sd.play(wave, samplerate=sr, device=output_device, blocking=True)
    except Exception:
        pass


def _play_startup_chime(output_device=None) -> None:
    """
    Boot chime: two soft bell strikes played close together.
    Tells the user 'the Pi is alive, wait a moment'.
    Runs before any model loads so it's always instant.
    """
    sr = 22050
    strike1 = _bell_strike(freq=880.0,  dur=0.55, sr=sr)   # A5
    gap     = np.zeros(int(sr * 0.08),  dtype=np.float32)   # 80 ms gap
    strike2 = _bell_strike(freq=880.0,  dur=0.55, sr=sr)   # A5 again (soft double-knock)
    chime   = np.concatenate([strike1, gap, strike2])
    _play_audio(chime, sr=sr, output_device=output_device)


def _play_ready_chime(output_device=None) -> None:
    """
    Ready chime: two bell strikes rising by a minor third (C6 → Eb6).
    Plays *before* the mic switches to SESSION mode so it cannot be
    mistaken for speech.  Ascending interval = 'I'm ready for you'.
    """
    sr = 22050
    strike1 = _bell_strike(freq=1047.0, dur=0.55, sr=sr)   # C6
    gap     = np.zeros(int(sr * 0.10),  dtype=np.float32)   # 100 ms gap
    strike2 = _bell_strike(freq=1245.0, dur=0.60, sr=sr)   # Eb6  (minor third up)
    chime   = np.concatenate([strike1, gap, strike2])
    _play_audio(chime, sr=sr, output_device=output_device)


def _speak_via_piper(text: str, piper_model: Path, output_device=None) -> None:
    """
    Speak a phrase using the Piper TTS binary.
    Pipes raw PCM straight to aplay — no temp file.
    Fails silently if Piper or aplay is unavailable.
    """
    try:
        piper_cmd = ["piper", "--model", str(piper_model),
                     "--output_raw", "--quiet"]
        aplay_cmd = ["aplay", "-t", "raw", "-f", "S16_LE",
                     "-r", "22050", "-c", "1"]
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
        pass   # never block startup over a failed announcement


# ---------------------------------------------------------------------------
# Vosk pre-warm
# ---------------------------------------------------------------------------

def _prewarm_vosk(model_path=None) -> None:
    """
    Load the shared Vosk model singleton before any session starts.
    After this call, PersistentVoskSTT.__init__ finds _SHARED_MODEL already
    set and returns instantly — eliminating the first-press delay.
    """
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


# ---------------------------------------------------------------------------
# Startup sequence
# ---------------------------------------------------------------------------

def _run_startup(args) -> None:
    """
    Runs once at process start.  Timeline:

        t=0       Boot chime (two soft bell strikes — instant)
        t=0       Piper says "Hello I'm Peppo, warming up the models"  ← background thread
        t=0       Vosk model loads                                      ← foreground (parallel)
        t=warm    Wait for Piper announcement to finish
        t=warm    Ready chime (two rising bell strikes)
        → user may now trigger a session
    """
    print("[STARTUP] Starting up Peppo...")

    # 1. Instant boot chime — tells user the Pi is alive before anything loads
    _play_startup_chime(output_device=args.output_device)

    # 2. Voice announcement — start in background so Vosk loads in parallel
    piper_ok = Path(args.piper_model).exists()
    if piper_ok:
        ann_thread = threading.Thread(
            target=_speak_via_piper,
            args=("Hello, I'm Peppo. Warming up the models.",
                  args.piper_model),
            kwargs={"output_device": args.output_device},
            daemon=True,
        )
        ann_thread.start()
    else:
        ann_thread = None
        print("[STARTUP] Piper model not found — skipping voice announcement.")

    # 3. Vosk pre-warm (blocks here — runs while Piper speaks)
    _prewarm_vosk()

    # 4. Wait for Piper to finish before the ready chime
    if ann_thread is not None:
        ann_thread.join(timeout=14)

    # 5. Ready chime — safe to trigger from this point
    _play_ready_chime(output_device=args.output_device)
    print("[STARTUP] Peppo is ready.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Unified wake-word + button mode
# ---------------------------------------------------------------------------

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

    # --- Wake word detector ---
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

    # --- GPIO button thread — always active alongside wake word ---
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

    # --- Session runner ---
    def session_runner():
        while True:
            _wake_event.wait()
            _wake_event.clear()
            _session_active.set()
            assistant = None
            try:
                # 1. Build assistant while mic is still in WAKE mode
                #    (Vosk is already warm — this is fast)
                assistant = _build_assistant(args, recorder)

                # 2. Play ready chime BEFORE switching to SESSION mode
                #    Mic still routes to OWW here — chime cannot be heard as speech
                _play_ready_chime(output_device=args.output_device)

                # 3. Switch mic to SESSION and discard anything queued before now
                mic.set_mode(SharedMicStream.SESSION)
                recorder.clear_queue()

                # 4. Begin listening
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    if args.list_devices:
        print("Available audio input devices:")
        WakeWordDetector.list_devices()
        return

    _run_startup(args)      # chime → "Hello I'm Peppo" → Vosk warm → ready chime
    main_wake_word(args)    # wake word + button coexist forever


if __name__ == "__main__":
    main()