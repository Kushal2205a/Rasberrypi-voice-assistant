from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional,Any,Iterable,Callable, Dict, List
from concurrent.futures import Future, ThreadPoolExecutor

import os, subprocess, threading, queue, json, selectors,wave,tempfile,shutil
import numpy as np, sounddevice as sd 
import time

from config import PIPER_MODEL_PATH


@dataclass
class SpeechSegment:
    path: str
    raw: bytes
    sample_rate: int
    channels: int = 1
    sampwidth: int = 2
    text: str = ""


@dataclass
class PiperVoiceInfo:
    sample_rate: int = 22050
    speaker_id: Optional[int] = None
    channels: int = 1
    metadata_path: Optional[Path] = None


class BufferedTTS:
    """Generate speech with Piper asynchronously and stream playback via a CLI player."""


    def __init__(
        self,
        model_path: Path = PIPER_MODEL_PATH,
        playback_cmd: Optional[Iterable[str]] = None,

        output_device: Optional[Any] = None,
        use_subprocess: bool = False,
        on_playback_start: Optional[Callable[[str, float], None]] = None,
        on_playback_error: Optional[Callable[[], None]] = None,
        timeout :int = 30,
        blocking_playback :bool = True

    ) -> None:
        self.model_path = Path(model_path)
        self.timeout = int(timeout)
        self._voice_info = self._load_voice_info()
        self.speech_queue: "queue.Queue[SpeechSegment]" = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.playing = False
        self._playing_audio = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None
        self.use_subprocess = bool(use_subprocess)

        info = self._voice_info
        if playback_cmd:
            self.playback_cmd = list(playback_cmd)
        else:
            paplay = shutil.which("paplay")
            aplay = shutil.which("aplay")
            if paplay:
                # paplay works well with PulseAudio/Bluetooth
                self.playback_cmd = ["paplay", "--raw",
                                    "--rate", str(info.sample_rate),
                                    "--channels", str(info.channels),
                                    "--format", "s16le", "-"]
            elif aplay:
                self.playback_cmd = ["aplay", "-t", "raw",
                                    "-f", "S16_LE",
                                    "-r", str(info.sample_rate),
                                    "-c", str(info.channels), "-"]
            else:
                # fallback to direct PortAudio playback if no cli players present
                self.playback_cmd = []
                self.use_subprocess = False
                
        self.blocking_playback = bool (blocking_playback)
        
        
        if output_device is None:
            self.output_device = None
        elif isinstance(output_device, int):
            self.output_device = output_device
        elif isinstance(output_device, str):
            try:
                self.output_device = int(output_device)
            except ValueError:
                self.output_device = output_device
        else:
            self.output_device = output_device
        self.on_playback_start = on_playback_start
        self.on_playback_error = on_playback_error
        
        self._piper_proc = None
        self._piper_lock = threading.Lock()
        
        self.out_dir = None 

        self._playback_env = os.environ.copy()
        if isinstance(self.output_device, str):
            # Hint to PulseAudio-based players which sink to target.
            self._playback_env.setdefault("PULSE_SINK", self.output_device)


    def _load_voice_info(self) -> PiperVoiceInfo:
        candidates = [
            self.model_path.with_suffix(self.model_path.suffix + ".json"),
            self.model_path.with_suffix(".json"),
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue

            audio = metadata.get("audio", {}) if isinstance(metadata, dict) else {}
            sample_rate = int(audio.get("sample_rate") or metadata.get("sample_rate", 22050))
            channels = int(audio.get("channels", 1) or 1)

            speaker_id: Optional[int] = None
            if "speaker_id" in metadata:
                try:
                    speaker_id = int(metadata.get("speaker_id"))
                except Exception:
                    speaker_id = None
            elif isinstance(metadata.get("speakers"), dict):
                speakers_dict: Dict[str, Any] = metadata.get("speakers", {})
                if speakers_dict:
                    first_key = next(iter(speakers_dict))
                    first_val = speakers_dict[first_key]
                    if isinstance(first_val, dict) and "id" in first_val:
                        try:
                            speaker_id = int(first_val["id"])
                        except Exception:
                            speaker_id = None
                    else:
                        try:
                            speaker_id = int(first_key)
                        except Exception:
                            speaker_id = None

            return PiperVoiceInfo(
                sample_rate=sample_rate or 22050,
                speaker_id=speaker_id,
                channels=channels or 1,
                metadata_path=candidate,
            )

        return PiperVoiceInfo()
    
    def _ensure_piper(self):
        if self._piper_proc and self._piper_proc.poll() is None:
            return

        if self.out_dir is None:
            self.out_dir = Path(tempfile.mkdtemp(prefix="piper_out_"))

        # Resolve piper exe: 1) env PIPER_EXE  2) PATH (which)  3) common fallback
        import shutil
        piper_exe = (os.environ.get("PIPER_EXE")
                or shutil.which("piper")
                or os.path.expanduser("/usr/local/bin/piper/piper"))

        if not Path(piper_exe).exists() and shutil.which(piper_exe) is None:
            print(f"[TTS] Piper not found at '{piper_exe}'. Set PIPER_EXE or install piper.")
            self._piper_proc = None
            return

        info = self._voice_info
        cmd = [piper_exe, "-m", str(self.model_path), "--output_raw", "-"]
        if info.speaker_id is not None:
            cmd += ["--speaker", str(info.speaker_id)]

        try:
            self._piper_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # change to subprocess.STDOUT to debug
                text=False,
                bufsize=0,
            )
            print(f"[TTS] Started Piper: {' '.join(cmd)}")
        except Exception as e:
            print(f"[TTS] Failed to start Piper: {e}  (cmd: {' '.join(cmd)})")
            self._piper_proc = None


    def start_playback(self) -> None:
        if self.playing:
            return
        self.playing = True
        self._playback_thread = threading.Thread(target=self._playback_loop, name="BufferedTTS", daemon=True)
        self._playback_thread.start()

    def _playback_loop(self) -> None:
        while self.playing:
            try:

                segment = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not segment:
                continue

            self._playing_audio.set()
            try:
                played = False
                if not self.use_subprocess:
                    played = self._play_via_sounddevice(segment)

                if not played and self.playback_cmd:
                    played = self._play_via_subprocess(segment)

                if not played:
                    print(f"[TTS] Playback failed for {segment.path}")
                    self._notify_playback_error()
            finally:
                self._playing_audio.clear()

            try:
                if segment.path:
                    Path(segment.path).unlink(missing_ok=True)

            except OSError:
                pass
    def is_playing_audio(self) -> bool:
        return self._playing_audio.is_set()
    def _notify_playback_error(self) -> None:
        if self.on_playback_error is None:
            return
        try:
            self.on_playback_error()
        except Exception as exc:
            print(f"[TTS] Playback error callback failed: {exc}")


    def _play_via_sounddevice(self, segment: SpeechSegment) -> bool:
        try:
            if segment.raw:
                audio = np.frombuffer(segment.raw, dtype=np.int16)
                max_val = float(np.iinfo(np.int16).max)
                audio = audio.astype(np.float32) / max_val
                if segment.channels > 1:
                    audio = audio.reshape(-1, segment.channels)
                sample_rate = segment.sample_rate
            else:
                with wave.open(segment.path, "rb") as wf:
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    frames = wf.getnframes()
                    audio_bytes = wf.readframes(frames)

                dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
                dtype = dtype_map.get(sampwidth)
                if dtype is None:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")

                audio = np.frombuffer(audio_bytes, dtype=dtype)
                if dtype == np.uint8:
                    audio = audio.astype(np.float32)
                    audio = (audio - 128.0) / 128.0
                else:
                    max_val = float(np.iinfo(dtype).max)
                    if not max_val:
                        raise ValueError("Invalid max value for dtype")
                    audio = audio.astype(np.float32) / max_val

                if channels > 1:
                    audio = audio.reshape(-1, channels)

            sd.stop()
            sd.play(audio, sample_rate, device=self.output_device, blocking=self.blocking_playback)
            if self.on_playback_start:
                self.on_playback_start(segment.path, time.time())
            if self.blocking_playback:
                sd.wait()
            return True
        except Exception as exc:
            print(f"[TTS] Direct playback failed for {segment.path}: {exc}")
            return False

    def _play_via_subprocess(self, segment: SpeechSegment) -> bool:
        try:
            if self.on_playback_start:
                self.on_playback_start(segment.path, time.time())
            cmd = list(self.playback_cmd)
            if isinstance(self.output_device, str) and cmd:
                if (
                    cmd[0] == "paplay"
                    and not any(str(arg).startswith("--device=") for arg in cmd[1:])
                ):
                    cmd = cmd + [f"--device={self.output_device}"]
                elif cmd[0] == "aplay" and "-D" not in cmd:
                    cmd = cmd[:1] + ["-D", self.output_device] + cmd[1:]
            if any("{file}" in str(part) for part in cmd):
                resolved = [str(part).replace("{file}", segment.path) for part in cmd]
                subprocess.run(resolved, check=True, env=self._playback_env)
            elif cmd and cmd[-1] == "-" and segment.raw is not None:
                subprocess.run(
                    cmd,
                    input=segment.raw,
                    check=True,
                    env=self._playback_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
            else:
                subprocess.run(cmd + [segment.path],
                               check=True,
                               env=self._playback_env,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL
                               )
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[TTS] Subprocess playback failed (exit {exc.returncode}) for {segment.path}: {exc}")
            return False
        except Exception as exc:
            print(f"[TTS] Subprocess playback failed for {segment.path}: {exc}")
            return False


    def generate_and_queue(self, text: str, segment_id: int) -> Optional[Future]:
        clean_text = " ".join((text or "").split())
        if not clean_text:
            return None
        return self.executor.submit(self._generate_speech, clean_text, segment_id)


    def _generate_speech(self, text: str, segment_id: int) -> Optional[SpeechSegment]:

        if not self.model_path.exists():
            print(f"[TTS] Piper model not found: {self.model_path}")
            return None

        utterance = " ".join((text or "").split())
        if not utterance:
            return None
        
        with self._piper_lock:
            self._ensure_piper()
            proc = self._piper_proc
            
            if not proc or proc.poll() is not None:
                print("[TTS] Piper proc is not running")
                return None 
            
            try:
                proc.stdin.write((utterance + "\n" ).encode("utf-8"))
                proc.stdin.flush()
            except Exception as e:
                print(f"[TTS] failed writing to piper: {e}")
                
                self._restart_piper_and_retry(utterance)
                proc = self._piper_proc
                
                if not proc or proc.poll() is not None:
                    return None 
            
            payload = self._read_pcm_until_idle(proc.stdout, idle_ms=80)
            if not payload or self._looks_like_text(payload):
                print("[TTS] Piper did not produce PCM; check --output_raw '-'")
                return None 
        
            info = self._voice_info
            segment = SpeechSegment(
                path = "",
                raw = payload,
                sample_rate = info.sample_rate or 22050,
                channels = info.channels or 1,
                text = utterance,
                
            )
            
            
            
            self.speech_queue.put(segment)
            return segment 
            
    
    def _restart_piper_and_retry(self,utterance:str):
        try:
            if self._piper_proc:
                self._piper_proc.kill()
        except Exception:
            pass
        
        self._piper_proc= None
        self._ensure_piper()
        if self._piper_proc and self._piper_proc.poll() is None:
            try:
                self._piper_proc.stdin.write((utterance + "\n").encode("utf-8"))
                self._piper_proc.stdin.flush()
            except Exception as e : 
                print(f"[TTS] Piper failed to retry : {e} ")
                
    def _read_pcm_until_idle(self,stdout,idle_ms = 120,max_ms=30_000)->bytes:
        start = time.time()
        idle_s = idle_ms/1000.0
        deadline = start + (max_ms/1000.0)
        
        chunks = []
        
        stdout_fd = stdout.fileno()
        sel = selectors.DefaultSelector()
        
        sel.register(stdout_fd, selectors.EVENT_READ)
        
        try:
            any_bytes = False
            while time.time() < deadline:
                events = sel.select(timeout=idle_s)
                if not events:
                    if not any_bytes:
                        continue
                    break
                for _key,_mask in events :
                    data = os.read(stdout_fd, 8192)
                    if not data:
                        payload = b"".join(chunks)
                        return payload if (len(payload) & 1) == 0 else payload[:-1]
                    
                    chunks.append(data)
                    any_bytes = True
            payload = b"".join(chunks)
            return payload if (len(payload)&1) == 0 else payload[:-1]
        finally:
            try:
                sel.unregister(stdout_fd)
            except Exception:
                pass
            
            try:
                sel.close()
            except Exception:
                pass
        

    @staticmethod
    def _looks_like_text(payload: bytes) -> bool:
        """Heuristic check to detect when Piper prints text instead of PCM."""

        if not payload:
            return False

        sample = payload[:64]
        printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in sample)
        # Random PCM rarely decodes into predominantly printable ASCII. Treat a
        # mostly printable prefix as an indication that Piper emitted text/logs.
        return printable >= max(10, len(sample) * 0.6)

    def stop(self) -> None:
        self.playing = False
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None
        try:
            with self._piper_lock:
                if self._piper_proc and self._piper_proc.poll() is None :
                    self._piper_proc.terminate()
                    try:
                        self._piper_proc.wait(timeout=0.5)
                    except Exception:
                        pass
        except Exception:
            pass
        self._piper_proc = None
        
        try:
            if self.out_dir:
                import shutil
                shutil.rmtree(self.out_dir,ignore_errors=True)
        finally:
            self.out_dir = None
        self.executor.shutdown(wait=False)
    
