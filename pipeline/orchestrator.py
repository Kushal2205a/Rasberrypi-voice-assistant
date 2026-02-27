from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Iterable, Dict, List, Deque, Tuple, Set
from collections import deque
from concurrent.futures import Future
import time 

import os, threading, queue, re, math, psutil, numpy as np 

from config import (
    CHUNK_DURATION,
    SAMPLE_RATE,
    WHISPER_EXE,
    WHISPER_MODEL,
    PIPER_MODEL_PATH,
    DEFAULT_SILENCE_THRESHOLD,
    DEFAULT_SILENCE_TIMEOUT,
    MODEL,
    MODEL_LITE,
    MODEL_PRO,
    SWITCH_CODEWORD,
    SWITCH_REQUIRE_CODEWORD,
)
from recorder import StreamingRecorder
"""
try:
    from stt_faster import PersistentWhisperSTT as ParallelSTT
except Exception:
    from stt import ParallelSTT
"""
from tts import BufferedTTS

#from llm_model import StreamingLLM, speak_text_timed
import os
from llm_model import speak_text_timed
from llm_ollama_adapter import OllamaStreamingLLM

from stt_vosk import PersistentVoskSTT as ParallelSTT


@dataclass

class PendingOutput:
    timestamp: float
    segments_expected: int
    latency_recorded: bool = False


@dataclass

class PipelineStats:
    stt_chunks: int = 0
    llm_responses: int = 0
    tts_segments: int = 0
    total_latency: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    stt_latencies: List[float] = field(default_factory=list)
    llm_latencies: List[float] = field(default_factory=list)
    tts_generation_latencies: List[float] = field(default_factory=list)
    input_to_output_latencies: List[float] = field(default_factory=list)
    pending_outputs: Deque[PendingOutput] = field(default_factory=deque)

    recording_stop_time: Optional[float] = None
    recording_to_first_llm_latency: Optional[float] = None
    recording_stop_to_first_tts_latency: Optional[float] = None
    recording_start_to_first_tts_latency: Optional[float] = None


class ParallelVoiceAssistant:
    """Coordinate the streaming recorder, STT workers, llama110, and Piper TTS."""

    def __init__(
        self,
        chunk_duration: float = CHUNK_DURATION,
        sample_rate: int = SAMPLE_RATE,
        stt_workers: int = 2,

        whisper_exe: Path = WHISPER_EXE,
        whisper_model: Path = WHISPER_MODEL,
        whisper_threads: Optional[int] = None,
        emit_stt_partials: bool = False,
        piper_model_path: Path = PIPER_MODEL_PATH,
        llama_kwargs: Optional[Dict[str, Any]] = None,

        output_device: Optional[Any] = None,
        playback_cmd: Optional[Iterable[str]] = None,
        use_subprocess_playback: bool = True,
        silence_timeout: float = DEFAULT_SILENCE_TIMEOUT,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
        whisper_server: Optional[str] = None,
        
        model_lite: str = MODEL_LITE,
        model_pro: str = MODEL_PRO,
        switch_codeword: str = SWITCH_CODEWORD,
        switch_require_codeword: bool = SWITCH_REQUIRE_CODEWORD,
        switch_reset_history: bool = True,
        
        

    ) -> None:
        self._chunk_duration = float(chunk_duration)
        self.recorder = StreamingRecorder(chunk_duration=chunk_duration, sample_rate=sample_rate)
        
        
       
        
        if whisper_server:
            # HTTP (persistent) STT
            from stt import ParallelSTTHTTP
            self.stt = ParallelSTTHTTP(
                num_workers=stt_workers,
                sample_rate=sample_rate,
                server_url=whisper_server,
                emit_partials=True,
            )
        else:
            self.stt = ParallelSTT(
                num_workers=stt_workers,
                sample_rate=sample_rate,
                emit_partials=True,              # keep partials on (good for UX + LLM overlap)
                vad_aggressiveness=0,            # 0..3, tune higher in noisy rooms
                min_partial_interval_ms=120    
            )
        self._stt_streaming = bool(getattr(self.stt, "IS_STREAMING", False))
        
        #self.llm = StreamingLLM(llama_kwargs=llama_kwargs)
        initial_model = os.getenv("OLLAMA_MODEL") or (model_lite or MODEL)
        self.llm = OllamaStreamingLLM(
        model=initial_model,
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        keep_alive=os.getenv("LLM_KEEP_ALIVE", "30m"),
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "256")),
        num_thread=int(os.getenv("OLLAMA_NUM_THREAD", "4")),
        )
        
        
        self._model_lite = (model_lite or MODEL_LITE).strip()
        self._model_pro = (model_pro or MODEL_PRO).strip()
        self._switch_codeword = (switch_codeword or "").strip().lower()
        self._switch_require_codeword = bool(switch_require_codeword)
        self._switch_reset_history = bool(switch_reset_history)

        # --- Command (model switch) audio buffer ---
        # Keep a short rolling window of raw audio so we can re-decode switch commands with a restricted grammar.
        self._cmd_audio_window_s = 2.5
        self._cmd_audio: Deque[np.ndarray] = deque(
            maxlen=max(1, int(self._cmd_audio_window_s / max(0.01, self._chunk_duration)))
        )


        
        self.tts = BufferedTTS(
            model_path=piper_model_path,
            playback_cmd=["aplay", "-t", "raw", "-f", "S16_LE", "-r", "22050", "-c", "1", "-"],          # defaults to: aplay -t raw ...
            output_device=output_device,
            use_subprocess=True,  # keep True for Pi
            on_playback_start=self._on_tts_playback_start,
            on_playback_error=self._on_tts_playback_error,
        )

        # --- LLM → TTS streaming buffer state ---
        self._spoken_buf: List[str] = []
        self._spoken_words: int = 0
        self._segment_id: int = 0
        self.stt_futures: "queue.Queue[Tuple[int, Future, float]]" = queue.Queue()
        self.llm_futures: "queue.Queue[Tuple[Future, float, float]]" = queue.Queue()
        self.stats = PipelineStats()
        self._stt_done = threading.Event()
        self._pending_lock = threading.Lock()

        self._activity_lock = threading.Lock()
        self._stop_lock = threading.Lock()

        self._activity_event = threading.Event()
        self._last_voice_time = time.time()
        self._has_detected_speech = False
        self._first_voice_time: Optional[float] = None
        self._recording_stop_time: Optional[float] = None
        self._silence_timeout = float(silence_timeout)
        self._silence_threshold = float(silence_threshold)

        self._stop_requested = False
        self._stop_reason: Optional[str] = None
        self._consecutive_silent_chunks = 0
        self._silent_chunks_before_stop = max(2,int(math.ceil(self._silence_timeout/max(0.1,self._chunk_duration))))
        
        self._noise_blacklist = {
            "wind blowing",
            "bird chirping",
            "blank_audio",
            "blank audio",
            "[blank_audio]",
            "(wind blowing)",
            "(bird chirping)",
        }

        # optional: regex to catch bracket/parenthesis or small variations
        self._noise_regex = re.compile(
            r"\b(blank[_ ]?audio|wind blowing|bird chirping)\b", flags=re.IGNORECASE
        )

        self._tts_futures_lock = threading.Lock()
        self._pending_tts_futures: Set[Future] = set()

        self._chunk_activity: Dict[int, bool] = {}
        self._awaiting_transcript_chunks = 0
        self._awaiting_transcript_started_at: Optional[float] = None
        self._awaiting_transcript_chunk_limit = max(2, int(math.ceil(1.0 / max(0.1, self._chunk_duration))))
        self._awaiting_transcript_timeout = max(0.65, self._chunk_duration * 1.0)
        self._stt_flush_in_progress = False
        self._next_finalize_id = 1_000_000
        self._active_flush_ids: Set[int] = set()
        
        self._flushed_since_last_speech = False
        self.espeak_tts = lambda text, segment_id: speak_text_timed(text)
        
        self.llm.set_stream_callback(self._on_llm_token)




    def _register_activity(self) -> None:
        now = time.time()
        with self._activity_lock:
            if not self._has_detected_speech:
                self._first_voice_time = now
            self._has_detected_speech = True
            self._last_voice_time = now
        self._activity_event.set()

    def _reset_awaiting_transcript_state(self) -> None:
        self._awaiting_transcript_chunks = 0
        self._awaiting_transcript_started_at = None

    def _should_force_intermediate_transcription(self) -> bool:
        if self._stt_flush_in_progress:
            return False
        if self._awaiting_transcript_chunks >= self._awaiting_transcript_chunk_limit:
            return True
        if self._awaiting_transcript_started_at is not None:
            elapsed = time.time() - self._awaiting_transcript_started_at
            if elapsed >= self._awaiting_transcript_timeout:
                return True
        return False

    def _queue_intermediate_transcription(self, reason: str) -> None:
        # Never force a mid-utterance finalize for streaming backends (e.g., Vosk)
        if getattr(self, "_stt_streaming", False):
            return

        if self._stt_flush_in_progress:
            return

        flush_id = self._next_finalize_id
        future = self.stt.finalize(flush_id, mark_final=True)
        if future is None:
            return

        self._stt_flush_in_progress = True
        self._active_flush_ids.add(flush_id)
        self.stt_futures.put((flush_id, future, time.time()))
        print(reason)
        self._reset_awaiting_transcript_state()
        self._next_finalize_id += 1

    def _is_silent_chunk(self, audio_chunk: np.ndarray) -> bool:
        if audio_chunk.size == 0:
            return True
        audio_view = np.asarray(audio_chunk, dtype=np.int16)
        if audio_view.ndim > 1:
            audio_view = audio_view.reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(audio_view.astype(np.float32)))))
        return rms < self._silence_threshold

    def _is_speech_chunk(self, audio_chunk: np.ndarray) -> bool:
        # Cheap RMS first
        if audio_chunk is None or audio_chunk.size == 0:
            return False
        audio_view = np.asarray(audio_chunk, dtype=np.int16).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(audio_view.astype(np.float32)))))

        # Hysteresis band around your threshold to avoid VAD calls most of the time
        low = self._silence_threshold * 0.85
        high = self._silence_threshold * 1.60

        if rms < low:
            return False
        if rms > high:
            return True

        # Only now call VAD (expensive)
        fn = getattr(self.stt, "is_speech", None)
        if callable(fn):
            v = fn(audio_chunk)
            if isinstance(v, bool):
                return v
        return rms >= self._silence_threshold



    def _request_stop(self, reason: str) -> None:
        with self._stop_lock:
            if self._stop_requested:
                return
            self._stop_requested = True
            self._stop_reason = reason
            now = time.time()
            if self._recording_stop_time is None:
                self._recording_stop_time = now
            self.stats.recording_stop_time = self._recording_stop_time
        print(reason)
        self.recorder.stop()
        self.recorder.clear_queue()
        self._activity_event.set()

    def _handle_silent_audio_chunk(self) -> None:
        if self._stop_requested:
            return
        self._consecutive_silent_chunks += 1
        if self._consecutive_silent_chunks < self._silent_chunks_before_stop:
            return
        if self._has_detected_speech:
            message = (
                f"[MAIN] No speech detected for {self._consecutive_silent_chunks} consecutive chunks; stopping recorder."
            )
        else:
            message = (
                f"[MAIN] Silence persisted for {self._consecutive_silent_chunks} consecutive chunks; stopping recorder."
            )
        self._request_stop(message)


    def _reference_timestamp_for_output(self, input_timestamp: float) -> float:
        candidates = [input_timestamp]
        with self._activity_lock:
            candidates.append(self._last_voice_time)
        if self._recording_stop_time is not None:
            candidates.append(self._recording_stop_time)
        return max(candidates)

    def run(self, duration: Optional[float] = None) -> None:
        max_duration = duration if duration and duration > 0 else None
        if max_duration is not None:
            print(
                f"[MAIN] Starting streaming assistant (max {max_duration:.1f}s, "
                f"silence timeout {self._silence_timeout:.1f}s)"
            )
        else:
            print(f"[MAIN] Starting streaming assistant (silence timeout {self._silence_timeout:.1f}s)")

        start_time = time.time()
        self.stats.start_time = start_time
        self.stats.recording_stop_time = None
        self.stats.recording_to_first_llm_latency = None
        self.stats.recording_start_time = start_time
        self.stats.recording_stop_to_first_tts_latency = None

        self._stt_done.clear()
        self._activity_event.clear()
        self._cmd_audio.clear()
        with self._activity_lock:
            self._has_detected_speech = False
            self._last_voice_time = start_time
            self._first_voice_time = None
        self._recording_stop_time = None

        with self._tts_futures_lock:
            self._pending_tts_futures.clear()

        with self._stop_lock:
            self._stop_requested = False
            self._stop_reason = None
        self._consecutive_silent_chunks = 0
        self._reset_awaiting_transcript_state()
        self._stt_flush_in_progress = False
        self._active_flush_ids.clear()
        self._next_finalize_id = 1_000_000


        if self.tts:
            self.tts.start_playback()
        self.recorder.start()

        stt_thread = threading.Thread(target=self._stt_pipeline, name="STTPipeline", daemon=True)
        llm_thread = threading.Thread(target=self._llm_pipeline, name="LLMPipeline", daemon=True)

        stt_thread.start()
        llm_thread.start()

        try:

            while True:
                if self._stop_requested:
                    break
                now = time.time()
                if max_duration is not None and now - start_time >= max_duration:
                    self._request_stop(
                        f"[MAIN] Max duration {max_duration:.1f}s reached; wrapping up."
                    )

                    break

                with self._activity_lock:
                    has_voice = self._has_detected_speech
                    last_voice = self._last_voice_time

                if has_voice and (now - last_voice) >= self._silence_timeout:

                    self._request_stop(
                        f"[MAIN] Detected {self._silence_timeout:.1f}s of silence; stopping recorder."
                    )

                    break

                wait_timeout = 0.5
                if has_voice:
                    remaining = self._silence_timeout - (now - last_voice)
                    wait_timeout = max(0.1, min(0.5, remaining))

                triggered = self._activity_event.wait(timeout=wait_timeout)
                if triggered:
                    self._activity_event.clear()

        except KeyboardInterrupt:
            print("\n[MAIN] Interrupted by user")
            self._request_stop("[MAIN] Interrupted by user; stopping recorder.")
        finally:
            self.recorder.stop()
            if self._recording_stop_time is None:
                self._recording_stop_time = time.time()

            self.stats.recording_stop_time = self._recording_stop_time

        stt_thread.join()

        # Decide whether to run a final STT pass
        finalize_future = self.stt.finalize(self.stats.stt_chunks + 1, mark_final=True)
        if finalize_future is not None:
            self.stt_futures.put((self.stats.stt_chunks + 1, finalize_future, time.time()))
            # Drain so LLM gets the final transcript immediately
            self._process_stt_results(wait=True)


        # Signal the LLM pipeline that no more text is coming once final STT results are queued.
        self._stt_done.set()
        llm_thread.join()

        self.stt.shutdown()
        self.llm.shutdown()
        self._wait_for_tts_completion()
        if self.tts:
            self.tts.stop()


        elapsed = time.time() - start_time
        self._print_stats(elapsed)

    # --------------------------- Pipelines -----------------------

    def _stt_pipeline(self) -> None:
        chunk_id = 0
        try:
            while self.recorder.recording or not self.recorder.chunk_queue.empty():
                audio_chunk = self.recorder.get_chunk(timeout=0.5)
                if audio_chunk is None:
                    self._process_stt_results(wait=False)
                    continue

                # keep raw audio for command re-decode
                self._cmd_audio.append(audio_chunk.copy())


                if self._stop_requested and not self.recorder.recording:
                    self.recorder.clear_queue()
                    self._process_stt_results(wait=False)
                    break

                # remember recorder sample rate for VAD logic if needed
                setattr(self, "_recorder_sample_rate", self.recorder.sample_rate)

                
                is_speech = self._is_speech_chunk(audio_chunk)
                self._chunk_activity[chunk_id] = is_speech

                if is_speech:
                    self._register_activity()
                    self._consecutive_silent_chunks = 0
                else:
                    self._handle_silent_audio_chunk()

    
                if is_speech or self._has_detected_speech:
                    future = self.stt.submit_chunk(audio_chunk, chunk_id)
                else:
                    future = self.stt.empty_future(chunk_id)
                self.stt_futures.put((chunk_id, future, time.time()))
                self.stats.stt_chunks += 1
                chunk_id += 1
                self._process_stt_results(wait=False)


            self._process_stt_results(wait=True)
        finally:
            # Ensure any exceptions don't leave futures undispatched.
            pass

    def _process_stt_results(self, wait: bool) -> None:

        pending: List[Tuple[int, Future, float]] = []
        while not self.stt_futures.empty():
            chunk_id, future, start_time = self.stt_futures.get()

            if wait or future.done():
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[STT Pipeline] Future for chunk {chunk_id} failed: {exc}")
                    if chunk_id in self._active_flush_ids:
                        self._active_flush_ids.discard(chunk_id)
                        self._stt_flush_in_progress = False
                        self._flushed_since_last_speech = True
                        self._reset_awaiting_transcript_state()
                    continue
            else:

                pending.append((chunk_id, future, start_time))

                continue

            if not result:
                continue

            res_chunk_id = result.get("chunk_id", chunk_id)
            if chunk_id in self._active_flush_ids or res_chunk_id in self._active_flush_ids:
                self._active_flush_ids.discard(chunk_id)
                self._active_flush_ids.discard(res_chunk_id)
                self._stt_flush_in_progress = False
                self._flushed_since_last_speech = True


            latency = max(0.0, time.time() - start_time)
            self.stats.stt_latencies.append(latency)


            text = (result.get("text") or "").strip()
            is_final = bool(result.get("is_final"))

            # Normalize for noise checks
            normalized = (text or "").strip().lower()
            had_activity = self._chunk_activity.pop(res_chunk_id, False)

            # Check blacklist exact matches first, then regex for variants
            is_noise = False
            if normalized:
                if normalized in self._noise_blacklist:
                    is_noise = True
                elif self._noise_regex.search(normalized):
                    is_noise = True

            if is_noise or not normalized:
                if had_activity and not normalized:
                    # Speech energy was observed for this chunk, but Whisper did not

                    # return any transcript yet. Treat as ongoing speech for a short
                    # period, but force an intermediate transcription if it persists.

                    print(
                        f"[STT] Chunk {res_chunk_id}: (speech detected, awaiting transcription)"
                    )

                    self._awaiting_transcript_chunks += 1
                    if self._awaiting_transcript_started_at is None:
                        self._awaiting_transcript_started_at = time.time()
                    """if self._awaiting_transcript_chunks >= self._silent_chunks_before_stop:
                        # Ambient noise can keep RMS high while Whisper emits nothing; treat
                        # this as silence so we still stop after the configured no-speech chunks.
                        self._reset_awaiting_transcript_state()
                        self._handle_silent_audio_chunk()
                        continue"""
                    if self._should_force_intermediate_transcription():
                        elapsed = 0.0
                        if self._awaiting_transcript_started_at is not None:
                            elapsed = time.time() - self._awaiting_transcript_started_at
                        self._queue_intermediate_transcription(
                            f"[STT] Forcing intermediate transcription after {elapsed:.1f}s without text"
                        )

                    continue
                if self._has_detected_speech and not self._stt_flush_in_progress and not self._flushed_since_last_speech:
                    self._queue_intermediate_transcription("[STT] Silence boundary -> flushing buffered speech")
                    self._flushed_since_last_speech = True
                # Treat as silent/noise: increment silent-chunk logic and DO NOT feed to LLM
                print(f"[STT] Chunk {res_chunk_id}: {text} (treated as noise/empty)")
                self._reset_awaiting_transcript_state()
                continue

            # Otherwise it's valid speech
            self._reset_awaiting_transcript_state()
            self._register_activity()
            self._consecutive_silent_chunks = 0
            self._flushed_since_last_speech = False

            print(f"[STT] Chunk {res_chunk_id}: {text}")

            if is_final and self._maybe_handle_model_switch(text):
                continue

            llm_trigger_time = time.time()
            llm_future = self.llm.process_incremental(text, is_final=is_final)
            if llm_future is not None:
                reference_time = self._reference_timestamp_for_output(llm_trigger_time)
                self.llm_futures.put((llm_future, llm_trigger_time, reference_time))


        for item in pending:
            self.stt_futures.put(item)

    def _llm_pipeline(self) -> None:
        segment_id = 0
        while not self._stt_done.is_set() or not self.llm_futures.empty():
            try:
                llm_future, _submit_time, reference_timestamp = self.llm_futures.get(timeout=0.5)
            except queue.Empty:
                continue

            response = ""
            try:
                response = llm_future.result(timeout=300)
                response_ready_time = time.time()
            except Exception as exc:
                print(f"[LLM Pipeline] Error: {exc}")
                continue

            latency = max(0.0, response_ready_time - reference_timestamp)
            self.stats.llm_latencies.append(latency)

            if self.stats.recording_to_first_llm_latency is None:
                if self._recording_stop_time is not None:
                    self.stats.recording_to_first_llm_latency = max(
                        0.0, response_ready_time - self._recording_stop_time
                    )
                else:
                    self.stats.recording_to_first_llm_latency = max(
                        0.0, response_ready_time - reference_timestamp
                    )

            response = (response or "").strip()
            if not response:
                continue

            print(f"[LLM] Response: {response[:120]}{'...' if len(response) > 120 else ''}")
            self.stats.llm_responses += 1

            # --- Streaming TTS path: flush any leftover words the token-callback hasn’t spoken ---
            if self.tts:
                if self._spoken_buf:
                    residual = "".join(self._spoken_buf).strip()
                    if residual:
                        self.tts.generate_and_queue(residual, self._segment_id)
                        self.stats.tts_segments += 1
                        self._segment_id += 1
                    self._spoken_buf.clear()
                    self._spoken_words = 0
                continue  # streaming already handled speaking

            # --- Fallback (no Piper): sentence-by-sentence via espeak ---
            sentences = self._split_sentences(response) or [response]
            for sentence in sentences:
                started_at = time.time()

                if (self.stats.recording_start_to_first_tts_latency is None
                        and getattr(self.stats, "recording_start_time", None) is not None):
                    self.stats.recording_start_to_first_tts_latency = max(
                        0.0, started_at - self.stats.recording_start_time
                    )
                if self.stats.recording_stop_to_first_tts_latency is None and self._recording_stop_time is not None:
                    self.stats.recording_stop_to_first_tts_latency = max(
                        0.0, started_at - self._recording_stop_time
                    )

                self.stats.input_to_output_latencies.append(max(0.0, started_at - reference_timestamp))

                self.espeak_tts(sentence, segment_id)
                self.stats.tts_segments += 1

                try:
                    if getattr(self, "recorder", None) and getattr(self.recorder, "recording", False):
                        self._request_stop("[MAIN] Stopping recorder during TTS to avoid feedback.")
                except Exception:
                    pass

                segment_id += 1


    def _queue_tts_confirmation(self, text: str) -> None:
        """Queue a TTS confirmation and properly track the future so _wait_for_tts_completion waits for it."""
        if not getattr(self, "tts", None):
            return
        try:
            submit_time = time.time()
            fut = self.tts.generate_and_queue(text, self._segment_id)
            if fut is None:
                return
            self.stats.tts_segments += 1
            self._segment_id += 1

            pending_ts = self._reference_timestamp_for_output(submit_time)
            pending = PendingOutput(timestamp=pending_ts, segments_expected=1)
            with self._pending_lock:
                self.stats.pending_outputs.append(pending)

            with self._tts_futures_lock:
                self._pending_tts_futures.add(fut)

            fut.add_done_callback(
                lambda f, start=submit_time, p=pending: self._on_tts_generated(f, start, p)
            )
        except Exception:
            pass

    def _decode_switch_command_from_audio(self) -> str:
        """Second-pass decode for switch commands using a restricted Vosk grammar."""
        try:
            chunks = list(self._cmd_audio)
            if not chunks:
                return ""
            audio = np.concatenate(chunks, axis=0)
        except Exception:
            return ""

        # Limit to the last N seconds
        try:
            max_samples = int(self.recorder.sample_rate * float(self._cmd_audio_window_s))
            if audio.shape[0] > max_samples:
                audio = audio[-max_samples:]
        except Exception:
            pass

        grammar = [
            "switch to smart", "switch smart",
            "switch to fast", "switch fast",
            "switch to pro", "switch pro",
            "switch to lite", "switch lite",
            "smart", "fast", "pro", "lite", "switch",
        ]

        fn = getattr(self.stt, "decode_grammar", None)
        if callable(fn):
            return (fn(audio, grammar) or "").strip()
        return ""

    def _maybe_handle_model_switch(self, text: str) -> bool:
        """Detect and execute 'switch to pro/lite' commands from a transcript.

        Rules (by default):
          - Accept if transcript contains a switch-verb (switch/mode/model/change) AND pro|lite
          - OR accept if it contains the configured codeword AND pro|lite
          - If SWITCH_REQUIRE_CODEWORD is True, the codeword is mandatory.
        """
        if not text:
            return False

        norm = re.sub(r"[^a-z0-9\s]+", " ", text.lower()).strip()
        if not norm:
            return False
        tokens = norm.split()
    
        if tokens and tokens[0] == "switch":
            current = (self.llm.get_model() or "").strip() if hasattr(self.llm, "get_model") else os.getenv("OLLAMA_MODEL", "")
            current = (current or "").strip()

        
            if current == self._model_pro:
                target_model, label = self._model_lite, "Lite"
            else:
                target_model, label = self._model_pro, "Pro"

            try:
                self.llm.set_model(target_model, reset_history=self._switch_reset_history)
            except Exception as exc:
                print(f"[Switch] Failed to switch model: {exc}")
                return False

            os.environ["OLLAMA_MODEL"] = target_model
            print(f"[Switch] Model toggled to {label}: {target_model}")

            self._queue_tts_confirmation(f'Switched to "{target_model}".')
            return True
        wants_pro = ("smart" in tokens) or ("pro" in tokens)
        wants_lite = ("fast" in tokens) or ("faster" in tokens) or ("lite" in tokens) or ("light" in tokens)

        has_switch_verb = any(t in tokens for t in ("switch", "switched", "change", "mode", "model"))
        has_codeword = bool(self._switch_codeword) and (self._switch_codeword in tokens)

        if self._switch_require_codeword:
            if not has_codeword:
                return False
        else:
            # Require *some* command framing to avoid accidental triggers in normal speech.
            if not (has_switch_verb or has_codeword):
                return False

        # If open-vocab decode didn't confidently capture smart/fast, try a restricted-grammar re-decode.
        if not (wants_pro or wants_lite):
            decoded = self._decode_switch_command_from_audio()
            if decoded:
                norm2 = re.sub(r"[^a-z0-9\s]+", " ", decoded.lower()).strip()
                t2 = norm2.split()
                wants_pro = ("smart" in t2) or ("pro" in t2)
                wants_lite = ("fast" in t2) or ("faster" in t2) or ("lite" in t2) or ("light" in t2)

        if not (wants_pro or wants_lite):
            return False
        target_model = self._model_pro if wants_pro else self._model_lite
        label = "Pro" if wants_pro else "Lite"

        try:
            self.llm.set_model(target_model, reset_history=self._switch_reset_history)
        except Exception as exc:
            print(f"[Switch] Failed to switch model: {exc}")
            return False
        os.environ["OLLAMA_MODEL"] = target_model
        print(f"[Switch] Model switched to {label}: {target_model}")

        # Speak a short confirmation so the user knows it worked.
        self._queue_tts_confirmation(f'Switched to "{target_model}".')
        return True

    def _on_tts_generated(self, future: Future, start_time: float, pending: PendingOutput) -> None:
        try:
            result = future.result()
        except Exception as exc:
            print(f"[TTS Pipeline] Generation failed: {exc}")
            self._handle_failed_tts_generation(pending)
        else:
            if result:
                latency = max(0.0, time.time() - start_time)
                self.stats.tts_generation_latencies.append(latency)
            else:
                self._handle_failed_tts_generation(pending)
        finally:
            with self._tts_futures_lock:
                self._pending_tts_futures.discard(future)

    def _handle_failed_tts_generation(self, pending: PendingOutput) -> None:
        with self._pending_lock:
            pending.segments_expected = max(0, pending.segments_expected - 1)
            if pending.segments_expected == 0:
                try:
                    self.stats.pending_outputs.remove(pending)
                except ValueError:
                    pass

    def _on_tts_playback_start(self, file_path: str, started_at: float) -> None:
        
        if (self.stats.recording_start_to_first_tts_latency is None and getattr(self.stats, "recording_start_time", None) is not None):
                self.stats.recording_start_to_first_tts_latency = max(
                0.0, started_at - self.stats.recording_start_time
            )

        if (
            self.stats.recording_stop_to_first_tts_latency is None
            and self._recording_stop_time is not None
        ):
            self.stats.recording_stop_to_first_tts_latency = max(
                0.0, started_at - self._recording_stop_time
            )

        with self._pending_lock:
            while self.stats.pending_outputs:
                pending = self.stats.pending_outputs[0]
                if pending.segments_expected <= 0:
                    self.stats.pending_outputs.popleft()
                    continue

                if not pending.latency_recorded:
                    latency = max(0.0, started_at - pending.timestamp)
                    self.stats.input_to_output_latencies.append(latency)
                    pending.latency_recorded = True

                pending.segments_expected = max(0, pending.segments_expected - 1)
                if pending.segments_expected == 0:
                    self.stats.pending_outputs.popleft()
                break
        try:
            if getattr(self, "recorder", None) and getattr(self.recorder, "recording", False):
                self._request_stop("[MAIN] Stopping recorder during TTS to avoid feedback.")
        except Exception as e:
            self._log(f"[TTS] playback_start hook error: {e}")
    
    
    def _on_llm_token(self, tok: str) -> None:
        """Called by Ollama adapter for each streamed token/chunk."""
        if not tok:
            return
        # buffer words; speak at sentence end or every ~12 words
        self._spoken_buf.append(tok)
        self._spoken_words += tok.count(" ") + 1
        should_flush = any(tok.endswith(p) for p in (".", "!", "?")) or (self._spoken_words >= 12)

        if should_flush and self.tts:
            text = "".join(self._spoken_buf).strip()
            self._spoken_buf.clear()
            self._spoken_words = 0
            if not text:
                return

            submit_time = time.time()
            fut = self.tts.generate_and_queue(text, self._segment_id)
            if fut is None:
                return

            self.stats.tts_segments += 1
            self._segment_id += 1

            # pending output: one segment for this flush
            pending_ts = self._reference_timestamp_for_output(submit_time)
            pending = PendingOutput(timestamp=pending_ts, segments_expected=1)
            with self._pending_lock:
                self.stats.pending_outputs.append(pending)

            with self._tts_futures_lock:
                self._pending_tts_futures.add(fut)

            fut.add_done_callback(
                lambda f, start=submit_time, p=pending: self._on_tts_generated(f, start, p)
            )

    def _on_tts_playback_error(self) -> None:
        with self._pending_lock:
            if not self.stats.pending_outputs:
                return

            pending = self.stats.pending_outputs[0]
            pending.segments_expected = max(0, pending.segments_expected - 1)
            if pending.segments_expected == 0:
                self.stats.pending_outputs.popleft()

    # --------------------------- Helpers -------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        words = text.split()
        
        out: List[str] = []
        cur: List[str] = []
        MAX_WORDS = 12
        for w in words :
            cur.append(w)
            if len(cur) >= MAX_WORDS or any(w.endswith(p) for p in (".","!","?")):
                out.append(" ".join(cur))
                cur = []
        
        if cur:
            out.append(" ".join(cur))
            
        return out
        
    @staticmethod
    def _print_latency_summary(label: str, samples: List[float]) -> None:
        if not samples:
            print(f"{label}: n/a")
            return

        avg = sum(samples) / len(samples)
        min_val = min(samples)
        max_val = max(samples)
        print(f"{label}: avg {avg:.2f}s (min {min_val:.2f}s, max {max_val:.2f}s, n={len(samples)})")


    def _print_stats(self, elapsed: float) -> None:
        print("\n--- PIPELINE STATS ---")
        print(f"Runtime: {elapsed:.2f}s")
        print(f"STT chunks processed: {self.stats.stt_chunks}")
        print(f"LLM responses generated: {self.stats.llm_responses}")
        print(f"TTS segments queued: {self.stats.tts_segments}")

        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Memory usage: {mem_mb:.1f} MB")

        self._print_latency_summary("STT chunk latency", list(self.stats.stt_latencies))
        self._print_latency_summary("LLM response latency", list(self.stats.llm_latencies))
        if self.stats.recording_to_first_llm_latency is not None:
            print(
                "Recording -> first LLM response: "
                f"{self.stats.recording_to_first_llm_latency:.2f}s"
            )
        else:
            print("Recording stop -> first LLM response: n/a")

        if self.stats.recording_stop_to_first_tts_latency is not None:
            print(
                "Recording stop -> first TTS audio: "
                f"{self.stats.recording_stop_to_first_tts_latency:.2f}s"
            )
        else:
            print("Recording stop -> first TTS audio: n/a")
        
       

        self._print_latency_summary("TTS generation latency", list(self.stats.tts_generation_latencies))
        self._print_latency_summary("Input -> first audio gap", list(self.stats.input_to_output_latencies))

        print("----------------------\n")

    def _wait_for_tts_completion(self, timeout: float = 15.0) -> None:
        if self.stats.tts_segments == 0 or self.tts is None :
            return

        deadline = time.time() + max(0.0, timeout)
        while True:
            with self._tts_futures_lock:
                pending_futures = len(self._pending_tts_futures)

            with self._pending_lock:
                pending_outputs = sum(
                    1 for pending in self.stats.pending_outputs if pending.segments_expected > 0
                )

            queue_empty = self.tts.speech_queue.empty()

            playing_audio = bool(getattr(self.tts, "is_playing_audio", lambda: False)())
            if pending_futures == 0 and pending_outputs == 0 and queue_empty and not playing_audio:
                break

            if time.time() >= deadline:
                print("[TTS] Timeout waiting for pending audio playback; continuing shutdown.")
                break

            time.sleep(0.05)