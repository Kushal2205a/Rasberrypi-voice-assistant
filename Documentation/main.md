# `pipeline/main.py`

## Purpose
Entry point for the Raspberry Pi voice assistant. It parses CLI options, sets up the GPIO push-button, builds the LLM settings, optionally warms models, constructs the assistant, and runs a session when the button is pressed.

## Imports
- `from pathlib import Path`  
  Path handling for CLI parameters that point to binaries or model files.
- `import os, argparse, RPi.GPIO as GPIO, time`  
  OS utilities, CLI parsing, Raspberry Pi GPIO access, and simple sleeps in the button loop. 
- `from config import CHUNK_DURATION, SAMPLE_RATE, WHISPER_EXE, WHISPER_MODEL, PIPER_MODEL_PATH, DEFAULT_SILENCE_THRESHOLD, DEFAULT_SILENCE_TIMEOUT`  
  Pulls runtime defaults such as audio rate, chunk size, silence settings, and default paths. 
- `from warmup import ModelPreloader`  
  Helper class that can preload models to reduce first-run latency. 
- `from orchestrator import ParallelVoiceAssistant`  
  The main coordinator that wires recorder, STT, LLM, and TTS together.

## GPIO setup
- `BUTTON_PIN = 17`  
- `GPIO.setmode(GPIO.BCM)`  
- `GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)`  
Uses BCM numbering and an internal pull-up so the input reads **LOW** when the button is pressed. 


## `_parse_args()`
Builds the command-line interface and returns an `argparse.Namespace`. Options:

Audio and timing
- `--duration` seconds to run the session before auto-stop. 
- `--silence-timeout` seconds of silence to stop automatically.  
- `--silence-threshold` RMS level that counts as silence. 

STT and legacy Whisper CLI
- `--whisper-cli` path to `whisper.cpp` binary  
- `--whisper-model` path to model file  
- `--whisper-threads` CPU threads for Whisper CLI  
- `--enable-stt-partials` turn on per-chunk interim transcripts 

LLM controls
- `--threads` worker threads for the local LLM  
- `--n-predict` tokens to generate  
- `--temperature` sampling temperature  
- `--llama-cli` binary override  
- `--llama-model` model path override 

Playback controls for the optional Piper path (safe to ignore when using eSpeak)
- `--output-device` sounddevice output selector  
- `--playback-cmd` external player command fallback  
- `--force-subprocess-playback` always use the external player  
- `--direct-playback` route audio directly through `sounddevice` 

A note on warmup
- `--warmup` is available, and the code also performs warm steps in practice, so the flag is usually unnecessary. 

## `main()`
High-level flow:

1) Parse arguments and assemble `llama_kwargs` for the LLM wrapper  
   Includes threads, `n_predict`, `temperature`, plus optional overrides for paths.
2) Optional warmup  
   Calls `ModelPreloader` to touch Whisper, Llama, and Piper models if `--warmup` is set. 
3) Decide playback route  
   Chooses direct vs subprocess playback based on the two flags.  
4) Build the assistant  
   Instantiates `ParallelVoiceAssistant` with audio settings, STT preferences, LLM kwargs, and playback choices. 
5) Run it  
   Uses `assistant.run(duration=...)`, where duration is `None` for unlimited sessions or the value from `--duration`. 

### Parameters passed to `ParallelVoiceAssistant`
- Audio: `chunk_duration`, `sample_rate`  
- STT: `stt_workers`, whisper CLI path/model/threads, `emit_stt_partials`  
- TTS: `piper_model_path` and playback routing flags  
- LLM: `llama_kwargs`  
- Silence handling: `silence_timeout`, `silence_threshold`  
These become the orchestrator’s configuration for wiring recorder → STT → LLM → TTS. 

## Button-driven session loop
Top-level code that waits for a press, runs a session, then waits for release:

- Prints a prompt to press the button  
- When input on GPIO 17 goes LOW, calls `main()` to run a session  
- After the session, waits until the button is released before listening again  
- Cleans up GPIO on `Ctrl+C` 

## Hand-off to the next file
When `main()` constructs `ParallelVoiceAssistant`, execution moves into `pipeline/orchestrator.py`, which uses these values to connect the recorder, STT engine, LLM, and TTS, handle partials, and manage timing. We will document `orchestrator.py` next using the exact parameters listed above. 

## Minimal usage example

    python -m pipeline.main \
      --duration 15 \
      --enable-stt-partials \
      --threads 4 \
      --n-predict 12 \
      --temperature 0.6

This keeps everything local with Faster-Whisper for STT, llama.cpp for the LLM, and eSpeak for TTS.

## History & Benchmarks

### Past models & performance  
- **whisper.cpp (tiny / base models)** — used originally for STT; results: ~10 s latency on Pi 4, memory use ~500 MB.  
  Repository: [whisper.cpp](https://github.com/ggerganov/whisper.cpp)  
- **Other GGUF models (e.g. 7B variants, etc.)** — tested but too heavy for local deployment, memory > 4 GB  

### Past STT / TTS tools & metrics  
- **Whisper CLI (whisper.cpp)** for STT — average latency ~10 seconds on my test phrases  
- **Piper TTS** — added ~2-3 s extra overhead (so I disabled it for now)  
  Repository: [Piper TTS](https://github.com/rhasspy/piper)  
- **eSpeak** — currently used; adds minimal latency (<0.2 s)  
  Project page: [eSpeak](http://espeak.sourceforge.net)  

### Hardware tested  
- **Raspberry Pi 4 (8 GB)** — main development target   
  Product info: [Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/)  


