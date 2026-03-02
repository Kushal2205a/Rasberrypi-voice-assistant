#!/bin/bash
# start_whisper_server.sh
# Starts the whisper.cpp HTTP server and keeps it running.
# Run this once in a separate terminal before starting main.py.
#
# Usage:
#   bash start_whisper_server.sh            # tiny.en (fast, default)
#   bash start_whisper_server.sh base.en    # base.en (more accurate, ~2x slower)

MODEL_NAME="${1:-tiny.en}"
WHISPER_DIR="$HOME/whisper.cpp"
MODEL_PATH="$WHISPER_DIR/models/ggml-${MODEL_NAME}.bin"
PORT=8080

# Find server binary
SERVER_BIN=""
for candidate in "$WHISPER_DIR/build/bin/whisper-server" "$WHISPER_DIR/build/bin/server"; do
    if [ -f "$candidate" ]; then
        SERVER_BIN="$candidate"
        break
    fi
done

if [ -z "$SERVER_BIN" ]; then
    echo "whisper-server binary not found. Building it now..."
    cd "$WHISPER_DIR"
    cmake -B build -DWHISPER_BUILD_SERVER=ON
    cmake --build build --config Release -j4
    SERVER_BIN="$WHISPER_DIR/build/bin/whisper-server"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH. Downloading..."
    cd "$WHISPER_DIR"
    bash models/download-ggml-model.sh "$MODEL_NAME"
fi

echo "Starting whisper server: $MODEL_NAME on port $PORT"
echo "Press Ctrl-C to stop."
echo ""

exec "$SERVER_BIN" \
    -m "$MODEL_PATH" \
    -t 3 \
    --port "$PORT" \
    --host 127.0.0.1 \
    --inference-path /inference