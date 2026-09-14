#!/usr/bin/env bash
# =============================================================================
# Interview AI Studio - All-in-One Launcher for Linux / macOS / WSL (Bash)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo " Interview AI Studio - Batch Processor Launcher"
echo "================================================================"

# 1. Check for Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python was not found in PATH. Please install Python 3.10+." >&2
    exit 1
fi

# 2. Check for FFmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "[WARN] FFmpeg was not found in PATH. Video rendering may fail without FFmpeg." >&2
    echo "       Please install FFmpeg (e.g. sudo apt install ffmpeg or brew install ffmpeg)." >&2
fi

# 3. Detect uv or fallback to standard python -m venv/pip
USE_UV=false
if command -v uv &>/dev/null; then
    USE_UV=true
fi

VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

# 4. Create virtual environment if missing
if [ ! -f "$VENV_PYTHON" ]; then
    echo "[SETUP] Initializing virtual environment in .venv..."
    if [ "$USE_UV" = true ]; then
        echo "[SETUP] Using uv for accelerated environment creation..."
        uv venv "$VENV_DIR"
    else
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
fi

# 5. Install / verify dependencies
REQ_FILE="$SCRIPT_DIR/requirements.txt"
FLAG_FILE="$VENV_DIR/.installed_requirements"

NEEDS_INSTALL=true
if [ -f "$FLAG_FILE" ] && [ -f "$REQ_FILE" ]; then
    if command -v md5sum &>/dev/null; then
        CURRENT_HASH=$(md5sum "$REQ_FILE" | awk '{print $1}')
    elif command -v md5 &>/dev/null; then
        CURRENT_HASH=$(md5 -q "$REQ_FILE")
    else
        CURRENT_HASH=$(cksum "$REQ_FILE" | awk '{print $1}')
    fi

    if [ -f "$FLAG_FILE" ]; then
        SAVED_HASH=$(cat "$FLAG_FILE")
        if [ "$CURRENT_HASH" = "$SAVED_HASH" ]; then
            NEEDS_INSTALL=false
        fi
    fi
fi

if [ "$NEEDS_INSTALL" = true ]; then
    echo "[SETUP] Installing required dependencies from requirements.txt..."
    if [ "$USE_UV" = true ]; then
        uv pip install --python "$VENV_PYTHON" -r "$REQ_FILE"
    else
        "$VENV_PYTHON" -m pip install -q --upgrade pip
        "$VENV_PYTHON" -m pip install -q -r "$REQ_FILE"
    fi

    if command -v md5sum &>/dev/null; then
        CURRENT_HASH=$(md5sum "$REQ_FILE" | awk '{print $1}')
    elif command -v md5 &>/dev/null; then
        CURRENT_HASH=$(md5 -q "$REQ_FILE")
    else
        CURRENT_HASH=$(cksum "$REQ_FILE" | awk '{print $1}')
    fi
    echo "$CURRENT_HASH" > "$FLAG_FILE"
    echo "[SETUP] Dependencies installed successfully."
fi

# 6. Resolve input/output arguments
if [ "$#" -gt 0 ]; then
    TARGET_ARGS=("$@")
else
    INPUT_DIR="$SCRIPT_DIR/inputs"
    OUTPUT_DIR="$SCRIPT_DIR/outputs"
    mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

    echo "[INFO] No arguments provided. Defaulting to:"
    echo "       Input  : $INPUT_DIR"
    echo "       Output : $OUTPUT_DIR"
    echo "       (Place raw video files in ./inputs/ to process)"

    TARGET_ARGS=("--input" "$INPUT_DIR" "--output" "$OUTPUT_DIR")
fi

# 7. Execute processor
PROCESSOR_SCRIPT="$SCRIPT_DIR/interview_processor.py"
echo ""
echo "[RUN] Starting Interview AI Studio processor..."
"$VENV_PYTHON" "$PROCESSOR_SCRIPT" "${TARGET_ARGS[@]}"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "[DONE] Processing batch finished successfully."
else
    echo ""
    echo "[FAIL] Processing failed with exit code $EXIT_CODE." >&2
fi

exit $EXIT_CODE
