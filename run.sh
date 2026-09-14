#!/usr/bin/env bash
# ==============================================================================
# Interview AI Studio - All-in-One Launcher for Linux / macOS / WSL (Bash)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN} Interview AI Studio - Linux / macOS Batch Processor Launcher${NC}"
echo -e "${CYAN}================================================================${NC}"

# 1. Check for Python
PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo -e "${RED}[ERROR] Python 3 was not found. Please install Python 3.10+.${NC}"
    exit 1
fi

# 2. Check for FFmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo -e "${YELLOW}[WARN] FFmpeg was not found in PATH. Video rendering may fail without FFmpeg.${NC}"
    echo -e "${YELLOW}       Please install ffmpeg (e.g. sudo apt install ffmpeg or brew install ffmpeg).${NC}"
fi

# 3. Setup .bin directory and precompiled DeepFilterNet binary
BIN_DIR="$SCRIPT_DIR/.bin"
DEEPFILTER_BIN="$BIN_DIR/deep-filter"
if ! command -v deep-filter >/dev/null 2>&1 && [ ! -f "$DEEPFILTER_BIN" ]; then
    echo -e "${YELLOW}[SETUP] Downloading precompiled DeepFilterNet binary...${NC}"
    mkdir -p "$BIN_DIR"
    ARCH=$(uname -m)
    OS=$(uname -s)
    DF_URL=""
    if [ "$OS" = "Linux" ]; then
        if [ "$ARCH" = "x86_64" ]; then
            DF_URL="https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-x86_64-unknown-linux-musl"
        elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
            DF_URL="https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-aarch64-unknown-linux-gnu"
        fi
    elif [ "$OS" = "Darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then
            DF_URL="https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-aarch64-apple-darwin"
        else
            DF_URL="https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-x86_64-apple-darwin"
        fi
    fi

    if [ -n "$DF_URL" ]; then
        if curl -fsSL "$DF_URL" -o "$DEEPFILTER_BIN"; then
            chmod +x "$DEEPFILTER_BIN"
            echo -e "${GREEN}[SETUP] DeepFilterNet binary downloaded successfully.${NC}"
        else
            echo -e "${YELLOW}[WARN] Could not download DeepFilterNet precompiled binary.${NC}"
        fi
    fi
fi
export PATH="$BIN_DIR:$PATH"

# 4. Detect uv or fallback to standard venv/pip
USE_UV=false
if command -v uv >/dev/null 2>&1; then
    USE_UV=true
fi

VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

# 5. Create virtual environment if missing
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${YELLOW}[SETUP] Initializing virtual environment in .venv...${NC}"
    if [ "$USE_UV" = true ]; then
        echo -e "${GRAY}[SETUP] Using uv for accelerated environment creation...${NC}"
        uv venv "$VENV_DIR"
    else
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
fi

# 6. Install / verify dependencies
REQ_FILE="$SCRIPT_DIR/requirements.txt"
FLAG_FILE="$VENV_DIR/.installed_requirements"
NEEDS_INSTALL=true

if [ -f "$FLAG_FILE" ] && [ -f "$REQ_FILE" ]; then
    REQ_MD5=$(md5sum "$REQ_FILE" 2>/dev/null | awk '{print $1}' || md5 -q "$REQ_FILE" 2>/dev/null || true)
    SAVED_MD5=$(cat "$FLAG_FILE" 2>/dev/null || true)
    if [ -n "$REQ_MD5" ] && [ "$REQ_MD5" = "$SAVED_MD5" ]; then
        NEEDS_INSTALL=false
    fi
fi

if [ "$NEEDS_INSTALL" = true ]; then
    echo -e "${YELLOW}[SETUP] Installing required dependencies from requirements.txt...${NC}"
    if [ "$USE_UV" = true ]; then
        uv pip install --python "$VENV_PYTHON" -r "$REQ_FILE"
    else
        "$VENV_PYTHON" -m pip install --upgrade pip -q
        "$VENV_PYTHON" -m pip install -r "$REQ_FILE" -q
    fi
    REQ_MD5=$(md5sum "$REQ_FILE" 2>/dev/null | awk '{print $1}' || md5 -q "$REQ_FILE" 2>/dev/null || true)
    echo "$REQ_MD5" > "$FLAG_FILE"
    echo -e "${GREEN}[SETUP] Dependencies installed successfully.${NC}"
fi

# 7. Resolve input/output arguments
TARGET_ARGS=()
if [ $# -gt 0 ]; then
    TARGET_ARGS=("$@")
else
    INPUT_DIR="$SCRIPT_DIR/inputs"
    OUTPUT_DIR="$SCRIPT_DIR/outputs"
    mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

    echo -e "${CYAN}[INFO] No arguments provided. Defaulting to:${NC}"
    echo -e "${CYAN}       Input  : $INPUT_DIR${NC}"
    echo -e "${CYAN}       Output : $OUTPUT_DIR${NC}"
    echo -e "${GRAY}       (Place raw video files in ./inputs/ to process)${NC}"

    TARGET_ARGS=("--input" "$INPUT_DIR" "--output" "$OUTPUT_DIR")
fi

# 8. Execute processor
PROCESSOR_SCRIPT="$SCRIPT_DIR/interview_processor.py"
echo -e "\n${CYAN}[RUN] Starting Interview AI Studio processor...${NC}"
"$VENV_PYTHON" "$PROCESSOR_SCRIPT" "${TARGET_ARGS[@]}"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}[DONE] Processing batch finished successfully.${NC}"
else
    echo -e "\n${RED}[FAIL] Processing failed with exit code $EXIT_CODE.${NC}"
fi

exit $EXIT_CODE
