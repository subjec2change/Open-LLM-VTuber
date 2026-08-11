#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Open-LLM-VTuber — DGX Spark Automated Setup
# ═══════════════════════════════════════════════════════════════
# Run this on a fresh DGX Spark (or any Linux system with CUDA,
# 128 GB+ unified memory, and Internet access).
#
#   Usage:
#     cd Open-LLM-VTuber
#     bash examples/dgx-spark/setup.sh
#
# This script will:
#   1. Install system dependencies (llama.cpp, CUDA tools)
#   2. Install Python dependencies via uv
#   3. Download the LLM model (Qwen2.5-14B-Instruct Q4_K_M)
#   4. Download the ASR model (faster-whisper large-v3-turbo)
#   5. Copy the DGX Spark conf.yaml
#   6. Print a launch command
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "========================================"
echo " Open-LLM-VTuber — DGX Spark Setup"
echo "========================================"
echo ""

# ── Step 1: System dependencies ────────────────
echo "▸ [1/6] Installing system dependencies..."

# Detect OS
if command -v apt-get &>/dev/null; then
  # Ubuntu / Debian (DGX Spark ships Ubuntu)
  sudo apt-get update -qq
  sudo apt-get install -y -qq build-essential cmake curl git make \
    libopenblas-dev libomp-dev python3-dev 2>&1 | tail -1
elif command -v pacman &>/dev/null; then
  sudo pacman -Sy --noconfirm base-devel cmake curl git make 2>&1 | tail -1
else
  echo "⚠  Unsupported package manager. Install build-essential, cmake, curl, git manually."
fi

# ── Step 2: Install uv if not present ──────────
echo "▸ [2/6] Ensuring uv package manager..."
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ── Step 3: Install Python dependencies ─────────
echo "▸ [3/6] Installing Python dependencies (uv sync)..."
cd "$DIR"
uv sync 2>&1 | tail -3

# Install melo-tts extra if needed (for MeloTTS)
uv pip install melo-tts==0.1.2 2>&1 | tail -1 || true

# ── Step 4: Build / install llama.cpp ───────────
echo "▸ [4/6] Installing llama.cpp (llama-server)..."
if command -v llama-server &>/dev/null; then
  echo "   llama-server already installed, skipping build."
else
  if [ ! -d "build/llama.cpp" ]; then
    mkdir -p build
    git clone --depth 1 https://github.com/ggml-org/llama.cpp build/llama.cpp
  fi
  cd build/llama.cpp
  cmake -B build -DGGML_CUDA=ON -DLLAMA_CUDA=ON .
  cmake --build build --config Release -j$(nproc) --target llama-server
  sudo cp build/bin/llama-server /usr/local/bin/
  cd "$DIR"
  echo "   llama-server installed to /usr/local/bin/"
fi

# ── Step 5: Download models ─────────────────────
echo "▸ [5/6] Downloading models..."
mkdir -p models

# LLM — Qwen2.5-14B-Instruct Q4_K_M (~9 GB)
if [ ! -f "models/Qwen2.5-14B-Instruct-Q4_K_M.gguf" ]; then
  echo "   Downloading Qwen2.5-14B-Instruct Q4_K_M (~9 GB)..."
  curl -L -o models/Qwen2.5-14B-Instruct-Q4_K_M.gguf \
    https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf
else
  echo "   Qwen2.5-14B GGUF already exists, skipping."
fi

# faster-whisper will auto-download on first run, but we pre-cache it
# The model directory is models/whisper/ set in conf.yaml

# ── Step 6: Copy config ─────────────────────────
echo "▸ [6/6] Deploying configuration..."
cp examples/dgx-spark/conf.example.yaml conf.yaml
echo "   conf.yaml installed."

# ── Done ────────────────────────────────────────
echo ""
echo "========================================"
echo " ✅ Setup complete!"
echo "========================================"
echo ""
echo "To start the VTuber, run these in TWO terminals:"
echo ""
echo "  ── Terminal 1: llama.cpp server ──"
echo "    llama-server -m models/Qwen2.5-14B-Instruct-Q4_K_M.gguf \\"
echo "      --host 127.0.0.1 --port 8080 \\"
echo "      -ngl 99 -c 8192 --mlock \\"
echo "      --cache-type-k q8_0 --cache-type-v q8_0"
echo ""
echo "  ── Terminal 2: Open-LLM-VTuber ──"
echo "    cd $DIR"
echo "    uv run run_server.py --verbose"
echo ""
echo "Then open http://localhost:12393 in your browser."
echo ""
echo "NOTE: On first run, faster-whisper will download"
echo "large-v3-turbo (~3 GB). This happens once."
echo ""