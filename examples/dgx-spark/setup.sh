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
# This script:
#   1. System dependencies   — build tools, CUDA headers
#   2. uv package manager    — installed if not present
#   3. Python deps           — uv sync + faster-whisper, melo-tts, pytest
#   4. llama.cpp from source — CUDA-enabled llama-server
#   5. Download LLM          — HauhauCS Qwen3.6-35B-A3B Q5_K_P (~28 GB)
#   6. Pre-cache ASR model   — faster-whisper large-v3-turbo (~3 GB)
#   7. Deploy config         — conf.yaml from conf.example.yaml
#   8. Hermes verify manifest — .hermes/environment.json
#   9. (optional) Qwen3-TTS   — if --with-qwen3-tts flag is passed
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "========================================"
echo " Open-LLM-VTuber — DGX Spark Setup"
echo "========================================"
echo ""

# ── Step 1: System dependencies ────────────────
echo "▸ [1/8] Installing system dependencies..."

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
echo "▸ [2/8] Ensuring uv package manager..."
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ── Step 3: Install Python dependencies ─────────
echo "▸ [3/8] Installing Python dependencies (uv sync)..."
cd "$DIR"
uv sync 2>&1 | tail -3

# Optional backends — the base project intentionally does not force a
# particular ASR or TTS backend; this config uses faster-whisper + MeloTTS.
echo "   Installing faster-whisper, melo-tts, and pytest..."
uv pip install faster-whisper melo-tts==0.1.2 pytest 2>&1 | tail -1 || true

# ── Step 4: Build / install llama.cpp ───────────
echo "▸ [4/8] Installing llama.cpp (llama-server)..."
if command -v llama-server &>/dev/null; then
  echo "   llama-server already installed, skipping build."
else
  if [ ! -d "build/llama.cpp" ]; then
    mkdir -p build
    git clone --depth 1 https://github.com/ggml-org/llama.cpp build/llama.cpp
  fi
  cd build/llama.cpp
  cmake -B build -DGGML_CUDA=ON .
  cmake --build build --config Release -j$(nproc) --target llama-server
  sudo cp build/bin/llama-server /usr/local/bin/
  cd "$DIR"
  echo "   llama-server installed to /usr/local/bin/"
fi

# ── Step 5: Download LLM model ──────────────────
echo "▸ [5/8] Downloading LLM model..."
mkdir -p models

# HauhauCS Qwen3.6-35B-A3B Aggressive Q5_K_P (~28 GB)
LLM_FILE="Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf"
LLM_REPO="HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
if [ ! -f "models/$LLM_FILE" ]; then
  echo "   Downloading $LLM_FILE (~28 GB)..."
  if command -v hf &>/dev/null; then
    hf download "$LLM_REPO" "$LLM_FILE" --local-dir models
  else
    curl -L --fail --retry 3 -C - -o "models/$LLM_FILE" \
      "https://huggingface.co/$LLM_REPO/resolve/main/$LLM_FILE"
  fi
else
  echo "   HauhauCS Qwen3.6 GGUF already exists, skipping."
fi

# ── Step 6: Pre-cache ASR model ─────────────────
echo "▸ [6/8] Pre-caching faster-whisper model..."
# faster-whisper auto-downloads on first use, but we pre-cache it now so the
# VTuber starts faster on first boot.
uv run python3 -c "
from faster_whisper import WhisperModel
import os
os.makedirs('models/whisper', exist_ok=True)
print('   Downloading faster-whisper large-v3-turbo (~3 GB, one-time)...')
model = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8',
                     download_root='models/whisper')
print('   faster-whisper model cached.')
" 2>&1 || echo "   (non-fatal) faster-whisper pre-cache failed; will download on first boot"

# ── Step 7: Copy config ─────────────────────────
echo "▸ [7/8] Deploying configuration..."
cp examples/dgx-spark/conf.example.yaml conf.yaml
echo "   conf.yaml installed."

# ── Step 8: Hermes verify manifest ──────────────
echo "▸ [8/8] Installing Hermes verify manifest..."
mkdir -p .hermes
cat > .hermes/environment.json <<'ENVEOF'
{
  "version": 1,
  "recipe": {
    "name": "Open-LLM-VTuber",
    "kind": "python",
    "bootstrap": ["uv sync"],
    "build": [],
    "test": ["pytest"],
    "start": "uv run run_server.py",
    "port": 12393,
    "readinessPath": "/",
    "evidence": ["Project entry point: run_server.py", "Configured server port: 12393"]
  }
}
ENVEOF
echo "   .hermes/environment.json installed."

# ── Step 9: Qwen3-TTS (optional) ──────────────
INSTALL_QWEN3_TTS=false
for arg in "$@"; do
  if [ "$arg" = "--with-qwen3-tts" ]; then
    INSTALL_QWEN3_TTS=true
  fi
done

if $INSTALL_QWEN3_TTS; then
  echo "▸ [9/9] Installing Qwen3-TTS (studio-quality local TTS)..."
  if command -v qwen3-tts &>/dev/null; then
    echo "   qwen3-tts already installed, skipping."
  else
    if command -v cargo &>/dev/null; then
      echo "   Building from source (requires Rust toolchain)..."
      git clone --depth 1 https://github.com/darkautism/qwen3-tts.git build/qwen3-tts 2>/dev/null || true
      cd build/qwen3-tts
      cargo build --release 2>&1 | tail -3
      sudo cp target/release/qwen3-tts /usr/local/bin/
      cd "$DIR"
      echo "   qwen3-tts installed to /usr/local/bin/"
      echo "   Run: qwen3-tts serve --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --quant q4"
    else
      echo "   Rust/cargo not found. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
      echo "   Then re-run: bash examples/dgx-spark/setup.sh --with-qwen3-tts"
    fi
  fi
else
  echo "▸ [9/9] Qwen3-TTS skipped (pass --with-qwen3-tts to install)"
fi

# ── Done ────────────────────────────────────────
echo ""
echo "========================================"
echo " ✅ Setup complete!"
echo "========================================"
echo ""
echo "To start the VTuber, run these in TWO terminals:"
echo ""
echo "  ── Terminal 1: llama.cpp server ──"
echo "    llama-server -m models/$LLM_FILE \\"
echo "      --host 127.0.0.1 --port 8080 \\"
echo "      -ngl 99 -c 131072 --jinja --mlock -fa on \\"
echo "      --cache-type-k q8_0 --cache-type-v q8_0 \\"
echo "      --chat-template-kwargs '{\"enable_thinking\":false}'"
echo ""
echo "  ── Terminal 2: Open-LLM-VTuber ──"
echo "    cd $DIR"
echo "    uv run run_server.py --verbose"
echo ""
echo "Then open http://localhost:12393 in your browser."
echo ""
echo "To verify the full stack:"
echo "  hermes verify --json --ready-timeout 120"
echo ""
