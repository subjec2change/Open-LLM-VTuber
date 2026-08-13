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
#   3. CTranslate2 from source — CUDA-enabled (CUDA 12.x and 13.x)
#   4. Python deps           — uv sync + faster-whisper, MeloTTS (GitHub), pytest
#   5. llama.cpp from source — CUDA-enabled llama-server
#   6. Download LLM          — HauhauCS Qwen3.6-35B-A3B Q5_K_P (~28 GB)
#   7. Pre-cache ASR model   — faster-whisper large-v3-turbo on GPU (~3 GB)
#   8. Deploy config         — conf.yaml from conf.example.yaml
#   9. Hermes verify manifest — .hermes/environment.json
#   10. (optional) Qwen3-TTS  — if --with-qwen3-tts flag is passed
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "========================================"
echo " Open-LLM-VTuber — DGX Spark Setup"
echo "========================================"
echo ""

# ── Step 1: System dependencies ────────────────
echo "▸ [1/10] Installing system dependencies..."

if command -v apt-get &>/dev/null; then
  # Ubuntu / Debian (DGX Spark ships Ubuntu)
  sudo apt-get update -qq
  sudo apt-get install -y -qq build-essential cmake curl git make \
    libopenblas-dev libomp-dev python3-dev pkg-config 2>&1 | tail -1
elif command -v pacman &>/dev/null; then
  sudo pacman -Sy --noconfirm base-devel cmake curl git make pkg-config 2>&1 | tail -1
else
  echo "⚠  Unsupported package manager. Install build-essential, cmake, curl, git, pkg-config manually."
fi

# ── Step 2: Install uv if not present ──────────
echo "▸ [2/10] Ensuring uv package manager..."
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ── Step 3: Build CTranslate2 from source with CUDA ─
echo "▸ [3/10] Building CTranslate2 from source with CUDA support..."

if python3 -c "import ctranslate2; print(ctranslate2.cuda_is_available())" 2>/dev/null | grep -q "True"; then
  echo "   CTranslate2 with CUDA support already installed, skipping build."
else
  echo "   Cloning CTranslate2 repository..."
  if [ ! -d "build/CTranslate2" ]; then
    mkdir -p build
    git clone --depth 1 https://github.com/OpenNMT/CTranslate2.git build/CTranslate2
  fi
  
  cd build/CTranslate2
  
  # Initialize submodules (required for spdlog, etc.)
  echo "   Initializing git submodules..."
  git submodule update --init --recursive 2>&1 | tail -1
  
  # Detect CUDA architecture (90 for Blackwell, 80 for A100, 86 for RTX 30 series, etc.)
  echo "   Detecting CUDA architecture..."
  CUDA_ARCH="90"  # Default to Blackwell for DGX Spark
  if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "   Detected GPU: $GPU_NAME"
    if [[ "$GPU_NAME" == *"A100"* ]]; then
      CUDA_ARCH="80"
    elif [[ "$GPU_NAME" == *"RTX"* ]]; then
      CUDA_ARCH="86"
    elif [[ "$GPU_NAME" == *"H100"* ]]; then
      CUDA_ARCH="90"
    fi
  fi
  echo "   Using CUDA architecture: $CUDA_ARCH"
  
  # Build with CUDA support
  echo "   Building CTranslate2 with CUDA (this may take 10-15 minutes)..."
  mkdir -p build && cd build
  cmake -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DCUDA_ARCH_LIST="$CUDA_ARCH" \
    -DBUILD_SHARED_LIBS=ON \
    .. 2>&1 | grep -E "(Error|CMake|Building|100%)" || true
  
  # Check if cmake succeeded
  if [ ! -f "CMakeFiles/cmake.check_cache" ]; then
    echo "   ⚠ CMake configuration failed. Installing dependencies..."
    sudo apt-get install -y -qq libcudnn8-dev libcudnn8 libmkl-dev || true
    cmake -DCMAKE_BUILD_TYPE=Release \
      -DWITH_CUDA=ON \
      -DWITH_CUDNN=ON \
      -DCUDA_ARCH_LIST="$CUDA_ARCH" \
      -DBUILD_SHARED_LIBS=ON \
      .. 2>&1 | tail -10
  fi
  
  cmake --build . --config Release -j$(nproc) --target all 2>&1 | tail -5
  
  echo "   Installing CTranslate2 libraries..."
  sudo make install 2>&1 | tail -3
  sudo ldconfig  # Update library cache
  
  cd "$DIR"
  echo "   CTranslate2 built and installed successfully."
fi

# ── Step 4: Install Python dependencies ─────────
echo "▸ [4/10] Installing Python dependencies (uv sync)..."
cd "$DIR"
uv sync 2>&1 | tail -3

# Install faster-whisper (now that CTranslate2 with CUDA is ready)
echo "   Installing faster-whisper, MeloTTS (from GitHub), and pytest..."
# MeloTTS 0.1.2 is not available on PyPI, so install it directly from GitHub.
uv pip install faster-whisper git+https://github.com/myshell-ai/MeloTTS.git pytest 2>&1 | tail -1 || true

# Verify CTranslate2 CUDA support
echo "   Verifying CTranslate2 CUDA support..."
if python3 -c "import ctranslate2; assert ctranslate2.cuda_is_available(), 'CUDA not available'" 2>&1; then
  echo "   ✓ CTranslate2 CUDA support verified."
else
  echo "   ⚠ Warning: CTranslate2 CUDA support not detected. Falling back to CPU mode."
fi

# ── Step 5: Build / install llama.cpp ───────────
echo "▸ [5/10] Installing llama.cpp (llama-server)..."
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

# ── Step 6: Download LLM model ──────────────────
echo "▸ [6/10] Downloading LLM model..."
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

# ── Step 7: Pre-cache ASR model on GPU ──────────
echo "▸ [7/10] Pre-caching faster-whisper model on GPU..."
# faster-whisper auto-downloads on first use, but we pre-cache it now so the
# VTuber starts faster on first boot. Cache with device='cuda' so it runs on GPU.
uv run python3 -c "
from faster_whisper import WhisperModel
import os
os.makedirs('models/whisper', exist_ok=True)
print('   Downloading faster-whisper large-v3-turbo (~3 GB, one-time)...')
try:
  model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16',
                       download_root='models/whisper')
  print('   ✓ faster-whisper model cached with CUDA support.')
except Exception as e:
  print(f'   ⚠ CUDA caching failed, will use CPU fallback: {e}')
  model = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8',
                       download_root='models/whisper')
  print('   ✓ faster-whisper model cached with CPU fallback.')
" 2>&1 || echo "   (non-fatal) faster-whisper pre-cache failed; will download on first boot"

# ── Step 8: Copy config ─────────────────────────
echo "▸ [8/10] Deploying configuration..."
cp examples/dgx-spark/conf.example.yaml conf.yaml
echo "   conf.yaml installed."

# ── Step 9: Hermes verify manifest ──────────────
echo "▸ [9/10] Installing Hermes verify manifest..."
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

# ── Step 10: Qwen3-TTS (optional) ──────────────
echo "▸ [10/10] Qwen3-TTS setup..."
INSTALL_QWEN3_TTS=false
for arg in "$@"; do
  if [ "$arg" = "--with-qwen3-tts" ]; then
    INSTALL_QWEN3_TTS=true
  fi
done

if $INSTALL_QWEN3_TTS; then
  echo "   Installing Qwen3-TTS (studio-quality local TTS)..."
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
  echo "   Qwen3-TTS skipped (pass --with-qwen3-tts to install)"
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
