#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Open-LLM-VTuber — ASUS Ascent GX10 (DGX OS 7.5.0-2) Setup
# ═══════════════════════════════════════════════════════════════
# Run this on ASUS Ascent GX10 with NVIDIA DGX OS 7.5.0-2
#
# System Specs:
#   - Architecture: ARM64 (Grace Blackwell GB10)
#   - CUDA: 13.0.2 (pre-installed)
#   - Driver: 580.x (pre-installed)
#   - Memory: 128 GB unified memory
#   - OS: NVIDIA DGX OS 7.5.0-2 (Ubuntu-based, ARM64)
#
#   Usage:
#     cd Open-LLM-VTuber
#     bash examples/dgx-spark/setup_gx10.sh
#
# This script:
#   1. Verify CUDA 13.0.2 environment
#   2. Install system dependencies (ARM64)
#   3. uv package manager — installed if not present
#   4. CTranslate2 from source — ARM64 + CUDA 13 support
#   5. Python deps — uv sync + faster-whisper, MeloTTS, pytest
#   6. llama.cpp from source — CUDA-enabled (ARM64)
#   7. Download LLM — HauhauCS Qwen3.6-35B-A3B Q5_K_P (~28 GB)
#   8. Pre-cache ASR model — faster-whisper on GPU
#   9. Deploy config — conf.yaml from conf.example.yaml
#   10. Hermes verify manifest — .hermes/environment.json
#   11. (optional) Qwen3-TTS
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "========================================"
echo " Open-LLM-VTuber — ASUS Ascent GX10"
echo " NVIDIA DGX OS 7.5.0-2 (ARM64)"
echo "========================================"
echo ""

# ── Verify ARM64 and CUDA ────────────────────────
echo "▸ [0/11] Verifying system environment..."

if ! uname -m | grep -q aarch64; then
  echo "❌ ERROR: This system is not ARM64. This script is for ASUS Ascent GX10 only."
  echo "   System architecture: $(uname -m)"
  exit 1
fi
echo "   ✓ ARM64 architecture confirmed."

if ! command -v nvidia-smi &>/dev/null; then
  echo "❌ ERROR: NVIDIA drivers not found. Ensure NVIDIA DGX OS is properly installed."
  exit 1
fi

CUDA_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "   ✓ NVIDIA Driver: $DRIVER_VERSION"
echo "   ✓ GPU Compute Capability: $CUDA_VERSION"

if ! command -v nvcc &>/dev/null; then
  echo "⚠  CUDA toolkit not in PATH. Attempting to source it..."
  if [ -f /opt/nvidia/hpc_sdk/Linux_aarch64/22.11/cuda/bin/nvcc ]; then
    export PATH="/opt/nvidia/hpc_sdk/Linux_aarch64/22.11/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_aarch64/22.11/cuda/lib64:$LD_LIBRARY_PATH"
  elif [ -f /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
  fi
fi

NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "Cuda compilation tools" | awk '{print $NF}' || echo "unknown")
echo "   ✓ CUDA Toolkit: $NVCC_VERSION"

# ── Step 1: System dependencies (ARM64) ──────────
echo ""
echo "▸ [1/11] Installing system dependencies (ARM64)..."

sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake curl git make \
  libopenblas-dev libomp-dev python3-dev pkg-config \
  libcudnn8-dev libcudnn8 2>&1 | tail -1

echo "   ✓ System dependencies installed."

# ── Step 2: Install uv if not present ───────────
echo "▸ [2/11] Ensuring uv package manager..."
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo "   ✓ uv installed."
else
  echo "   ✓ uv already installed."
fi

# ── Step 3: Build CTranslate2 from source (ARM64 + CUDA 13) ─
echo "▸ [3/11] Building CTranslate2 from source (ARM64 + CUDA 13)..."

if python3 -c "import ctranslate2; print(ctranslate2.cuda_is_available())" 2>/dev/null | grep -q "True"; then
  echo "   ✓ CTranslate2 with CUDA support already installed."
  ASR_DEVICE="cuda"
  ASR_COMPUTE="float16"
else
  echo "   Building CTranslate2 with CUDA 13 support..."
  if [ ! -d "build/CTranslate2" ]; then
    mkdir -p build
    git clone --depth 1 https://github.com/OpenNMT/CTranslate2.git build/CTranslate2
  fi
  
  cd build/CTranslate2
  git submodule update --init --recursive 2>&1 | tail -1
  
  # For Grace Blackwell (GB10), CUDA arch is 90
  CUDA_ARCH="90"
  echo "   Using CUDA architecture: $CUDA_ARCH (Grace Blackwell)"
  
  # Clean previous build
  rm -rf build
  mkdir -p build
  cd build
  
  # Configure with CUDA 13 for ARM64
  echo "   Configuring CMake for ARM64 + CUDA 13..."
  if ! cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DCUDA_ARCH_LIST="$CUDA_ARCH" \
    -DBUILD_SHARED_LIBS=ON \
    ..; then
    echo "   ❌ CMake configuration failed."
    echo "   Attempting fallback: skipping CTranslate2 build, will use CPU mode."
    cd "$DIR"
    ASR_DEVICE="cpu"
    ASR_COMPUTE="int8"
  else
    # Build
    echo "   Building CTranslate2 (this may take 15-20 minutes)..."
    if ! cmake --build . --config Release -j$(nproc) --target all 2>&1 | tail -10; then
      echo "   ⚠ CTranslate2 build failed. Continuing with CPU mode for ASR."
      cd "$DIR"
      ASR_DEVICE="cpu"
      ASR_COMPUTE="int8"
    else
      # Install
      echo "   Installing CTranslate2 libraries..."
      sudo make install 2>&1 | tail -3
      sudo ldconfig
      cd "$DIR"
      echo "   ✓ CTranslate2 built and installed successfully."
      ASR_DEVICE="cuda"
      ASR_COMPUTE="float16"
    fi
  fi
fi

# ── Step 4: Install Python dependencies ────────
echo "▸ [4/11] Installing Python dependencies (uv sync)..."
cd "$DIR"
uv sync 2>&1 | tail -3

# Install faster-whisper and other backends
echo "   Installing faster-whisper, MeloTTS, and pytest..."
uv pip install faster-whisper git+https://github.com/myshell-ai/MeloTTS.git pytest 2>&1 | tail -1 || true

# Verify CTranslate2 CUDA support
echo "   Verifying CTranslate2..."
if python3 -c "import ctranslate2; assert ctranslate2.cuda_is_available()" 2>&1 >/dev/null; then
  echo "   ✓ CTranslate2 CUDA support confirmed."
  ASR_DEVICE="cuda"
  ASR_COMPUTE="float16"
else
  echo "   ⚠ CTranslate2 CUDA not available. Using CPU mode."
  ASR_DEVICE="cpu"
  ASR_COMPUTE="int8"
fi

# ── Step 5: Build llama.cpp (CUDA + ARM64) ──────
echo "▸ [5/11] Installing llama.cpp (llama-server)..."
if command -v llama-server &>/dev/null; then
  echo "   ✓ llama-server already installed."
else
  if [ ! -d "build/llama.cpp" ]; then
    mkdir -p build
    git clone --depth 1 https://github.com/ggml-org/llama.cpp build/llama.cpp
  fi
  cd build/llama.cpp
  echo "   Building llama.cpp with CUDA support..."
  cmake -B build -DGGML_CUDA=ON . 2>&1 | tail -5
  cmake --build build --config Release -j$(nproc) --target llama-server 2>&1 | tail -5
  sudo cp build/bin/llama-server /usr/local/bin/
  sudo chmod +x /usr/local/bin/llama-server
  cd "$DIR"
  echo "   ✓ llama-server installed to /usr/local/bin/"
fi

# ── Step 6: Download LLM model ─────────────────
echo "▸ [6/11] Downloading LLM model..."
mkdir -p models

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
  echo "   ✓ LLM model downloaded."
else
  echo "   ✓ LLM model already exists."
fi

# ── Step 7: Pre-cache ASR model ────────────────
echo "▸ [7/11] Pre-caching faster-whisper model (device: $ASR_DEVICE)..."
mkdir -p models/whisper

uv run python3 -c "
from faster_whisper import WhisperModel
import os
os.makedirs('models/whisper', exist_ok=True)
print('   Downloading faster-whisper large-v3-turbo (~3 GB)...')
try:
  model = WhisperModel('large-v3-turbo', device='$ASR_DEVICE', compute_type='$ASR_COMPUTE',
                       download_root='models/whisper')
  print('   ✓ faster-whisper cached with device=$ASR_DEVICE')
except Exception as e:
  print(f'   ⚠ Failed with $ASR_DEVICE: {e}')
  print('   Falling back to CPU...')
  model = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8',
                       download_root='models/whisper')
  print('   ✓ faster-whisper cached with CPU')
" 2>&1 || echo "   (non-fatal) Will download on first boot"

# ── Step 8: Deploy configuration ───────────────
echo "▸ [8/11] Deploying configuration..."
cp examples/dgx-spark/conf.example.yaml conf.yaml
echo "   ✓ conf.yaml deployed."

# ── Step 9: Hermes verify manifest ─────────────
echo "▸ [9/11] Installing Hermes verify manifest..."
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
echo "   ✓ Hermes manifest installed."

# ── Step 10: Update PATH for CUDA (if needed) ──
echo "▸ [10/11] Configuring environment..."
if ! echo $PATH | grep -q "/opt/nvidia"; then
  echo "   Adding CUDA to PATH..."
  cat >> ~/.bashrc <<'EOF'

# CUDA 13.0.2 (DGX OS 7.5.0-2)
export PATH="/opt/nvidia/hpc_sdk/Linux_aarch64/22.11/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_aarch64/22.11/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/22.11/cuda"
EOF
  source ~/.bashrc
fi
echo "   ✓ Environment configured."

# ── Step 11: Qwen3-TTS (optional) ──────────────
echo "▸ [11/11] Qwen3-TTS setup..."
INSTALL_QWEN3_TTS=false
for arg in "$@"; do
  if [ "$arg" = "--with-qwen3-tts" ]; then
    INSTALL_QWEN3_TTS=true
  fi
done

if $INSTALL_QWEN3_TTS; then
  echo "   Installing Qwen3-TTS..."
  if command -v qwen3-tts &>/dev/null; then
    echo "   ✓ qwen3-tts already installed."
  else
    if command -v cargo &>/dev/null; then
      git clone --depth 1 https://github.com/darkautism/qwen3-tts.git build/qwen3-tts 2>/dev/null || true
      cd build/qwen3-tts
      cargo build --release 2>&1 | tail -3
      sudo cp target/release/qwen3-tts /usr/local/bin/
      cd "$DIR"
      echo "   ✓ qwen3-tts installed."
    else
      echo "   ⚠ Rust/cargo not found. Skipping qwen3-tts."
    fi
  fi
else
  echo "   Qwen3-TTS skipped."
fi

# ── Done ────────────────────────────────────────
echo ""
echo "========================================"
echo " ✅ Setup complete!"
echo "========================================"
echo ""
echo "ASR Device: $ASR_DEVICE"
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
