# Open-LLM-VTuber on NVIDIA DGX Spark — Setup Guide

This guide walks you through installing and running **Open-LLM-VTuber** on a
**NVIDIA DGX Spark** (GB10 Grace-Blackwell, 128 GB unified memory) — or any
Linux system with an NVIDIA GPU and at least 32 GB of RAM.

All models run **fully locally** on the same GPU simultaneously.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   DGX Spark (Linux)                      │
│  128 GB unified memory  •  Blackwell GPU (GB10)         │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  llama.cpp    │  │  faster-     │  │  MeloTTS     │  │
│  │  llama-server │  │  whisper     │  │  (CUDA)      │  │
│  │  (CUDA)       │  │  (CUDA)      │  │              │  │
│  │  ~9 GB VRAM   │  │  ~3 GB VRAM  │  │  ~2 GB VRAM  │  │
│  │  port 8080    │  │              │  │              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
│         └─────────────────┼─────────────────┘           │
│                           │                             │
│                    ┌──────┴──────┐                      │
│                    │  Open-LLM-  │                      │
│                    │  VTuber     │                      │
│                    │  FastAPI    │                      │
│                    │  port 12393 │                      │
│                    └─────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

**Component**  | **Model**  | **VRAM**  | **Port**  | **Role**
--------------|----------|----------|--------|--------
LLM           | Qwen2.5-14B-Instruct (Q4_K_M) | ~9 GB | 8080 | Conversation AI
ASR           | faster-whisper large-v3-turbo  | ~3 GB | — | Speech-to-text
TTS           | MeloTTS EN-Default             | ~2 GB | — | Text-to-speech
VAD           | Silero VAD                     | ~0.1 GB | — | Voice activity detection
**Total**     |                                | **~14 GB** | |

---

## Prerequisites

- **Hardware**: NVIDIA DGX Spark (or any NVIDIA GPU with 8+ GB VRAM, 32 GB+ RAM)
- **OS**: Ubuntu 22.04 / 24.04 (DGX Spark ships with Ubuntu) or any Linux
- **Internet**: ~15 GB total download for models
- **Disk space**: ~30 GB free (models: ~9 GB + ~3 GB + ~2 GB + cache)

---

## Step-by-Step Installation

### Step 1: Clone the Fork

```bash
git clone https://github.com/subjec2change/Open-LLM-VTuber.git
cd Open-LLM-VTuber
```

### Step 2: Run the Automated Setup

```bash
chmod +x examples/dgx-spark/setup.sh
bash examples/dgx-spark/setup.sh
```

This script does everything below automatically. Read on for the manual
equivalent if you want to understand each step.

---

### Manual Equivalent (Step-by-Step)

#### 2a. System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake curl git make \
  libopenblas-dev libomp-dev python3-dev
```

#### 2b. Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

#### 2c. Install Python Dependencies

```bash
cd Open-LLM-VTuber
uv sync
uv pip install melo-tts==0.1.2
```

> **About `uv sync`**: This installs all dependencies from `pyproject.toml`
> in a virtual environment. It's the project's standard package manager.

#### 2d. Build llama.cpp (with CUDA support)

```bash
mkdir -p build
git clone --depth 1 https://github.com/ggml-org/llama.cpp build/llama.cpp
cd build/llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CUDA=ON .
cmake --build build --config Release -j$(nproc) --target llama-server
sudo cp build/bin/llama-server /usr/local/bin/
cd ../..
```

> **Why build from source?** The DGX Spark has a Blackwell GPU. The pre-built
> binaries may not include CUDA support or may target older architectures.
> Building with `-DGGML_CUDA=ON` ensures full GPU acceleration.

#### 2e. Download the LLM Model

```bash
mkdir -p models
curl -L -o models/Qwen2.5-14B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf
```

> **Why Qwen2.5-14B Q4_K_M?**
> - 14B parameters at Q4 quantization = ~9 GB — fits comfortably in 128 GB
> - Q4_K_M is the sweet spot between quality and speed
> - Excellent English and Chinese, 128K context window
> - If you want more speed, use `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (~5 GB)
> - If you want more intelligence, use `Qwen2.5-32B-Instruct-Q4_K_M.gguf` (~20 GB)
> - The 14B version is the best balance for a real-time voice assistant

#### 2f. Deploy Configuration

```bash
cp examples/dgx-spark/conf.example.yaml conf.yaml
```

---

## Running the VTuber

You need **two terminal sessions** (or use tmux/screen).

### Terminal 1: Start the LLM Backend

```bash
cd Open-LLM-VTuber

llama-server -m models/Qwen2.5-14B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 8192 --mlock \
  --cache-type-k q8_0 --cache-type-v q8_0
```

**Flag explanation:**

| Flag | Meaning |
|------|---------|
| `-m models/...` | Path to the GGUF model file |
| `--host 127.0.0.1` | Bind to localhost only (Open-LLM-VTuber is on the same machine) |
| `--port 8080` | HTTP API port |
| `-ngl 99` | Offload 99 layers to GPU (all of them) |
| `-c 8192` | Context window of 8192 tokens |
| `--mlock` | Lock model in RAM, prevent swapping |
| `--cache-type-k q8_0` | KV cache for K in 8-bit (saves VRAM) |
| `--cache-type-v q8_0` | KV cache for V in 8-bit (saves VRAM) |

Wait for the line: **`llama server listening at http://127.0.0.1:8080`**

### Terminal 2: Start the VTuber

```bash
cd Open-LLM-VTuber
uv run run_server.py --verbose
```

Wait for: **`Starting server on localhost:12393`**

### Open in Browser

Navigate to **http://localhost:12393** (or the machine's IP if accessing remotely).

Click the microphone icon to start speaking. The LLM will respond through
the Live2D avatar with voice.

---

## Exposing to Other Machines on Your LAN

In `conf.yaml`, change:

```yaml
system_config:
  host: '0.0.0.0'   # instead of 'localhost'
```

Then access from another machine at **http://<DGX-IP>:12393**.

---

## Model Selection Guide

### LLM Options (pick one — adjust `conf.yaml`)

| Model | GGUF File | Size | Quality | Notes |
|-------|-----------|------|---------|-------|
| **Qwen2.5-14B Q4_K_M** (default) | `bartowski/Qwen2.5-14B-Instruct-GGUF` | ~9 GB | ★★★★☆ | Best balance for real-time voice |
| Qwen2.5-7B Q4_K_M | `bartowski/Qwen2.5-7B-Instruct-GGUF` | ~5 GB | ★★★☆☆ | Lower latency, lower quality |
| Qwen2.5-32B Q4_K_M | `bartowski/Qwen2.5-32B-Instruct-GGUF` | ~20 GB | ★★★★★ | Higher latency, best quality |
| Llama 3.1 8B Q4_K_M | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GGUF` | ~5 GB | ★★★★☆ | Good English-only option |
| DeepSeek R1 Distill Qwen 14B IQ4_XS | `bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF` | ~9 GB | ★★★★☆ | Good reasoning model |

### ASR Options (set `asr_config:asr_model`)

| Model | VRAM | Latency | Accuracy |
|-------|------|---------|----------|
| **faster-whisper large-v3-turbo** (default) | ~3 GB | ~100ms | ★★★★★ |
| faster-whisper medium | ~1.5 GB | ~50ms | ★★★★☆ |
| sherpa-onnx sense-voice | ~1 GB | ~50ms | ★★★★☆ |
| whisper.cpp small | ~1 GB | ~80ms | ★★★☆☆ |

### TTS Options (set `tts_config:tts_model`)

| Model | VRAM | Quality | Notes |
|-------|------|---------|-------|
| **MeloTTS** (default) | ~2 GB | ★★★★☆ | Fully local, GPU-accelerated |
| edge-tts | 0 GB | ★★★★★ | Zero GPU, needs internet, fastest |
| Piper TTS | ~0.5 GB | ★★★☆☆ | Local, CPU-only, supports many voices |
| Sherpa-ONNX TTS | ~1 GB | ★★★☆☆ | Local, GPU-accelerated |

---

## Tuning for Your Setup

### If you have less VRAM (e.g., 24 GB GPU)

Use `edge-tts` instead of `melo_tts` and `faster-whisper medium` instead of `large-v3-turbo`:

```yaml
asr_config:
  asr_model: 'faster_whisper'
  faster_whisper:
    model_path: 'medium'       # smaller
    compute_type: 'int8'       # less GPU memory
    device: 'cuda'

tts_config:
  tts_model: 'edge_tts'        # zero GPU
```

### If you want to run a bigger LLM (e.g., Qwen2.5-32B)

Use the smallest TTS and ASR to free up VRAM:

```yaml
asr_config:
  asr_model: 'sherpa_onnx_asr'    # uses CPU, not GPU
  sherpa_onnx_asr:
    model_type: 'sense_voice'
    provider: 'cpu'

tts_config:
  tts_model: 'edge_tts'           # zero GPU
```

---

## Troubleshooting

### "CUDA out of memory"

Reduce model sizes:
1. Use a smaller LLM: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (~5 GB)
2. Use `edge-tts` (zero GPU) instead of `melo_tts`
3. Use `faster-whisper medium` with `compute_type: 'int8'`
4. Reduce llama.cpp context: `-c 4096`

### "Connection refused" on 127.0.0.1:8080

Make sure `llama-server` is running in Terminal 1 before starting the VTuber.

### Audio doesn't work

- Check microphone permissions: `pactl list sources short`
- Set the correct input device in your browser
- Try `arecord -d 3 test.wav && aplay test.wav` to verify audio hardware

### Frontend shows "Not Found"

Initialize the frontend submodule:

```bash
git submodule update --init --recursive
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `examples/dgx-spark/conf.example.yaml` | DGX Spark configuration example |
| `examples/dgx-spark/setup.sh` | Automated setup script |
| `conf.yaml` | Active configuration (copied from the example) |
| `models/Qwen2.5-14B-Instruct-Q4_K_M.gguf` | LLM model |
| `models/whisper/` | ASR model cache |
| `logs/debug_*.log` | Server logs |
| `chat_history/` | Saved conversations |

---

## License

This guide covers the setup of Open-LLM-VTuber (MIT license) and llama.cpp
(MIT license). Model files have their own licenses (Qwen2.5: Apache 2.0,
Llama: custom, etc.).