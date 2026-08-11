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
LLM           | HauhauCS Qwen3.6-35B-A3B Aggressive (Q5_K_P) | ~28 GB | 8080 | Conversation AI
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
cmake -B build -DGGML_CUDA=ON .
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
LLM_FILE='Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf'
curl -L --fail --retry 3 -C - -o "models/$LLM_FILE" \
  "https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/resolve/main/$LLM_FILE"
```

> **Why Qwen3.6-35B-A3B Aggressive Q5_K_P?**
> - HauhauCS describes this as its fully unlocked aggressive variant.
> - It is a 35B total / approximately 3B active MoE model, so generation remains practical.
> - The Q5_K_P file is about 28 GB and leaves substantial room in 128 GB unified memory.
> - For maximum stability in long coding/tool chains, use the 27B Balanced Q6_K_P instead.
> - For lower memory and faster startup, use the same repo's Q4_K_M file (~21 GB).

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

llama-server -m models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 131072 --jinja --mlock -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --chat-template-kwargs '{"enable_thinking":false}'
```

**Flag explanation:**

| Flag | Meaning |
|------|---------|
| `-m models/...` | Path to the GGUF model file |
| `--host 127.0.0.1` | Bind to localhost only (Open-LLM-VTuber is on the same machine) |
| `--port 8080` | HTTP API port |
| `-ngl 99` | Offload 99 layers to GPU (all of them) |
| `-c 131072` | 128K context, matching the model's documented context support |
| `--jinja` | Use llama.cpp's native chat-template handling |
| `-fa on` | Enable flash attention when supported by the build |
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
| **Qwen3.6-35B-A3B Aggressive Q5_K_P** (default) | `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` | ~28 GB | ★★★★★ | Best quality/latency choice for 128 GB unified memory |
| Qwen3.6-27B Balanced Q6_K_P | `HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced` | ~23 GB | ★★★★★ | Recommended for long agentic coding/tool chains |
| Qwen3.6-35B-A3B Aggressive Q4_K_M | `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` | ~21 GB | ★★★★☆ | Lower memory and faster startup |
| Gemma4-12B QAT Balanced + MTP | `HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced` | ~7.4 GB | ★★★★☆ | Fastest option; supports MTP speculative decoding |
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

### If you want to run a bigger LLM (e.g., Qwen3.6-35B-A3B)

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
1. Use a smaller LLM: HauhauCS Qwen3.6-35B-A3B Q4_K_M (~21 GB), or Gemma4-12B Q4_K_M (~7.4 GB)
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
| `models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf` | LLM model |
| `models/whisper/` | ASR model cache |
| `logs/debug_*.log` | Server logs |
| `chat_history/` | Saved conversations |

---

## License

This guide covers the setup of Open-LLM-VTuber (MIT license) and llama.cpp
(MIT license). Model files have their own licenses; review the model card and
license for the HauhauCS release before redistribution.