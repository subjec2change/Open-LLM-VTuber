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
│  │  ~28 GB       │  │  ~3 GB       │  │  ~2 GB       │  │
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

**Component**  | **Model**  | **VRAM**  | **Notes**
--------------|----------|----------|--------
LLM           | HauhauCS Qwen3.6-35B-A3B Aggressive (Q5_K_P) | ~28 GB | Runs as `llama-server` on port 8080
ASR           | faster-whisper large-v3-turbo  | ~3 GB | Downloads ~3 GB on first run
TTS           | MeloTTS EN-Default             | ~2 GB | Can swap to edge-tts for zero GPU
VAD           | Silero VAD                     | ~0.1 GB | CPU-bound, negligible
KV cache      | 128K context @ q8_0            | ~8 GB | Per-session cache
OS + overhead |                                | ~8 GB | Linux + desktop
**Total**     |                                | **~49 GB** | **Leaves ~79 GB free of 128 GB**

---

## Prerequisites

- **Hardware**: NVIDIA DGX Spark (or any NVIDIA GPU with 8+ GB VRAM, 32 GB+ RAM)
- **OS**: Ubuntu 22.04 / 24.04 (DGX Spark ships with Ubuntu) or any Linux
- **Internet**: ~35 GB total download for models (28 GB LLM + ~3 GB ASR + ~2 GB TTS + deps)
- **Disk space**: ~50 GB free (models: ~28 GB LLM + ~3 GB ASR + ~2 GB TTS + code + cache)
- **CUDA**: NVIDIA driver 530+ (comes pre-installed on DGX Spark)

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

# Install optional backends used by this config.
# The base project does not force a specific ASR/TTS backend.
uv pip install faster-whisper melo-tts==0.1.2

# Install pytest so `hermes verify` can run the project's tests.
uv pip install pytest
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
> - HauhauCS describes this as its fully unlocked aggressive variant (0/465
>   refusals on the benchmark). No safety filter remains — the model never
>   refuses a prompt.
> - It is a 35B total / approximately 3B active MoE model, so generation
>   speed remains practical for real-time voice conversation.
> - The Q5_K_P file is about 28 GB and leaves substantial room in 128 GB
>   unified memory for the ASR, TTS, and KV cache.
> - For maximum stability in long coding/tool chains, use the 27B Balanced
>   Q6_K_P instead (~23 GB). The Balanced variant keeps the reasoning step
>   inline and is recommended for agentic work.
> - For lower memory and faster startup, use the same repo's Q4_K_M file
>   (~21 GB).

> **What is K_P?** K_P ("Perfect") quants are HauhauCS custom quantizations
> that use model-specific analysis to selectively preserve quality where
> it matters most. A K_P quant effectively bumps quality up by 1–2 levels
> at only a ~5–15% larger file size. Fully compatible with standard llama.cpp
> — no special build needed.

#### 2f. Deploy Configuration

```bash
cp examples/dgx-spark/conf.example.yaml conf.yaml
```

#### 2g. Install the Hermes Verify Manifest (recommended)

```bash
mkdir -p .hermes
cat > .hermes/environment.json <<'EOF'
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
EOF
```

This tells `hermes verify` to use `run_server.py` on port 12393 (not the
auto-detected `uvicorn main:app` on port 8000).

> **Note:** `hermes verify` runs a readiness poll against the server. On the
> very first boot, the ASR backend will auto-download its model (~3 GB for
> faster-whisper large-v3-turbo), which can exceed the default 60-second
> readiness timeout. Run the server once manually first, or pass
> `--ready-timeout 300` to `hermes verify`.

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
| `--host 127.0.0.1` | Bind to localhost only (VTuber is on the same machine) |
| `--port 8080` | HTTP API port — matches the VTuber config |
| `-ngl 99` | Offload 99 layers to GPU (all of them — Blackwell handles this) |
| `-c 131072` | 128K context, matching the model's documented context support |
| `--jinja` | Use llama.cpp's native chat-template handling |
| `-fa on` | Enable flash attention when supported by the build |
| `--mlock` | Lock model in RAM, prevent swapping |
| `--cache-type-k q8_0` | KV cache for K in 8-bit (saves VRAM) |
| `--cache-type-v q8_0` | KV cache for V in 8-bit (saves VRAM) |
| `--chat-template-kwargs` | Disable thinking mode for faster responses in voice chat |

> **Note on quoting:** The `--chat-template-kwargs` value uses single quotes.
> If your shell strips them, use double quotes with escaped inner quotes:
> `--chat-template-kwargs "{\"enable_thinking\":false}"`

Wait for the line: **`llama server listening at http://127.0.0.1:8080`**

### Terminal 2: Start the VTuber

```bash
cd Open-LLM-VTuber
uv run run_server.py --verbose
```

Wait for: **`Starting server on localhost:12393`**

> **First-run note:** On the very first boot, faster-whisper will download
> `large-v3-turbo` (~3 GB) into `models/whisper/`. This takes a few minutes
> and happens once. MeloTTS will also download its voice model (~0.5 GB) on
> first use.

### Open in Browser

Navigate to **http://localhost:12393** (or the machine's IP if accessing
remotely).

Click the microphone icon to start speaking. The LLM will respond through
the Live2D avatar with voice.

---

### Verify with Hermes

After the first manual boot (to cache ASR/TTS downloads), you can verify
the full stack with:

```bash
hermes verify --json --ready-timeout 120
```

This runs `uv sync`, `pytest`, starts the server, polls for readiness, and
tears down. Expect `"ok": true` when everything is working.

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

All models below are from HauhauCS and are fully uncensored (0/465 refusals).

| Model | Quant | Repo | Size | Notes |
|-------|-------|------|------|-------|
| **Qwen3.6-35B-A3B Aggressive Q5_K_P** (default) | Q5_K_P | `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` | ~28 GB | Best quality/latency for 128 GB memory; MoE, ~3B active per token |
| Qwen3.6-27B Balanced Q6_K_P | Q6_K_P | `HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced` | ~23 GB | Dense 27B; recommended for long agentic coding/tool chains |
| Qwen3.6-35B-A3B Aggressive Q4_K_M | Q4_K_M | `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` | ~21 GB | Lower memory, faster startup; same model, lighter quant |
| Gemma4-12B QAT Balanced + MTP | Q4_K_M | `HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced` | ~7.4 GB | Fastest option; ~60% faster with MTP speculative decoding |
| Qwen3.6-35B-A3B Aggressive Q8_K_P | Q8_K_P | Same as default repo | ~44 GB | Maximum quality; needs ~60 GB total with cache |

All repos have additional quants (IQ2_M through Q8_K_P). See the Hugging Face
pages for the full list.

### ASR Options (set `asr_config:asr_model`)

| Model | VRAM | Latency | Accuracy | First Download |
|-------|------|---------|----------|---------------|
| **faster-whisper large-v3-turbo** (default) | ~3 GB | ~100ms | ★★★★★ | ~3 GB (auto) |
| faster-whisper medium | ~1.5 GB | ~50ms | ★★★★☆ | ~1.5 GB (auto) |
| sherpa-onnx sense-voice | ~1 GB (CPU) | ~50ms | ★★★★☆ | ~1 GB (auto) |
| whisper.cpp small | ~1 GB | ~80ms | ★★★☆☆ | ~0.5 GB (auto) |

> **First-boot note:** All ASR backends auto-download their model on first
> run. The download happens during server startup and can take several minutes.
> This is normal and happens only once.

### TTS Options (set `tts_config:tts_model`)

| Model | GPU | Quality | Notes |
|-------|-----|---------|-------|
| **MeloTTS** (default) | ~2 GB | ★★★★☆ | Fully local, GPU-accelerated; downloads ~0.5 GB on first use |
| edge-tts | 0 GB | ★★★★★ | Zero GPU, needs internet, fastest startup |
| Piper TTS | 0 GB (CPU) | ★★★☆☆ | Local, CPU-only, supports many voices |
| Sherpa-ONNX TTS | ~1 GB | ★★★☆☆ | Local, GPU-accelerated |

---

## Tuning for Your Setup

### If you have less VRAM (e.g., 24 GB GPU)

Use `edge-tts` instead of `melo_tts`, a smaller LLM, and `faster-whisper medium`:

```yaml
agent_config:
  llm_configs:
    openai_compatible_llm:
      model: 'Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced'  # ~7.4 GB

asr_config:
  asr_model: 'faster_whisper'
  faster_whisper:
    model_path: 'medium'       # smaller
    compute_type: 'int8'       # less GPU memory
    device: 'cuda'

tts_config:
  tts_model: 'edge_tts'        # zero GPU
```

Also reduce llama.cpp context: `-c 4096`.

### If you want maximum quality (ample VRAM)

```yaml
agent_config:
  llm_configs:
    openai_compatible_llm:
      model: 'HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive'
      temperature: 0.6
```

Download the Q8_K_P variant (~44 GB) and start llama-server with `-c 131072`.

---

## Troubleshooting

### "CUDA out of memory"

1. Use a smaller LLM: Gemma4-12B Q4_K_M (~7.4 GB) or Qwen3.6-35B-A3B Q4_K_M (~21 GB)
2. Use `edge-tts` (zero GPU) instead of `melo_tts`
3. Use `faster-whisper medium` with `compute_type: 'int8'`
4. Reduce llama.cpp context: `-c 4096`
5. Reduce `--cache-type-k` and `--cache-type-v` to `q4_0`

### "Connection refused" on 127.0.0.1:8080

Make sure `llama-server` is running in Terminal 1 before starting the VTuber.
Check with: `curl -s http://127.0.0.1:8080/v1/models | head -5`

### Server fails to start / times out on first boot

On the very first run, the ASR backend downloads its model automatically:

- `faster-whisper large-v3-turbo`: ~3 GB download into `models/whisper/`
- `sherpa-onnx sense-voice`: ~1 GB download into `models/sherpa-onnx-*/`

This download happens during server startup and can take several minutes
depending on your internet speed. The server will become ready once the
download completes. On subsequent boots the model is cached and startup
is much faster (~5–15 seconds).

If using `hermes verify`, pass a longer readiness timeout:

```bash
hermes verify --json --ready-timeout 300
```

### Audio doesn't work

- Check microphone permissions: `pactl list sources short`
- Set the correct input device in your browser
- Try `arecord -d 3 test.wav && aplay test.wav` to verify audio hardware

### Frontend shows "Not Found"

Initialize the frontend submodule:

```bash
git submodule update --init --recursive
```

### hermes verify detects wrong port/command

The auto-detector assumes `uvicorn main:app --port 8000` for FastAPI projects.
Install the correct manifest (see step 2g above) or run:

```bash
hermes verify --detect-only --json      # see what it detects
hermes verify --save                    # save + edit .hermes/environment.json
```

### llama-server: "unknown option --chat-template-kwargs"

Your llama.cpp build is too old. Rebuild from the latest source:

```bash
cd build/llama.cpp && git pull && cmake --build build -j$(nproc)
```

### llama-server: "failed to load model" / CUDA errors

- Verify CUDA is available: `nvidia-smi`
- Rebuild llama.cpp with: `cmake -B build -DGGML_CUDA=ON .`
- On DGX Spark Blackwell, ensure you're using CUDA 12.4+

---

## Files Reference

| File | Purpose |
|------|---------|
| `examples/dgx-spark/conf.example.yaml` | DGX Spark configuration example |
| `examples/dgx-spark/setup.sh` | Automated setup script |
| `conf.yaml` | Active configuration (copied from the example) |
| `.hermes/environment.json` | `hermes verify` manifest (recommended) |
| `models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf` | LLM model |
| `models/whisper/` | ASR model cache (faster-whisper) |
| `models/sherpa-onnx-*/` | ASR model cache (sherpa-onnx, if used) |
| `logs/debug_*.log` | Server logs |
| `chat_history/` | Saved conversations |

---

## Dependencies Recap

This setup adds the following beyond the base project:

| Package | Why | Size |
|---------|-----|------|
| `llama-server` (from source) | LLM inference, CUDA-enabled | ~50 MB binary |
| `faster-whisper` | Speech-to-text via CTranslate2/CUDA | ~20 MB pip |
| `melo-tts==0.1.2` | GPU-accelerated text-to-speech | ~30 MB pip |
| `pytest` | Test runner for `hermes verify` | ~2 MB pip |
| HauhauCS Qwen3.6 GGUF | Uncensored LLM | ~28 GB download |
| faster-whisper large-v3-turbo | ASR model (auto-download) | ~3 GB download |
| MeloTTS EN-Default | TTS voice (auto-download) | ~0.5 GB download |

---

## License

This guide covers the setup of Open-LLM-VTuber (MIT license) and llama.cpp
(MIT license). Model files have their own licenses; review the model card and
license for the HauhauCS release before redistribution.
