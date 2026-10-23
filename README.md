# NVIDIA PersonaPlex — Notes and Quickstart

Notes on installing and running [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex). The official repository is included in this repo as a git submodule at [`personaplex/`](personaplex).

PersonaPlex (`nvidia/personaplex-7b-v1`) is an open, real-time, **full-duplex speech-to-speech** model. It is built on Kyutai's [Moshi](https://arxiv.org/abs/2410.00037) architecture and weights. You control the persona with two inputs:

- a **voice prompt**, which sets how it sounds
- a **text role prompt**, which sets who it is and what it knows

## Contents

- [Getting this repository](#getting-this-repository)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the live server (web UI)](#running-the-live-server-web-ui)
- [Offline inference (WAV in, WAV out)](#offline-inference-wav-in-wav-out)
- [Choosing a persona](#choosing-a-persona)
- [Hardware and GPU compatibility](#hardware-and-gpu-compatibility)
- [License](#license)
- [Citation](#citation)
- [References and sources](#references-and-sources)

## Getting this repository

Clone with submodules:

```bash
git clone --recurse-submodules <this-repo-url>
```

If you already cloned without them:

```bash
git submodule update --init --recursive
```

Run all commands below from inside the submodule:

```bash
cd personaplex
```

## Prerequisites

1. **Opus audio codec development library**

   | OS | Command |
   |---|---|
   | Ubuntu / Debian | `sudo apt install libopus-dev` |
   | Fedora / RHEL | `sudo dnf install opus-devel` |
   | macOS (Homebrew) | `brew install opus` (not listed upstream) |

2. **Python** with `pip`.

3. **Hugging Face access.** Log in to Hugging Face and accept the PersonaPlex model license at [huggingface.co/nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1). Then export your token:

   ```bash
   export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
   ```

## Installation

```bash
pip install moshi/.
```

**Blackwell GPUs only:** install the PyTorch build for CUDA 13.0 ([NVIDIA/personaplex#2](https://github.com/NVIDIA/personaplex/issues/2)):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**Not enough GPU memory:** install `accelerate` and add `--cpu-offload` to any command:

```bash
pip install accelerate
```

The upstream repository also ships a `Dockerfile` and `docker-compose.yaml`.

## Running the live server (web UI)

Start the server with temporary SSL certificates:

```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
```

With CPU offload:

```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --cpu-offload
```

Open `https://localhost:8998` and talk to it through your microphone. On a remote machine, use the link the server prints, or forward the port over SSH:

```bash
ssh -L 8998:localhost:8998 <user>@<gpu-host>
```

## Offline inference (WAV in, WAV out)

The offline script streams an input WAV through the model and writes the model's reply to an output WAV of the same length, plus a JSON transcript.

**Assistant example** (voice `NATF2`):

```bash
python -m moshi.offline \
  --voice-prompt "NATF2.pt" \
  --input-wav "assets/test/input_assistant.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

**Customer-service example** (voice `NATM1`, role prompt read from a file):

```bash
python -m moshi.offline \
  --voice-prompt "NATM1.pt" \
  --text-prompt "$(cat assets/test/prompt_service.txt)" \
  --input-wav "assets/test/input_service.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

For offline evaluation without a GPU, install CPU-only PyTorch. Expect it to be slow.

### CLI flags

| Flag | Used with | Purpose |
|---|---|---|
| `--ssl <dir>` | server | Directory for SSL certificates |
| `--cpu-offload` | server, offline | Offload layers to CPU when GPU memory is insufficient (needs `accelerate`) |
| `--voice-prompt <file>` | offline | Voice embedding, e.g. `NATF2.pt` |
| `--text-prompt <text>` | offline | Role prompt text |
| `--input-wav <file>` | offline | Input audio |
| `--output-wav <file>` | offline | Output audio |
| `--output-text <file>` | offline | Output transcript (JSON) |
| `--seed <int>` | offline | Random seed for reproducibility |

## Choosing a persona

### Voices (`--voice-prompt`)

| Family | Female | Male | Character |
|---|---|---|---|
| Natural | `NATF0`–`NATF3` | `NATM0`–`NATM3` | Conversational, natural-sounding |
| Variety | `VARF0`–`VARF4` | `VARM0`–`VARM4` | More varied voices |

### Role prompts (`--text-prompt`)

- **Assistant:** a teacher-style instruction, e.g. "answer questions clearly and engagingly."
- **Customer service:** give the company, the agent's name, and the facts it should know: prices, schedules, policies, the customer's name.
- **Casual conversation:** start with `You enjoy having a good conversation.` and optionally add a topic or backstory, for example:

  ```
  You enjoy having a good conversation. Have a casual conversation about favorite foods and cooking experiences. You are David Green, a former baker now living in Boston.
  ```

- **Anything else:** PersonaPlex inherits broad generalization from the underlying Helium LLM, so custom roles outside its training distribution often work. Experiment.

## Hardware and GPU compatibility

**Short answer: not every GPU.**

### Officially documented

From the [model card](https://huggingface.co/nvidia/personaplex-7b-v1):

| Item | Value |
|---|---|
| Supported GPUs | NVIDIA Ampere (A100) and Hopper (H100) |
| Tested hardware | NVIDIA A100 80 GB |
| Operating system | Linux |
| Runtime | PyTorch |
| Weight precision | BF16 |
| VRAM requirement | Not published |

The upstream README also gives setup steps for **Blackwell** GPUs (the CUDA 13.0 PyTorch build).

### Inferred compatibility (not officially documented)

| GPU family | Likely outcome |
|---|---|
| **NVIDIA Ampere / Ada / Hopper / Blackwell** (A100, A6000, RTX 30/40/50, L40S, H100, B200) | Should work. These have native BF16. Ada isn't listed but should behave like Ampere. |
| **Older NVIDIA** (V100, T4, RTX 20, GTX 10) | Risky. No native BF16, so PyTorch emulates it slowly or the weights must be converted to FP16. Real-time speech probably won't keep up. |
| **AMD (ROCm)** | Unknown. The code is plain PyTorch, so ROCm PyTorch *might* work, but there are no reports either way. |
| **Apple Silicon** | The official code has no CUDA to use here, so it won't run on the GPU. A blog post describes a community Swift port; unverified. |

### VRAM estimate

NVIDIA publishes no number. A 7B model in BF16 needs about **14–15 GB for the weights alone**, plus the Mimi audio codec and the context cache on top. Plan on **16 GB as a bare minimum and 24 GB or more to be comfortable**.

`--cpu-offload` lets the model load on smaller cards, but it will likely be too slow for live conversation. It is fine for offline WAV processing.

### HPC note (PSC Bridges-2)

Bridges-2's long-standing GPU nodes are V100s, which have no native BF16. If newer GPUs (A100, H100, L40S) are available, use those. Run the server on the compute node and reach port 8998 through an SSH tunnel.

## License

- **Code** (upstream repo): MIT.
- **Model weights:** [NVIDIA Open Model License](https://huggingface.co/nvidia/personaplex-7b-v1). You must accept it on Hugging Face before downloading.

## Citation

```bibtex
@misc{roy2026personaplexvoicerolecontrol,
      title={PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models},
      author={Rajarshi Roy and Jonathan Raiman and Sang-gil Lee and Teodor-Dumitru Ene and Robert Kirby and Sungwon Kim and Jaehyeon Kim and Bryan Catanzaro},
      year={2026},
      eprint={2602.06053},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.06053},
}
```

## References and sources

### Official

- [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex): official code repository (submodule at [`personaplex/`](personaplex))
- [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1): model card and weights on Hugging Face
- [PersonaPlex paper (arXiv:2602.06053)](https://arxiv.org/abs/2602.06053)
- [PersonaPlex demo page, NVIDIA ADLR](https://research.nvidia.com/labs/adlr/personaplex/)
- [NVIDIA/personaplex#2](https://github.com/NVIDIA/personaplex/issues/2): Blackwell / CUDA 13.0 install note
- [PersonaPlex Discord](https://discord.gg/5jAXrrbwRb)

### Background

- [Moshi paper (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037): the base architecture
- [Helium LLM, Kyutai](https://kyutai.org/blog/2025-04-30-helium): the backbone language model
- [Opus audio codec](https://github.com/xiph/opus)
- [PyTorch CUDA 13.0 wheels](https://download.pytorch.org/whl/cu130)

### Third-party (unofficial)

- [Nvidia PersonaPlex 7B on Apple Silicon: Full-Duplex Speech-to-Speech in Swift (Mewayz blog)](https://mewayz.com/en/blog/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-swift)
- [Local AI Hardware Comparison: NVIDIA vs. AMD vs. Apple Silicon (Forgenex)](https://www.forgenex.com/en/blog/comparativa-de-hardware-para-ia-local-que-gpu-necesitas-nvidia-vs-amd-vs-apple-silicon)
