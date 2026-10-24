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
- [Example: text prompts to audio (MacBook Pro M4)](#example-text-prompts-to-audio-macbook-pro-m4)
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

### Option A: conda (recommended)

[`environment.yml`](environment.yml) installs Python 3.12, the Opus library, PersonaPlex's `moshi` package from the submodule (editable), and the extras the code imports: `accelerate`, `pyloudnorm`, `tqdm`, and `gradio`. Run it from the root of this repository, not from inside `personaplex/`:

```bash
conda env create -f environment.yml
```

```bash
conda activate personaplex
```

### Option B: pip

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

For offline evaluation without a GPU, install CPU-only PyTorch and add `--device cpu`, since the default is `cuda`. Expect it to be slow.

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

### Example: text prompts to audio (MacBook Pro M4)

[`example1/`](example1) is for a **MacBook Pro M4**. You write your side of a conversation as text in [`prompts.txt`](example1/prompts.txt). Your lines are spoken with macOS `say`, run through PersonaPlex on the Mac's CPU, and the agent's reply is saved as audio with a transcript. Run from the root of this repository:

```bash
python example1/run_example.py
```

See [example1/README.md](example1/README.md) for the prompts format and options.

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
| **Apple Silicon** | CPU only with the official code; MPS is unsupported. See [Apple MacBook Pro (M4)](#apple-macbook-pro-m4). |

### VRAM estimate

NVIDIA publishes no number. A 7B model in BF16 needs about **14–15 GB for the weights alone**, plus the Mimi audio codec and the context cache on top. Plan on **16 GB as a bare minimum and 24 GB or more to be comfortable**.

`--cpu-offload` lets the model load on smaller cards, but it will likely be too slow for live conversation. It is fine for offline WAV processing.

### PSC Bridges-2 GPUs

Bridges-2's GPU nodes, from the [PSC user guide](https://www.psc.edu/resources/bridges-2/user-guide/) and [ACCESS CiDeR](https://cider.access-ci.org/public/resources/RDR_000646). The "PersonaPlex fit" column is my assessment, not PSC's.

| GPU type (`--gres`) | Nodes | GPUs / node | GPU memory | RAM / node | Native BF16 | SUs / GPU-hour | PersonaPlex fit |
|---|---|---|---|---|---|---|---|
| `h100-80` (H100 SXM5) | 10 | 8 | 80 GB | 2 TB | Yes (Hopper) | 2 | **Best.** Officially supported architecture; one GPU is plenty. |
| `l40s-48` (L40S) | 3 | 8 | 48 GB | 1 TB | Yes (Ada) | 1 | **Good.** Fits on one GPU; Ada isn't officially listed but should work. |
| `v100-32` (V100 SXM2) | 24 | 8 | 32 GB | 512 GB | No (Volta) | 1 | **Risky.** Fits in memory, but BF16 is emulated; likely too slow for live use. |
| `v100-32` (DGX-2) | 1 | 16 | 32 GB | 1.5 TB | No (Volta) | 1 | Same as above. |
| `v100-16` (V100) | 9 | 8 | 16 GB | 192 GB | No (Volta) | 1 | **Poor.** Too tight for the weights plus cache; offline with `--cpu-offload` at best. |

All node types are available in the `GPU` (whole nodes) and `GPU-shared` (up to 4 GPUs of one node) partitions, each with a 48-hour limit. One GPU is enough for PersonaPlex, so use `GPU-shared`. For example, for an interactive session on one L40S:

```bash
interact -p GPU-shared --gres=gpu:l40s-48:1 -t 02:00:00
```

Run the server on the compute node, then forward port 8998 from your laptop through a login node:

```bash
ssh -L 8998:<compute-node>:8998 <user>@bridges2.psc.edu
```

### NVIDIA GeForce RTX 2080 Ti

**Not practical for live conversation. Offline WAV processing only, and slowly.** The 2080 Ti is a Turing GPU with **11 GB** of memory and no native BF16. The 7B model's BF16 weights alone (~14–15 GB) don't fit, so you must use `--cpu-offload`. That moves part of the model to system RAM and streams it over PCIe on every step, far too slow for real-time speech. Upstream ships no smaller quantized model. If you have a 2080 Ti, use it for offline experiments, or run on a Bridges-2 `l40s-48` or `h100-80` instead.

### Apple MacBook Pro (M4)

**It runs on the CPU, slowly. The Apple GPU (MPS) is unsupported and untested.**

- The upstream code is CUDA-first. Both `moshi.server` and `moshi.offline` default to `--device cuda`, so on a Mac you must pass `--device cpu`. The server's automatic MPS detection is commented out in the code.
- PyTorch 2.4.1 from [`environment.yml`](environment.yml) does detect MPS on an M4 Max (checked on macOS 15.6.1), and the model loader has an MPS code path. You can try `--device mps`, but expect unsupported-op errors or poor performance. Nobody upstream has tested it.
- **Memory:** Apple Silicon shares one pool of RAM between the CPU and GPU. M4 MacBook Pros ship with 16–128 GB depending on the chip. A 16 GB model can't comfortably hold the ~15 GB of weights plus macOS. **24 GB is a practical minimum, and 36–48 GB+ (M4 Pro / M4 Max) is better.**
- **Bottom line:** fine for trying offline WAV inference on the CPU. For live, full-duplex conversation, use an NVIDIA GPU, such as a Bridges-2 node. A community Swift port for Apple Silicon has been [described](https://mewayz.com/en/blog/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-swift), but it's unverified and not from NVIDIA.

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

### PSC Bridges-2

- [Bridges-2 User Guide, PSC](https://www.psc.edu/resources/bridges-2/user-guide/): GPU node types, partitions, SU rates
- [PSC Bridges-2 GPU, ACCESS CiDeR](https://cider.access-ci.org/public/resources/RDR_000646): node specifications
- [PSC secures $4.9M NSF grant to enhance Bridges-2 with NVIDIA H100 GPUs (HPCwire)](https://www.hpcwire.com/off-the-wire/psc-secures-4-9m-nsf-grant-to-enhance-bridges-2-supercomputer-with-nvidia-h100-gpus)

### Background

- [Moshi paper (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037): the base architecture
- [Helium LLM, Kyutai](https://kyutai.org/blog/2025-04-30-helium): the backbone language model
- [Opus audio codec](https://github.com/xiph/opus)
- [PyTorch CUDA 13.0 wheels](https://download.pytorch.org/whl/cu130)

### Third-party (unofficial)

- [Nvidia PersonaPlex 7B on Apple Silicon: Full-Duplex Speech-to-Speech in Swift (Mewayz blog)](https://mewayz.com/en/blog/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-swift)
- [Local AI Hardware Comparison: NVIDIA vs. AMD vs. Apple Silicon (Forgenex)](https://www.forgenex.com/en/blog/comparativa-de-hardware-para-ia-local-que-gpu-necesitas-nvidia-vs-amd-vs-apple-silicon)
