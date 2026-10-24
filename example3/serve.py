#!/usr/bin/env python
"""Start PersonaPlex's live web UI (microphone in, voice out) with a SenNet or HuBMAP persona.

This wraps upstream's `python -m moshi.server`, which serves a web page that
streams your microphone to the model and plays its voice back, full duplex.
The wrapper adds what this repository needs:

  - checks your Hugging Face access before loading the 17 GB model
  - picks the device (CUDA if available, else CPU) and warns when it can't
    keep up in real time
  - on CPU/MPS, makes CPU copies of the voice prompts (see the main README)
  - prints the persona to paste into the page, and copies it on macOS
  - prints the SSH tunnel command when running on a cluster node

Run it inside the `personaplex` conda environment (see ../environment.yml):

  python example3/serve.py                      # SenNet persona, auto device
  python example3/serve.py --persona hubmap
  python example3/serve.py --device cuda        # NVIDIA GPU, e.g. on Bridges-2
"""

import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
from pathlib import Path

MODEL_REPO = "nvidia/personaplex-7b-v1"  # gated: license must be accepted
HERE = Path(__file__).resolve().parent
# Persona file and the voice to pick, as the web UI's voice menu labels it.
PERSONAS = {"sennet": ("sennet.txt", "NATURAL_F2"), "hubmap": ("hubmap.txt", "NATURAL_M1")}
MAX_PROMPT_CHARS = 1000  # the web UI's text prompt limit

# Per 80 ms audio frame, measured on a MacBook Pro M4 Max (48 GB) with PyTorch 2.4.1.
# Live conversation needs to stay under 80 ms.
MAC_SPEED = {
    "cpu": "136 ms per 80 ms frame (1.7x slower than real time)",
    "mps": "735 ms per 80 ms frame (9x slower than real time; aten::index_copy "
           "falls back to the CPU)",
}


def check_model_access():
    """Fail early, with instructions, if the gated model can't be downloaded."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError

    try:
        hf_hub_download(MODEL_REPO, "config.json")  # small file
    except GatedRepoError:
        sys.exit(
            f"\nCan't download {MODEL_REPO}: Hugging Face access was denied.\n"
            f"  1. Sign in at https://huggingface.co and accept the license at\n"
            f"     https://huggingface.co/{MODEL_REPO}\n"
            f"  2. Create a read token at https://huggingface.co/settings/tokens\n"
            f"  3. Run `huggingface-cli login` (or `export HF_TOKEN=<token>`)"
        )


def cpu_voice_prompt_dir(out_dir: Path) -> Path:
    """Save CPU copies of all voice prompts and return their folder.

    The published voice files were saved from a CUDA GPU, and moshi loads them
    with torch.load(path) and no map_location, which fails without CUDA. The
    web UI lets you pick any voice, so every voice is copied.
    """
    import torch
    from huggingface_hub import hf_hub_download

    voices_tgz = Path(hf_hub_download(MODEL_REPO, "voices.tgz"))
    src_dir = voices_tgz.parent / "voices"
    if not src_dir.exists():
        with tarfile.open(voices_tgz) as tar:
            tar.extractall(voices_tgz.parent, filter="data")

    dst_dir = out_dir / "voices"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob("*.pt")):
        dst = dst_dir / src.name
        if not dst.exists():
            torch.save(torch.load(src, map_location="cpu", weights_only=True), dst)
    return dst_dir


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def show_persona(name: str) -> None:
    """Print the persona to paste into the web UI; copy it on macOS."""
    if name == "none":
        return
    filename, voice = PERSONAS[name]
    text = (HERE / "personas" / filename).read_text().strip()
    if len(text) > MAX_PROMPT_CHARS:
        print(f"Warning: persona is {len(text)} characters; the web UI accepts "
              f"{MAX_PROMPT_CHARS}.", file=sys.stderr)
    print(f"\nPersona ({name}, {len(text)} characters). Paste it into the Text Prompt box "
          f"and pick Voice {voice}:\n\n{text}\n")
    if shutil.which("pbcopy"):
        subprocess.run(["pbcopy"], input=text, text=True, check=False)
        print("(Copied to the clipboard.)\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--persona", default="sennet", choices=[*PERSONAS, "none"],
                        help="Persona to print and copy for the web UI (default: sennet)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"],
                        help="auto (default): cuda if available, else cpu")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload model layers to CPU on small GPUs (needs accelerate)")
    parser.add_argument("--out-dir", default=HERE / "output", type=Path,
                        help="Where to keep CPU voice copies (default: example3/output)")
    args = parser.parse_args()

    check_model_access()
    device = pick_device(args.device)
    cmd = [sys.executable, "-m", "moshi.server", "--device", device, "--port", str(args.port)]
    env = dict(os.environ)
    if device != "cuda":
        cmd += ["--voice-prompt-dir", str(cpu_voice_prompt_dir(args.out_dir))]
    if device == "mps":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # must be set before torch starts
    if args.cpu_offload:
        cmd.append("--cpu-offload")

    show_persona(args.persona)
    if device != "cuda":
        speed = MAC_SPEED.get(device, "")
        print(f"Warning: running on {device}. On a MacBook Pro M4 Max this measured {speed}. "
              "The page works, but the model falls behind and its replies lag and stutter. "
              "For real-time conversation, use an NVIDIA GPU (see example3/README.md).\n",
              file=sys.stderr)

    print(f"When the server says it's ready, open http://localhost:{args.port}")
    if platform.system() == "Linux":
        print(f"On a cluster node, tunnel from your laptop first:\n"
              f"  ssh -L {args.port}:{socket.gethostname()}:{args.port} "
              f"<user>@<login-host>\n")
    print("Use headphones, so the model doesn't hear itself. Press Ctrl+C to stop.\n")

    try:
        subprocess.run(cmd, env=env, check=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
