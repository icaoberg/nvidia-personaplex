#!/usr/bin/env python
"""Talk to PersonaPlex live in a web page on a MacBook Pro M4 (microphone in, voice out).

NVIDIA's PyTorch code is too slow for live conversation on Apple Silicon, so
this runs the community MLX port of PersonaPlex (the personaplex-mlx
submodule, mu-hashmi/personaplex-mlx) with 4-bit weights quantized from
NVIDIA's official model. Its web mode serves NVIDIA's own PersonaPlex page.
This wrapper adds:

  - checks that this is a Mac with enough memory and that the MLX port is installed
  - checks your Hugging Face access before loading the model
  - prints the persona to paste into the page, and copies it: a chef and
    cooking teacher by default, or a SenNet or HuBMAP data guide

Run it inside the `personaplex-mlx` conda environment (see ../environment-mlx.yml):

  python example3/serve.py                      # chef persona, 4-bit
  python example3/serve.py --persona sennet     # or hubmap
  python example3/serve.py --quant 8            # higher quality, slower
"""

import argparse
import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_REPO = "nvidia/personaplex-7b-v1"  # gated: license must be accepted
HERE = Path(__file__).resolve().parent
MIN_MEMORY_GB = 16  # 4-bit weights are ~4 GB, plus the audio codec and macOS
MAX_PROMPT_CHARS = 1000  # the web page's text prompt limit
# Persona file, and the voice to pick: (MLX port's ID, web page's menu label).
PERSONAS = {
    "chef": ("chef.txt", "NATM0", "NATURAL_M0"),
    "sennet": ("sennet.txt", "NATF2", "NATURAL_F2"),
    "hubmap": ("hubmap.txt", "NATM1", "NATURAL_M1"),
}


def sysctl(name: str) -> str:
    """Read a macOS system setting with `sysctl -n`.

    Args:
        name: The sysctl key, e.g. "machdep.cpu.brand_string" or "hw.memsize".

    Returns:
        The value as a string, or "" if the key doesn't exist.
    """
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def check_mac() -> None:
    """Require macOS on Apple Silicon and the MLX port; warn if memory is low.

    Prints the chip, memory, and macOS version. Exits if this isn't macOS on
    arm64, or if `personaplex_mlx` can't be imported (with the setup steps);
    only warns (on stderr) for less than MIN_MEMORY_GB of memory.
    """
    if sys.platform != "darwin" or platform.machine() != "arm64":
        sys.exit("This example needs a Mac with Apple Silicon (MLX doesn't run elsewhere).")
    if importlib.util.find_spec("personaplex_mlx") is None:
        sys.exit("personaplex_mlx isn't installed in this Python. From the repository root:\n"
                 "  git submodule update --init --recursive\n"
                 "  conda env create -f environment-mlx.yml\n"
                 "  conda activate personaplex-mlx")
    chip = sysctl("machdep.cpu.brand_string")
    memory_gb = int(sysctl("hw.memsize") or 0) / 2**30
    print(f"Mac:   {chip}, {memory_gb:.0f} GB unified memory, macOS {platform.mac_ver()[0]}")
    if memory_gb < MIN_MEMORY_GB:
        print(f"Warning: {memory_gb:.0f} GB of memory is below the ~{MIN_MEMORY_GB} GB "
              "this needs; expect swapping and choppy audio.", file=sys.stderr)


def check_model_access() -> None:
    """Fail early, with instructions, if the gated model can't be downloaded.

    Downloads the model's small config.json. If Hugging Face denies access
    (no login, or the license isn't accepted), exits with the steps to fix
    it before the MLX server starts loading the model.
    """
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


def load_persona(name: str) -> tuple:
    """Return (text, voice ID, voice menu label), and print and copy the text.

    The text is printed so it can be pasted into the page's Text Prompt box,
    and copied to the clipboard with `pbcopy` when available. Warns if it's
    longer than the page's MAX_PROMPT_CHARS limit.

    Args:
        name: A key of PERSONAS, e.g. "chef".

    Returns:
        (text, voice, label): the persona text from personas/, the voice ID
        for the MLX server (e.g. "NATM0"), and the matching label in the
        page's voice menu (e.g. "NATURAL_M0").
    """
    filename, voice, label = PERSONAS[name]
    text = (HERE / "personas" / filename).read_text().strip()
    if len(text) > MAX_PROMPT_CHARS:
        print(f"Warning: persona is {len(text)} characters; the page accepts "
              f"{MAX_PROMPT_CHARS}.", file=sys.stderr)
    print(f"\nPersona ({name}, {len(text)} characters). In the page, paste it into "
          f"Text Prompt and pick Voice {label}:\n\n{text}\n")
    if shutil.which("pbcopy"):
        subprocess.run(["pbcopy"], input=text, text=True, check=False)
        print("(Copied to the clipboard.)\n")
    return text, voice, label


def main() -> None:
    """Parse the command line, run the checks, and start the MLX web server.

    Runs `python -m personaplex_mlx.local_web` in the foreground with the
    persona as its default text prompt (the page's own Text Prompt box takes
    precedence), until Ctrl+C.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--persona", default="chef", choices=list(PERSONAS),
                        help="Persona to print and copy for the page (default: chef)")
    parser.add_argument("--quant", default="4", choices=["4", "8", "none"],
                        help="Weight quantization: 4 (default, fastest), 8, or none (bf16)")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open the page in the browser automatically")
    args = parser.parse_args()

    check_mac()
    check_model_access()
    text, voice, _ = load_persona(args.persona)

    cmd = [sys.executable, "-m", "personaplex_mlx.local_web",
           "--voice", voice, "--text-prompt", text, "--port", str(args.port)]
    if args.quant != "none":
        cmd += ["-q", args.quant]
    if args.no_browser:
        cmd.append("--no-browser")

    weights = f"{args.quant}-bit" if args.quant != "none" else "bf16"
    print(f"Loading PersonaPlex ({weights} weights). "
          f"When the log says 'listening', open http://localhost:{args.port}")
    print("Use headphones, so the model doesn't hear itself. Press Ctrl+C to stop.\n")
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
