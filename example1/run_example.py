#!/usr/bin/env python
"""Turn a text conversation into PersonaPlex audio on a MacBook Pro M4.

PersonaPlex is speech-to-speech: it listens to audio and answers in audio.
This script lets you write your side of the conversation as text instead:

  1. Read prompts.txt (voice, role prompt, and your lines).
  2. Speak each of your lines with macOS `say`, leaving a pause after each
     one for the agent to answer, and save the result as input.wav.
  3. Run `python -m moshi.offline` on input.wav, on the Mac's CPU.
  4. Save the agent's audio, its transcript, and a stereo mix of both sides.

Run it inside the `personaplex` conda environment (see ../environment.yml),
with HF_TOKEN set and the model license accepted on Hugging Face:

  python example1/run_example.py
  python example1/run_example.py --device mps       # Apple GPU (untested upstream)
  python example1/run_example.py --input-only       # only build input.wav
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import sphn

SAMPLE_RATE = 24000  # PersonaPlex (Mimi) sample rate
LEAD_IN_SECONDS = 1.0  # silence before the first user turn
MIN_MEMORY_GB = 24  # ~15 GB of BF16 weights plus macOS and the audio codec
SPECIAL_TOKENS = {"EPAD", "BOS", "EOS", "PAD"}
HERE = Path(__file__).resolve().parent


def sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def check_mac():
    """Require macOS on Apple Silicon; warn if this isn't an M4 or memory is low."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        sys.exit("This example targets a MacBook Pro M4 (macOS on Apple Silicon). "
                 "On Linux with an NVIDIA GPU, run `python -m moshi.offline` directly; "
                 "see the main README.")
    if not shutil.which("say"):
        sys.exit("macOS `say` command not found.")

    chip = sysctl("machdep.cpu.brand_string")
    memory_gb = int(sysctl("hw.memsize") or 0) / 2**30
    print(f"Mac:   {chip}, {memory_gb:.0f} GB unified memory, macOS {platform.mac_ver()[0]}")
    if "M4" not in chip:
        print(f"Warning: written for an M4; this is {chip!r}. Continuing.", file=sys.stderr)
    if memory_gb < MIN_MEMORY_GB:
        print(f"Warning: {memory_gb:.0f} GB of memory is below the ~{MIN_MEMORY_GB} GB "
              "PersonaPlex needs; expect heavy swapping or an out-of-memory failure.",
              file=sys.stderr)


def parse_prompts(path: Path) -> dict:
    """Parse the key: value prompts file into voice, role, and turns."""
    voice = "NATF2"
    role_lines = []
    pause = 6.0
    turns = []  # dicts: {"kind": "text"|"audio", "value": str, "pause": float}

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if not sep or not value:
            sys.exit(f"{path}:{lineno}: expected 'key: value', got {raw!r}")

        if key == "voice":
            voice = value.removesuffix(".pt")
        elif key == "role":
            role_lines.append(value)
        elif key == "pause":
            pause = float(value)
        elif key == "user":
            turns.append({"kind": "text", "value": value, "pause": pause})
        elif key == "audio":
            audio_path = (path.parent / value).resolve()
            if not audio_path.exists():
                sys.exit(f"{path}:{lineno}: audio file not found: {audio_path}")
            turns.append({"kind": "audio", "value": str(audio_path), "pause": pause})
        else:
            sys.exit(f"{path}:{lineno}: unknown key {key!r}")

    if not turns:
        sys.exit(f"{path}: no 'user:' or 'audio:' lines found")
    return {"voice": voice, "role": " ".join(role_lines), "turns": turns}


def synthesize(text: str, say_voice: str, workdir: Path, index: int) -> Path:
    """Speak `text` into a 24 kHz WAV file with macOS `say`."""
    wav = workdir / f"turn{index:02d}.wav"
    cmd = ["say", "-o", str(wav), "--file-format=WAVE", f"--data-format=LEI16@{SAMPLE_RATE}"]
    if say_voice:
        cmd += ["-v", say_voice]
    subprocess.run(cmd + [text], check=True)
    return wav


def load_mono(path: str) -> np.ndarray:
    """Load any WAV as mono float32 at SAMPLE_RATE."""
    pcm, sr = sphn.read(path)  # (channels, samples)
    pcm = sphn.resample(pcm, src_sample_rate=sr, dst_sample_rate=SAMPLE_RATE)
    return pcm.mean(axis=0).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def build_input(turns: list, say_voice: str) -> np.ndarray:
    """Concatenate the user's turns, each followed by its pause."""
    pieces = [silence(LEAD_IN_SECONDS)]
    with tempfile.TemporaryDirectory() as tmp:
        for i, turn in enumerate(turns):
            if turn["kind"] == "text":
                print(f"  [say]   {turn['value']}")
                wav = synthesize(turn["value"], say_voice, Path(tmp), i)
            else:
                print(f"  [audio] {turn['value']}")
                wav = turn["value"]
            pieces.append(load_mono(str(wav)))
            pieces.append(silence(turn["pause"]))
    return np.concatenate(pieces)


def run_personaplex(args, prompts: dict, input_wav: Path, agent_wav: Path, agent_json: Path):
    """Run PersonaPlex's offline inference on the input WAV."""
    cmd = [
        sys.executable, "-m", "moshi.offline",
        "--voice-prompt", f"{prompts['voice']}.pt",
        "--input-wav", str(input_wav),
        "--output-wav", str(agent_wav),
        "--output-text", str(agent_json),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    if prompts["role"]:
        cmd += ["--text-prompt", prompts["role"]]
    env = dict(os.environ)
    if args.device == "mps":
        # Run ops that MPS doesn't implement on the CPU instead of failing.
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print(f"Running moshi.offline on {args.device}. On a Mac this takes a while.")
    subprocess.run(cmd, check=True, env=env)


def write_transcript(agent_json: Path, transcript: Path) -> str:
    """Join the agent's text tokens into readable text."""
    tokens = json.loads(agent_json.read_text())
    text = "".join(t for t in tokens if t not in SPECIAL_TOKENS).strip()
    transcript.write_text(text + "\n")
    return text


def write_conversation(user: np.ndarray, agent_wav: Path, out: Path):
    """Stereo mix: user on the left channel, agent on the right."""
    agent = load_mono(str(agent_wav))
    length = max(len(user), len(agent))
    stereo = np.zeros((2, length), dtype=np.float32)
    stereo[0, :len(user)] = user
    stereo[1, :len(agent)] = agent
    sphn.write_wav(str(out), stereo, SAMPLE_RATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompts", nargs="?", default=HERE / "prompts.txt", type=Path,
                        help="Prompts file (default: example1/prompts.txt)")
    parser.add_argument("--out-dir", default=HERE / "output", type=Path,
                        help="Where to write results (default: example1/output)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                        help="cpu (default) or mps, the Apple GPU (untested upstream)")
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--say-voice", default="",
                        help="macOS voice for your lines, e.g. Samantha (see `say -v '?'`)")
    parser.add_argument("--input-only", action="store_true",
                        help="Only build input.wav; don't run the model")
    args = parser.parse_args()

    check_mac()
    prompts = parse_prompts(args.prompts)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_wav = args.out_dir / "input.wav"
    agent_wav = args.out_dir / "agent.wav"
    agent_json = args.out_dir / "agent.json"

    print(f"Voice: {prompts['voice']}")
    print(f"Role:  {prompts['role'] or '(upstream default)'}")
    print(f"Building {input_wav} from {len(prompts['turns'])} user turns:")
    user = build_input(prompts["turns"], args.say_voice)
    sphn.write_wav(str(input_wav), user, SAMPLE_RATE)
    print(f"Wrote {input_wav} ({len(user) / SAMPLE_RATE:.1f} s)")

    if args.input_only:
        return
    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN is not set; the model download will fail "
              "unless you ran `huggingface-cli login`.", file=sys.stderr)

    run_personaplex(args, prompts, input_wav, agent_wav, agent_json)
    text = write_transcript(agent_json, args.out_dir / "transcript.txt")
    write_conversation(user, agent_wav, args.out_dir / "conversation.wav")

    print("\nAgent said:\n  " + (text or "(nothing)"))
    print(f"\nResults in {args.out_dir}/:")
    for name in ["input.wav", "agent.wav", "conversation.wav", "transcript.txt", "agent.json"]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
