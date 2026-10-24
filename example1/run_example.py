#!/usr/bin/env python
"""Turn a text conversation into PersonaPlex audio on a MacBook Pro M4.

PersonaPlex is speech-to-speech: it listens to audio and answers in audio.
This script lets you write your side of the conversation as text instead:

  1. Read prompts.txt (voice, role prompt, and your lines).
  2. Speak each of your lines with macOS `say`, leaving a pause after each
     one for the agent to answer, and save the result as input.wav.
  3. Run `python -m moshi.offline` on input.wav, on the Mac's CPU.
  4. Save the agent's audio and transcript, plus two mixes of both sides:
     conversation.wav (one voice at a time, `--gap` seconds apart) and
     conversation_live.wav (stereo, with the real, overlapping timing).

Outputs (in example1/output/ by default):

  input.wav              your lines as speech, with the pauses
  agent.wav              the agent's reply audio, as long as input.wav
  agent.json             the agent's raw text tokens from moshi.offline
  transcript.txt         what the agent said, as text
  conversation.wav       mono, one voice at a time
  conversation_live.wav  stereo, you on the left, the agent on the right
  voices/                CPU copy of the voice prompt (see cpu_voice_prompt_dir)

Run it inside the `personaplex` conda environment (see ../environment.yml),
logged in to Hugging Face with the model license accepted:

  python example1/run_example.py
  python example1/run_example.py --input-only       # only build input.wav
  python example1/run_example.py --mix-only --gap 1 # remix an existing run
  python example1/run_example.py --device mps       # Apple GPU (slow; see README)
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

MODEL_REPO = "nvidia/personaplex-7b-v1"  # gated: license must be accepted
SAMPLE_RATE = 24000  # PersonaPlex (Mimi) sample rate
LEAD_IN_SECONDS = 1.0  # silence before the first user turn
MIN_MEMORY_GB = 24  # ~15 GB of BF16 weights plus macOS and the audio codec
SPECIAL_TOKENS = {"EPAD", "BOS", "EOS", "PAD"}
FRAME = SAMPLE_RATE // 20  # 50 ms frames for speech detection
SPEECH_RMS = 0.01  # frames louder than this count as speech
MIN_GAP_SECONDS = 0.4  # shorter silences stay inside one speech segment
PAD_SECONDS = 0.1  # kept around each clip so word onsets aren't cut
HERE = Path(__file__).resolve().parent


def sysctl(name: str) -> str:
    """Read a macOS system setting with `sysctl -n`.

    Args:
        name: The sysctl key, e.g. "machdep.cpu.brand_string" or "hw.memsize".

    Returns:
        The value as a string, or "" if the key doesn't exist.
    """
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def check_mac() -> None:
    """Require macOS on Apple Silicon; warn if this isn't an M4 or memory is low.

    Prints the chip, memory, and macOS version. Exits if this isn't macOS on
    arm64 or if the `say` command is missing; only warns (on stderr) for a
    non-M4 chip or less than MIN_MEMORY_GB of memory.
    """
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


def check_model_access() -> None:
    """Fail early, with instructions, if the gated model can't be downloaded.

    Downloads the model's small config.json. If Hugging Face denies access
    (no login, or the license isn't accepted), exits with the steps to fix
    it instead of letting moshi.offline fail later with a long traceback.
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
            f"  3. Run `huggingface-cli login` (or `export HF_TOKEN=<token>`)\n"
            f"Then run this script again. input.wav is already built."
        )


def cpu_voice_prompt_dir(voice: str, out_dir: Path) -> Path:
    """Save a CPU copy of the voice prompt and return the folder holding it.

    The published voice files were saved from a CUDA GPU, and moshi loads them
    with torch.load(path) and no map_location, which fails on a Mac with
    "Attempting to deserialize object on a CUDA device". A copy saved from the
    CPU loads anywhere, without patching the upstream submodule.

    Args:
        voice: Voice name without ".pt", e.g. "NATF2".
        out_dir: Output folder; the copy goes in out_dir/voices/.

    Returns:
        The folder holding the copy, to pass to moshi.offline as
        --voice-prompt-dir.

    Exits if `voice` isn't one of the published voices.
    """
    import tarfile

    import torch
    from huggingface_hub import hf_hub_download

    voices_tgz = Path(hf_hub_download(MODEL_REPO, "voices.tgz"))
    src_dir = voices_tgz.parent / "voices"
    if not src_dir.exists():
        with tarfile.open(voices_tgz) as tar:
            tar.extractall(voices_tgz.parent, filter="data")

    src = src_dir / f"{voice}.pt"
    if not src.exists():
        names = ", ".join(sorted(p.stem for p in src_dir.glob("*.pt")))
        sys.exit(f"Unknown voice {voice!r}. Available: {names}")

    dst_dir = out_dir / "voices"
    dst_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(src, map_location="cpu", weights_only=True)
    torch.save(state, dst_dir / src.name)
    return dst_dir


def parse_prompts(path: Path) -> dict:
    """Parse the key: value prompts file into voice, role, and turns.

    See prompts.txt for the format. A `pause:` line applies to the turns
    after it, and `audio:` paths are resolved relative to the prompts file.

    Args:
        path: The prompts file.

    Returns:
        {"voice": str, "role": str, "turns": list}, where each turn is
        {"kind": "text" | "audio", "value": str, "pause": float}: the line
        to speak (or the WAV path) and the seconds of silence after it.

    Exits, naming the line, on a malformed line, an unknown key, a missing
    audio file, or a file with no turns.
    """
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
    """Speak `text` into a 24 kHz WAV file with macOS `say`.

    Args:
        text: What to say.
        say_voice: macOS voice name (see `say -v '?'`), or "" for the default.
        workdir: Folder for the WAV file.
        index: Turn number, used in the file name (turnNN.wav).

    Returns:
        Path to the 16-bit, 24 kHz mono WAV file.
    """
    wav = workdir / f"turn{index:02d}.wav"
    cmd = ["say", "-o", str(wav), "--file-format=WAVE", f"--data-format=LEI16@{SAMPLE_RATE}"]
    if say_voice:
        cmd += ["-v", say_voice]
    subprocess.run(cmd + [text], check=True)
    return wav


def load_mono(path: str) -> np.ndarray:
    """Load any WAV as mono float32 at SAMPLE_RATE.

    Args:
        path: The WAV file. Any sample rate and channel count; channels are
            averaged and the audio is resampled.

    Returns:
        1-D float32 samples at SAMPLE_RATE.
    """
    pcm, sr = sphn.read(path)  # (channels, samples)
    pcm = sphn.resample(pcm, src_sample_rate=sr, dst_sample_rate=SAMPLE_RATE)
    return pcm.mean(axis=0).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    """Return `seconds` of silence as float32 samples at SAMPLE_RATE."""
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def build_input(turns: list, say_voice: str) -> tuple:
    """Concatenate the user's turns, each followed by its pause.

    The audio starts with LEAD_IN_SECONDS of silence. Text turns are spoken
    with `say` (in a temporary folder); audio turns are loaded from their WAV.

    Args:
        turns: Turns from parse_prompts().
        say_voice: macOS voice for text turns, or "" for the default.

    Returns:
        (audio, spans): the float32 audio at SAMPLE_RATE, and one
        (start, end) sample span per turn, covering just that turn's speech
        clip (not its pause). write_turn_mix() uses the spans to find each turn.
    """
    pieces = [silence(LEAD_IN_SECONDS)]
    spans = []
    position = len(pieces[0])
    with tempfile.TemporaryDirectory() as tmp:
        for i, turn in enumerate(turns):
            if turn["kind"] == "text":
                print(f"  [say]   {turn['value']}")
                wav = synthesize(turn["value"], say_voice, Path(tmp), i)
            else:
                print(f"  [audio] {turn['value']}")
                wav = turn["value"]
            speech = load_mono(str(wav))
            spans.append((position, position + len(speech)))
            pieces += [speech, silence(turn["pause"])]
            position += len(speech) + len(pieces[-1])
    return np.concatenate(pieces), spans


def speech_segments(pcm: np.ndarray) -> list:
    """(start, end) sample spans where `pcm` is louder than SPEECH_RMS.

    The audio is cut into FRAME-sample (50 ms) frames, and a frame counts as
    speech if its RMS level is above SPEECH_RMS. Gaps shorter than
    MIN_GAP_SECONDS are merged, so pauses between words and sentences stay
    inside one segment.

    Args:
        pcm: 1-D float audio at SAMPLE_RATE.

    Returns:
        A list of (start, end) sample indices, in order, end exclusive.
    """
    n = len(pcm) // FRAME
    rms = np.sqrt((pcm[:n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1))
    segments = []
    for i in np.flatnonzero(rms > SPEECH_RMS):
        start, end = i * FRAME, (i + 1) * FRAME
        if segments and start - segments[-1][1] < MIN_GAP_SECONDS * SAMPLE_RATE:
            segments[-1][1] = end
        else:
            segments.append([start, end])
    return [tuple(s) for s in segments]


def run_personaplex(args: argparse.Namespace, prompts: dict, input_wav: Path,
                    agent_wav: Path, agent_json: Path) -> None:
    """Run PersonaPlex's offline inference on the input WAV.

    Runs `python -m moshi.offline` in a subprocess, with the CPU voice copy
    from cpu_voice_prompt_dir() and the role prompt (if any). With
    --device mps, sets PYTORCH_ENABLE_MPS_FALLBACK=1 so operations MPS
    doesn't implement run on the CPU instead of failing.

    Args:
        args: Parsed command-line arguments (uses out_dir, device, seed).
        prompts: Output of parse_prompts() (uses voice and role).
        input_wav: The user audio to play to the model.
        agent_wav: Where moshi.offline writes the agent's audio.
        agent_json: Where moshi.offline writes the agent's text tokens.

    Exits with a one-line message if moshi.offline fails; its own error is
    printed above.
    """
    cmd = [
        sys.executable, "-m", "moshi.offline",
        "--voice-prompt", f"{prompts['voice']}.pt",
        "--voice-prompt-dir", str(cpu_voice_prompt_dir(prompts["voice"], args.out_dir)),
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
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(f"moshi.offline failed (exit code {e.returncode}); see the error above.")


def write_transcript(agent_json: Path, transcript: Path) -> str:
    """Join the agent's text tokens into readable text.

    Args:
        agent_json: moshi.offline's text output: a JSON list with one token
            per model frame (word pieces, or EPAD/BOS/EOS/PAD).
        transcript: Where to write the joined text.

    Returns:
        The text, without the special tokens.
    """
    tokens = json.loads(agent_json.read_text())
    text = "".join(t for t in tokens if t not in SPECIAL_TOKENS).strip()
    transcript.write_text(text + "\n")
    return text


def write_live_mix(user: np.ndarray, agent: np.ndarray, out: Path) -> None:
    """Stereo, time-aligned: user on the left, agent on the right, as the
    model heard and answered them (the voices can overlap).

    Args:
        user: The user audio (input.wav), float32 at SAMPLE_RATE.
        agent: The agent audio (agent.wav), float32 at SAMPLE_RATE.
        out: Where to write the stereo WAV. The shorter side is padded with
            silence.
    """
    length = max(len(user), len(agent))
    stereo = np.zeros((2, length), dtype=np.float32)
    stereo[0, :len(user)] = user
    stereo[1, :len(agent)] = agent
    sphn.write_wav(str(out), stereo, SAMPLE_RATE)


def write_turn_mix(user: np.ndarray, spans: list, agent: np.ndarray, out: Path,
                   gap: float) -> int:
    """Mono, one voice at a time: each user turn, `gap` seconds of silence,
    the agent's answer, `gap` seconds, the next turn.

    PersonaPlex is full-duplex, so its answers can run into the next user
    turn and it makes short sounds ("okay", "right") while the user talks.
    Each agent segment is placed by when it starts: before the first turn,
    it opens the conversation; during or after a turn, it is that turn's
    answer and is kept whole, unless it also ends before the user stops
    talking (a backchannel), in which case it is dropped.

    Args:
        user: The user audio (input.wav), float32 at SAMPLE_RATE.
        spans: Each turn's (start, end) samples in `user`, from build_input().
            Leading and trailing silence inside a span is trimmed.
        agent: The agent audio (agent.wav), aligned with `user`.
        out: Where to write the mono WAV.
        gap: Seconds of silence between voices. Each clip keeps PAD_SECONDS
            of near-silence at both ends, which counts toward the gap.

    Returns:
        The number of agent segments dropped as backchannels.
    """
    pad = int(PAD_SECONDS * SAMPLE_RATE)

    turns = []  # user speech bounds, without say's leading/trailing silence
    for start, end in spans:
        segments = speech_segments(user[start:end]) or [(0, end - start)]
        turns.append((start + segments[0][0], start + segments[-1][1]))

    opening, answers, dropped = [], [[] for _ in turns], 0
    for start, end in speech_segments(agent):
        if start < turns[0][0]:
            opening.append((start, end))
            continue
        i = max(j for j, (turn_start, _) in enumerate(turns) if turn_start <= start)
        if end <= turns[i][1]:
            dropped += 1
        else:
            answers[i].append((start, end))

    def clip(pcm: np.ndarray, segments: list) -> np.ndarray:
        """Cut `pcm` from the first segment's start to the last one's end,
        plus `pad` samples on each side."""
        return pcm[max(segments[0][0] - pad, 0):segments[-1][1] + pad]

    # Each clip carries `pad` of near-silence on both ends; count it in the gap.
    between = silence(max(gap - 2 * PAD_SECONDS, 0))
    pieces = [silence(gap)]
    if opening:
        pieces += [clip(agent, opening), between]
    for turn, answer in zip(turns, answers):
        pieces += [clip(user, [turn]), between]
        if answer:
            pieces += [clip(agent, answer), between]
    sphn.write_wav(str(out), np.concatenate(pieces), SAMPLE_RATE)
    return dropped


def main() -> None:
    """Parse the command line, build input.wav, run the model, and write the outputs.

    --input-only stops after input.wav. --mix-only skips the model and
    remakes transcript.txt and both conversation mixes from the existing
    agent.wav and agent.json.
    """
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
    parser.add_argument("--gap", type=float, default=0.5,
                        help="Seconds of silence between turns in conversation.wav (default: 0.5)")
    parser.add_argument("--input-only", action="store_true",
                        help="Only build input.wav; don't run the model")
    parser.add_argument("--mix-only", action="store_true",
                        help="Remake the conversation mixes from an existing agent.wav; "
                             "don't run the model")
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
    user, spans = build_input(prompts["turns"], args.say_voice)
    sphn.write_wav(str(input_wav), user, SAMPLE_RATE)
    print(f"Wrote {input_wav} ({len(user) / SAMPLE_RATE:.1f} s)")

    if args.input_only:
        return
    if args.mix_only:
        if not agent_wav.exists():
            sys.exit(f"{agent_wav} not found; run without --mix-only first.")
    else:
        check_model_access()
        run_personaplex(args, prompts, input_wav, agent_wav, agent_json)
    text = write_transcript(agent_json, args.out_dir / "transcript.txt")
    agent = load_mono(str(agent_wav))
    write_live_mix(user, agent, args.out_dir / "conversation_live.wav")
    dropped = write_turn_mix(user, spans, agent, args.out_dir / "conversation.wav", args.gap)

    print("\nAgent said:\n  " + (text or "(nothing)"))
    print(f"\nconversation.wav: one voice at a time, {args.gap:g} s between turns "
          f"({dropped} agent sounds made while you spoke were left out)")
    print(f"\nResults in {args.out_dir}/:")
    for name in ["input.wav", "agent.wav", "conversation.wav", "conversation_live.wav",
                 "transcript.txt", "agent.json"]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
