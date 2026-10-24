#!/usr/bin/env python
"""Generate a two-voice SenNet conversation with PersonaPlex on a MacBook Pro M4.

PersonaPlex makes both voices; there is no text-to-speech. The model is
loaded once and given a different voice and role prompt for each turn, the
way moshi.server does for each connection:

  1. Student turn: the student's question is forced into the model's text
     stream, paced like speech, so the model says exactly that question in
     the student's voice.
  2. Guide turn: with the guide's voice and role prompt (the SenNet facts),
     the model hears the student's question and answers it freely.
  3. The speech from each turn is cut out and joined one voice at a time,
     with a short gap between voices, so they never overlap.

Outputs (in example2/output/ by default):

  conversation.wav   the whole conversation, one voice at a time
  transcript.txt     who said what, from the model's text stream
  turns/NN_student.wav, turns/NN_guide.wav   each turn's clip
  config.json        the prompts and settings used for this run
  voices/            CPU copies of the two voice prompts

Run it inside the `personaplex` conda environment (see ../environment.yml),
logged in to Hugging Face with the model license accepted:

  python example2/run_example.py
  python example2/run_example.py --gap 1          # longer silence between voices
  python example2/run_example.py --device mps     # Apple GPU (slow; see README)
"""

import argparse
import json
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import sphn

MODEL_REPO = "nvidia/personaplex-7b-v1"  # gated: license must be accepted
SAMPLE_RATE = 24000  # PersonaPlex (Mimi) sample rate
MIN_MEMORY_GB = 24  # ~15 GB of BF16 weights plus macOS and the audio codec
FRAME = SAMPLE_RATE // 20  # 50 ms frames for speech detection
SPEECH_RMS = 0.01  # frames louder than this count as speech
MIN_GAP_SECONDS = 0.4  # shorter silences stay inside one speech segment
PAD_SECONDS = 0.1  # kept around each clip so word onsets aren't cut
ATTEMPTS = 3  # tries per guide turn (with different seeds) if it stays silent

# Text stream tokens. While speaking, the model's text stream holds one token
# per 80 ms frame: EPAD just before each word, the word's pieces, then PAD for
# as long as the word takes to say (measured from PersonaPlex's own output).
EPAD, PAD = 0, 3
FRAMES_PER_CHAR = 0.75  # "public" ~5 frames, "datasets" ~6
LEAD_FRAMES = 6  # silence before the forced words (~0.5 s)
TAIL_FRAMES = 20  # silence after them, so the audio can finish (~1.6 s)
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
    arm64; only warns (on stderr) for a non-M4 chip or less than
    MIN_MEMORY_GB of memory.
    """
    if sys.platform != "darwin" or platform.machine() != "arm64":
        sys.exit("This example targets a MacBook Pro M4 (macOS on Apple Silicon).")
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
    it before the 17 GB model starts loading.
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


def cpu_voice_prompt(voice: str, out_dir: Path) -> Path:
    """Save a CPU copy of a voice prompt and return its path.

    The published voice files were saved from a CUDA GPU, and moshi loads them
    with torch.load(path) and no map_location, which fails on a Mac. A copy
    saved from the CPU loads anywhere. (Same workaround as example1.)

    Args:
        voice: Voice name without ".pt", e.g. "NATM1".
        out_dir: Output folder; the copy goes in out_dir/voices/.

    Returns:
        Path to the copied .pt file, for PersonaPlex.respond().

    Exits if `voice` isn't one of the published voices.
    """
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

    dst = out_dir / "voices" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.load(src, map_location="cpu", weights_only=True), dst)
    return dst


def parse_prompts(path: Path) -> dict:
    """Parse the key: value prompts file.

    See prompts.txt for the format. Repeated `student_role:` and
    `guide_role:` lines are joined with spaces; each `question:` line is one
    question, in order.

    Args:
        path: The prompts file.

    Returns:
        {"student_voice": str, "guide_voice": str, "student_role": str,
        "guide_role": str, "question": list[str], "answer_time": float},
        with defaults for anything the file leaves out.

    Exits, naming the line, on a malformed line or an unknown key, or if
    there are no questions.
    """
    config = {"student_voice": "NATM1", "guide_voice": "NATF2",
              "student_role": [], "guide_role": [], "question": [],
              "answer_time": 15.0}

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if not sep or not value:
            sys.exit(f"{path}:{lineno}: expected 'key: value', got {raw!r}")
        if key not in config:
            sys.exit(f"{path}:{lineno}: unknown key {key!r}")

        if isinstance(config[key], list):
            config[key].append(value)
        elif isinstance(config[key], float):
            config[key] = float(value)
        else:
            config[key] = value.removesuffix(".pt")

    if not config["question"]:
        sys.exit(f"{path}: no 'question:' lines found")
    for key in ("student_role", "guide_role"):
        config[key] = " ".join(config[key])
    return config


def silence(seconds: float) -> np.ndarray:
    """Return `seconds` of silence as float32 samples at SAMPLE_RATE."""
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


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


class PersonaPlex:
    """PersonaPlex loaded once, able to speak as any voice and role.

    Mirrors moshi.offline.run_inference, split into a one-time load and a
    per-turn `respond`.

    Attributes:
        mimi, other_mimi: Mimi audio codecs (encoder/decoder), as in moshi.
        tokenizer: SentencePiece tokenizer for the text stream.
        lm_gen: The streaming generator wrapping the 7B language model.
        frame_size: Samples per model frame (1920, i.e. 80 ms at 24 kHz).
    """

    def __init__(self, device: str):
        """Download (if needed) and load the model, then warm it up.

        Args:
            device: Torch device, "cpu" or "mps".
        """
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import LMGen, loaders
        from moshi.offline import warmup

        print("Loading PersonaPlex (first run downloads ~17 GB)...")
        mimi_weight = hf_hub_download(MODEL_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device)
        self.other_mimi = loaders.get_mimi(mimi_weight, device)
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            hf_hub_download(MODEL_REPO, loaders.TEXT_TOKENIZER_NAME))
        lm = loaders.get_moshi_lm(hf_hub_download(MODEL_REPO, loaders.MOSHI_NAME), device=device)
        lm.eval()

        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=device,
            frame_rate=self.mimi.frame_rate,
        )
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        warmup(self.mimi, self.other_mimi, self.lm_gen, device, self.frame_size)

    def script(self, text: str) -> list:
        """Text-stream tokens that make the model say `text`, one per frame.

        Follows the rhythm PersonaPlex uses when it speaks on its own:
        LEAD_FRAMES of PAD, then for each word an EPAD, the word's
        SentencePiece pieces, and PAD for about FRAMES_PER_CHAR frames per
        character; extra PADs after punctuation; then TAIL_FRAMES of PAD so
        the audio can finish.

        Args:
            text: The words to say.

        Returns:
            A list of token IDs, one per 80 ms frame, for respond(script=...).
        """
        tokens = [PAD] * LEAD_FRAMES
        for word in text.split():
            pieces = self.tokenizer.encode(word)
            frames = max(int(FRAMES_PER_CHAR * len(word) + 0.5), len(pieces) + 1)
            tokens += [EPAD] + pieces + [PAD] * (frames - 1 - len(pieces))
            if word[-1] in ".?!":
                tokens += [PAD] * 5  # pause between sentences
            elif word[-1] in ",;:":
                tokens += [PAD] * 2
        return tokens + [PAD] * TAIL_FRAMES

    def respond(self, voice_prompt: Path, role: str, heard: np.ndarray,
                listen_seconds: float, seed: int, script: list | None = None) -> tuple:
        """Play `heard` to the model as the other person, then `listen_seconds`
        of silence, and return what the model said: audio aligned with the
        input, and one text piece per 80 ms model frame ("" for none).

        With `script`, the text stream is forced one token per frame (then PAD),
        so the model says the scripted words instead of choosing its own.

        Each call starts fresh: it resets the model's streaming state and
        replays the voice and role prompts, so turns don't share memory.

        Args:
            voice_prompt: CPU copy of the voice (.pt), from cpu_voice_prompt().
            role: The role (text) prompt for this turn.
            heard: The other speaker's audio, float32 at SAMPLE_RATE. May be
                empty.
            listen_seconds: Seconds of silence after `heard`, the speaker's
                time to talk.
            seed: Random seed for this turn.
            script: Optional token IDs from script() to force the words.

        Returns:
            (pcm, texts): the model's audio, as long as `heard` plus the
            silence, and its text stream as one string per decoded frame.
        """
        import torch
        from moshi.models.lm import _iterate_audio, encode_from_sphn
        from moshi.offline import decode_tokens_to_pcm, seed_all, wrap_with_system_tags

        seed_all(seed)
        self.lm_gen.load_voice_prompt_embeddings(str(voice_prompt))
        self.lm_gen.text_prompt_tokens = self.tokenizer.encode(wrap_with_system_tags(role))
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.lm_gen.step_system_prompts(self.mimi)
        self.mimi.reset_streaming()

        audio = np.concatenate([heard, silence(listen_seconds)])[None, :]
        frames, texts, step = [], [], 0
        with torch.no_grad():
            chunks = _iterate_audio(audio, sample_interval_size=self.frame_size, pad=True)
            for encoded in encode_from_sphn(self.mimi, chunks, max_batch=1):
                for c in range(encoded.shape[-1]):
                    forced = None
                    if script is not None:
                        forced = script[step] if step < len(script) else PAD
                    step += 1
                    tokens = self.lm_gen.step(encoded[:, :, c:c + 1], text_token=forced)
                    if tokens is None:
                        continue
                    frames.append(decode_tokens_to_pcm(self.mimi, self.other_mimi,
                                                       self.lm_gen, tokens))
                    token = tokens[0, 0, 0].item()
                    texts.append("" if token < 4 else  # EPAD, BOS, EOS, PAD
                                 self.tokenizer.id_to_piece(token).replace("▁", " "))

        pcm = np.concatenate(frames)[:audio.shape[-1]]
        return np.pad(pcm, (0, audio.shape[-1] - len(pcm))), texts


def clip_reply(pcm: np.ndarray, texts: list, frame_size: int, heard_samples: int) -> tuple:
    """Cut out what the speaker said after the heard audio ended.

    Speech that ends before the heard audio ends (an "mm-hm" over the other
    person) is dropped. The clip runs from the first kept segment to the
    last, plus PAD_SECONDS on each side.

    Args:
        pcm: The speaker's audio from PersonaPlex.respond().
        texts: The matching text pieces, one per model frame.
        frame_size: Samples per model frame, to line texts up with pcm.
        heard_samples: Length of the audio the speaker heard first.

    Returns:
        (clip, text, cut_off): the audio clip, the text said in it, and
        whether the speaker was still talking when its time ran out.
        (None, "", False) if the speaker said nothing.
    """
    segments = [s for s in speech_segments(pcm) if s[1] > heard_samples]
    if not segments:
        return None, "", False
    pad = int(PAD_SECONDS * SAMPLE_RATE)
    start, end = max(segments[0][0] - pad, 0), segments[-1][1] + pad
    text = "".join(texts[start // frame_size:end // frame_size + 1]).strip()
    cut_off = segments[-1][1] >= len(pcm) - FRAME  # still talking when time ran out
    return pcm[start:end], text, cut_off


def main() -> None:
    """Parse the command line, run every student and guide turn, and write the outputs.

    For each question: the student says it (forced text), then the guide
    answers, retried with new seeds up to ATTEMPTS times if it stays silent.
    A question the student produces no audio for is skipped.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompts", nargs="?", default=HERE / "prompts.txt", type=Path,
                        help="Prompts file (default: example2/prompts.txt)")
    parser.add_argument("--out-dir", default=HERE / "output", type=Path,
                        help="Where to write results (default: example2/output)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                        help="cpu (default) or mps, the Apple GPU (untested upstream)")
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--gap", type=float, default=0.5,
                        help="Seconds of silence between voices (default: 0.5)")
    args = parser.parse_args()

    check_mac()
    config = parse_prompts(args.prompts)
    check_model_access()
    turns_dir = args.out_dir / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)

    student_voice = cpu_voice_prompt(config["student_voice"], args.out_dir)
    guide_voice = cpu_voice_prompt(config["guide_voice"], args.out_dir)
    model = PersonaPlex(args.device)
    frame_seconds = model.frame_size / SAMPLE_RATE

    # Each clip carries PAD_SECONDS of near-silence on both ends; count it in the gap.
    between = silence(max(args.gap - 2 * PAD_SECONDS, 0))
    pieces, transcript = [silence(args.gap)], []
    for i, question in enumerate(config["question"], start=1):
        print(f"\nTurn {i}/{len(config['question'])}")

        script = model.script(question)
        pcm, texts = model.respond(student_voice, config["student_role"], silence(0),
                                   len(script) * frame_seconds, args.seed + i, script=script)
        asked, asked_text, _ = clip_reply(pcm, texts, model.frame_size, 0)
        if asked is None:
            print("  Student: (no audio; skipping this question)")
            continue
        print(f"  Student: {asked_text}")

        answer = None
        for attempt in range(ATTEMPTS):
            pcm, texts = model.respond(guide_voice, config["guide_role"], asked,
                                       config["answer_time"],
                                       args.seed + 100 + i + 1000 * attempt)
            answer, answer_text, cut_off = clip_reply(pcm, texts, model.frame_size, len(asked))
            if answer is not None:
                break
            print("  Guide:   (silent, retrying with a new seed)")
        print(f"  Guide:   {answer_text or '(no text)'}" if answer is not None
              else "  Guide:   (silent; skipping)")
        if answer is not None and cut_off:
            print(f"           (cut off after {config['answer_time']:g} s; raise answer_time)")

        sphn.write_wav(str(turns_dir / f"{i:02d}_student.wav"), asked, SAMPLE_RATE)
        pieces += [asked, between]
        transcript.append(f"Student: {asked_text}")
        if answer is not None:
            sphn.write_wav(str(turns_dir / f"{i:02d}_guide.wav"), answer, SAMPLE_RATE)
            pieces += [answer, between]
            transcript.append(f"Guide: {answer_text}")

    conversation = np.concatenate(pieces)
    sphn.write_wav(str(args.out_dir / "conversation.wav"), conversation, SAMPLE_RATE)
    (args.out_dir / "transcript.txt").write_text("\n\n".join(transcript) + "\n")
    (args.out_dir / "config.json").write_text(json.dumps(
        {**config, "seed": args.seed, "gap": args.gap, "device": args.device}, indent=2) + "\n")

    print(f"\nWrote {args.out_dir}/conversation.wav ({len(conversation) / SAMPLE_RATE:.1f} s, "
          f"one voice at a time, {args.gap:g} s between voices)")


if __name__ == "__main__":
    main()
