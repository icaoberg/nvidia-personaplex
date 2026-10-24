#!/usr/bin/env python
"""Talk live to a chef whose persona is built from retrieved facts (retrieve-then-prompt).

PersonaPlex only takes knowledge through its text prompt, played into the
model once before the conversation starts (see README.md). So instead of
retrieving during the conversation, this retrieves *before* it:

  1. You say what you want to cook (--topic "pasta with tomato sauce").
  2. The topic is matched against the fact cards in knowledge/cooking.txt
     with BM25, a classic keyword-ranking formula.
  3. The best cards are added to a short Chef Marco persona, up to a
     character budget, and the persona is printed and copied.
  4. The community MLX port's web server starts, as in example3, so you can
     talk to the chef live on a Mac.

Run it inside the `personaplex-mlx` conda environment (see ../environment-mlx.yml):

  python example4/serve.py --topic "pasta with tomato sauce"
  python example4/serve.py --topic "fluffy rice and roast chicken" --budget 1800
  python example4/serve.py --topic "buttermilk pancakes" --dry-run    # show the prompt only
"""

import argparse
import importlib.util
import math
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

MODEL_REPO = "nvidia/personaplex-7b-v1"  # gated: license must be accepted
HERE = Path(__file__).resolve().parent
VOICE, VOICE_LABEL = "NATM0", "NATURAL_M0"  # MLX port's ID, web page's menu label
MIN_MEMORY_GB = 16  # 4-bit weights are ~4 GB, plus the audio codec and macOS
BM25_K1, BM25_B = 1.5, 0.75  # standard BM25 parameters
KEYWORD_WEIGHT = 2  # a card's keyword line counts this many times in its score
STOPWORDS = {"a", "an", "and", "the", "to", "of", "for", "with", "in", "on", "i", "my",
             "want", "make", "how", "do", "is", "it", "some", "tonight", "cook", "cooking",
             "but", "have", "no", "not", "need", "or", "so", "what", "can", "should", "me"}

# The persona without facts. {topic} is filled in; retrieved facts follow it.
BASE_PERSONA = (
    "You enjoy teaching people to cook. Your name is Chef Marco and you are a warm, "
    "patient cooking teacher. You are talking with a student who wants to cook {topic}. "
    "Ask what ingredients and equipment they have, then teach one step at a time, explain "
    "why each step matters, and check in before moving on. Keep answers short and "
    "encouraging. If you aren't sure about something, say so. Use these facts:"
)


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

    Exits if this isn't macOS on arm64, or if `personaplex_mlx` can't be
    imported (with the setup steps); only warns for less than MIN_MEMORY_GB.
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
    (no login, or the license isn't accepted), exits with the steps to fix it.
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


def tokenize(text: str) -> list:
    """Lowercase words for matching, without stopwords, with a plural "s" removed.

    Args:
        text: Any text.

    Returns:
        The words, e.g. "Boiled Eggs" -> ["boiled", "egg"].
    """
    words = re.findall(r"[a-z]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in words if w not in STOPWORDS]


def load_cards(path: Path) -> list:
    """Read the fact cards from a knowledge file.

    Args:
        path: A file in the knowledge/cooking.txt format: paragraphs whose
            first line is "[id] keyword, keyword, ..." followed by the text.

    Returns:
        A list of {"id": str, "keywords": str, "text": str}.

    Exits, naming the card, if a paragraph doesn't start with "[id]".
    """
    cards = []
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    for block in "\n".join(lines).split("\n\n"):
        block = block.strip()
        if not block:
            continue
        header, _, body = block.partition("\n")
        match = re.fullmatch(r"\[([\w-]+)\]\s*(.*)", header.strip())
        if not match or not body.strip():
            sys.exit(f"{path}: malformed card starting {header[:40]!r}")
        cards.append({"id": match[1], "keywords": match[2], "text": " ".join(body.split())})
    return cards


def bm25_rank(query: str, cards: list) -> list:
    """Rank cards by BM25 relevance to the query.

    Each card is scored on its text plus its keyword line, counted
    KEYWORD_WEIGHT times so keywords matter more than incidental words.

    Args:
        query: What the student wants to cook.
        cards: Cards from load_cards().

    Returns:
        (score, card) pairs with a score above zero, best first.
    """
    docs = [tokenize(f"{(c['keywords'] + ' ') * KEYWORD_WEIGHT}{c['text']}") for c in cards]
    avg_len = sum(map(len, docs)) / len(docs)
    doc_freq = Counter(word for doc in docs for word in set(doc))
    ranked = []
    for card, doc in zip(cards, docs):
        counts, score = Counter(doc), 0.0
        for word in set(tokenize(query)):
            if word not in counts:
                continue
            idf = math.log(1 + (len(docs) - doc_freq[word] + 0.5) / (doc_freq[word] + 0.5))
            tf = counts[word]
            score += idf * tf * (BM25_K1 + 1) / (
                tf + BM25_K1 * (1 - BM25_B + BM25_B * len(doc) / avg_len))
        if score > 0:
            ranked.append((score, card))
    return sorted(ranked, key=lambda pair: -pair[0])


def build_persona(topic: str, ranked: list, budget: int) -> tuple:
    """Add the best cards to the base persona without going over the budget.

    Cards are taken in rank order; a card that doesn't fit is skipped and
    smaller, lower-ranked ones are still tried.

    Args:
        topic: What the student wants to cook, put into the persona.
        ranked: Output of bm25_rank().
        budget: Maximum persona length in characters.

    Returns:
        (persona, used): the prompt text, and the (score, card) pairs in it.
    """
    persona = BASE_PERSONA.format(topic=topic)
    used = []
    for score, card in ranked:
        if len(persona) + 1 + len(card["text"]) <= budget:
            persona += " " + card["text"]
            used.append((score, card))
    return persona, used


def main() -> None:
    """Retrieve facts for the topic, build and show the persona, then start the server."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--topic", required=True,
                        help='What you want to cook, e.g. "pasta with tomato sauce"')
    parser.add_argument("--budget", type=int, default=1500,
                        help="Maximum persona length in characters (default: 1500)")
    parser.add_argument("--knowledge", type=Path, default=HERE / "knowledge" / "cooking.txt",
                        help="Fact card file (default: example4/knowledge/cooking.txt)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show the retrieved cards and the persona; don't start the server")
    parser.add_argument("--no-copy", action="store_true",
                        help="Don't copy the persona to the clipboard")
    parser.add_argument("--quant", default="4", choices=["4", "8", "none"],
                        help="Weight quantization: 4 (default, fastest), 8, or none (bf16)")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open the page in the browser automatically")
    args = parser.parse_args()

    cards = load_cards(args.knowledge)
    ranked = bm25_rank(args.topic, cards)
    persona, used = build_persona(args.topic, ranked, args.budget)

    print(f"Topic: {args.topic!r}. Retrieved {len(used)} of {len(cards)} cards "
          f"({len(ranked)} matched at all):")
    for score, card in used:
        print(f"  {score:5.2f}  [{card['id']}]")
    if not used:
        print("  (none; the chef will rely on the base persona only. "
              "Try other words, or add cards to the knowledge file.)")
    print(f"\nPersona ({len(persona)} characters, budget {args.budget}). In the page, paste it "
          f"into Text Prompt and pick Voice {VOICE_LABEL}:\n\n{persona}\n")
    if not args.no_copy and shutil.which("pbcopy"):
        subprocess.run(["pbcopy"], input=persona, text=True, check=False)
        print("(Copied to the clipboard.)\n")
    if args.dry_run:
        return

    check_mac()
    check_model_access()
    cmd = [sys.executable, "-m", "personaplex_mlx.local_web",
           "--voice", VOICE, "--text-prompt", persona, "--port", str(args.port)]
    if args.quant != "none":
        cmd += ["-q", args.quant]
    if args.no_browser:
        cmd.append("--no-browser")
    print(f"Loading PersonaPlex. When the page opens (http://localhost:{args.port}), "
          "paste the persona, pick the voice, and click Connect.")
    print("Use headphones, so the model doesn't hear itself. Press Ctrl+C to stop.\n")
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
