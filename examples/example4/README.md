# Example 4: Giving the guide more expertise with retrieval (RAG)

Can retrieval-augmented generation (RAG) make PersonaPlex's guide more of an expert? This example answers that question and builds the approach that fits PersonaPlex best: **retrieve before the conversation, then prompt**. You say what you want to cook. The script picks the most relevant facts from a cooking knowledge base, builds Chef Marco's persona from them, and starts the live web page from [example3](../example3), running on your Mac.

```
--topic "pasta with tomato sauce"
        │
        ▼
knowledge/cooking.txt ──BM25──► best fact cards ──► Chef Marco persona (≤ 1,500 chars)
   (25 fact cards)                                        │
                                                          ▼
                        browser (mic) ◄──► personaplex_mlx.local_web (MLX, 4-bit)
```

## Why plain RAG doesn't drop in

In a chat LLM, RAG means: for each question, search a knowledge base and put the results into the prompt before the model answers. PersonaPlex can't work that way:

- **Knowledge enters only through the text prompt, once.** The voice and text prompts are played into the model before the conversation starts. During the conversation there's no channel for outside text. The model's text stream is its own words, and forcing text into it makes the model *say* that text. That's how [example2](../example2) makes the student speak, so it doesn't work for slipping in background facts.
- **The prompt has a budget.** The web page's text box shows a 1,000-character counter, though it's only a counter and isn't enforced. The model was trained on short role prompts, so very long prompts likely degrade its behavior.
- **The model remembers about 4 minutes.** It attends to the last 3,000 frames (`context=3000`), about 4 minutes at 80 ms per frame, and the prompt occupies the start of that window. Reading the attention cache code (`RingKVCache`), my inference is that once a conversation runs past about 4 minutes, the persona and its facts scroll out of the window. I haven't tested this.

## Ways to add expertise

| Approach | How it works | Expertise gain | Effort |
|---|---|---|---|
| **1. Retrieve before connecting** (this example) | Keep a knowledge base of short fact cards. You pick a topic, the closest cards are retrieved, and a focused persona is built and sent, within a length budget. | Moderate: the right facts for this session, not all facts | Small: one script plus a cards file; no model changes |
| **2. Re-prompt between sessions** | Same retrieval, but reconnect with a new prompt when the topic changes. Each connection reloads the persona in about 2 s on MLX. | Moderate, across topics | Small, but the conversation restarts |
| **3. PersonaPlex as the voice, an LLM as the brain** | Transcribe the user's speech (e.g. Whisper on MLX), answer with an LLM over a full RAG index, then force the answer into PersonaPlex's text stream as example2 does. | High: real RAG over any amount of material | Large: you lose most of the natural full-duplex interplay and add latency |
| **4. Fine-tuning** | Train on domain conversations. | High within the domain | Very large: needs GPUs and conversation data. I haven't checked whether NVIDIA released training code. |

**Recommendation:** start with option 1. It fits how PersonaPlex is meant to be used and runs live on a Mac, and the right 800–1,500 characters of facts for *tonight's dish* beat a generic persona. If you need real depth, such as answering from a whole cookbook, that requires option 3. At that point PersonaPlex is mainly the voice.

## How this example works

1. **Knowledge base:** [`knowledge/cooking.txt`](knowledge/cooking.txt) holds 25 fact cards: pasta, sauces, rice, eggs, searing, roasting, baking, substitutions, and food safety with USDA temperatures. Each card is one or two spoken-style sentences, because the model treats its prompt as instructions to follow, not a document to quote.
2. **Retrieval:** [`serve.py`](serve.py) ranks the cards against your topic with [BM25](https://en.wikipedia.org/wiki/Okapi_BM25), a standard keyword-ranking formula, in pure Python with no extra packages. Each card's keyword line counts double. Filler words ("I", "want", "with", "but") are ignored, and simple plurals are matched ("eggs" ↔ "egg").
3. **Persona:** a short base persona (Chef Marco: patient, step by step, checks in, admits uncertainty) gets the best cards added, in rank order, until the next card would exceed `--budget` (1,500 characters by default). Cards that match nothing are never added. For a topic the knowledge base doesn't cover, the script says so and uses the base persona alone.
4. **Live conversation:** the persona is printed and copied to the clipboard, and the community MLX server starts, exactly as in example3.

Examples of what gets retrieved:

| Topic | Cards retrieved | Persona length |
|---|---|---|
| `pasta with tomato sauce` | tomato-sauce, pasta-finish, pasta-water | 1,047 |
| `roast chicken and vegetables` | chicken, roast-vegetables, meat-temps | 1,021 |
| `fluffy rice` | rice | 633 |
| `buttermilk pancakes but I have no buttermilk` | buttermilk, pancakes, baking-powder | 915 |
| `sushi` | none: the knowledge base has no sushi cards, so it uses the base persona | 395 |

## The pieces

### The flow

```
--topic "pasta with tomato sauce"
   → tokenize → BM25 ranking over 25 fact cards → best cards
   → base persona + cards (≤ 1,500 chars) → printed and copied
   → personaplex_mlx.local_web → browser at localhost:8998 (mic in, voice out)
```

### Files in this folder

| File | What it is |
|---|---|
| [`README.md`](README.md) | This explanation: why plain RAG doesn't fit, the alternatives, how to run and extend the example, and the live test |
| [`knowledge/cooking.txt`](knowledge/cooking.txt) | The knowledge base: 25 fact cards, one per paragraph |
| [`serve.py`](serve.py) | The program: retrieves cards, builds the persona, and starts the live server |

### The knowledge base: `knowledge/cooking.txt`

Each card is one paragraph:

```
[pasta-water] pasta, spaghetti, noodles, penne, boil, water, salt
Cook pasta in a big pot of water at a rolling boil, salted with about one tablespoon...
```

- **First line:** `[id]` followed by keywords. The keywords are what retrieval matches against most strongly.
- **Body:** exactly what Chef Marco is told when the card is retrieved. It's written as one or two spoken-style sentences, because the model follows its prompt as instructions rather than quoting it.
- **Food-safety cards** (chicken, meat temperatures, leftovers) use the USDA safe minimum internal temperatures; the source is cited at the top of the file.
- **Formatting:** blank lines separate cards, and lines starting with `#` are comments.

### The program: `serve.py`

In the order they're used:

| Piece | What it does |
|---|---|
| Settings | `MODEL_REPO` (NVIDIA's gated model), the voice (`NATM0`, labeled "NATURAL_M0" in the page's menu), `MIN_MEMORY_GB`, and the BM25 tuning numbers `BM25_K1`, `BM25_B`, and `KEYWORD_WEIGHT` |
| `STOPWORDS` | Filler words ignored in matching ("I", "want", "with", "but", …). Without them, "but I have no buttermilk" also matched the seasoning card, whose text contains "but". |
| `BASE_PERSONA` | Chef Marco without facts: patient, asks what you have, teaches one step at a time, checks in, admits uncertainty. Your topic is filled in, and it ends with "Use these facts:". |
| `tokenize()` | Turns text into lowercase words, dropping stopwords and simple plurals: "Boiled Eggs" → `boiled`, `egg` |
| `load_cards()` | Reads the knowledge file into a list of cards (id, keywords, text), and stops with an error naming any malformed card |
| `bm25_rank()` | Scores every card against your topic with [BM25](https://en.wikipedia.org/wiki/Okapi_BM25). See [How the BM25 score works](#how-the-bm25-score-works). Returns the cards that match at all, best first. |
| `build_persona()` | Adds the best cards to the base persona, in rank order, until the next card would exceed `--budget` (1,500 characters by default). A card that doesn't fit is skipped and smaller ones are still tried. Cards that matched nothing are never added, so "sushi" gets the base persona alone. |
| `main()` | Reads the options, retrieves the cards, prints their scores and the finished persona, and copies it with `pbcopy`. With `--dry-run` it stops there. Otherwise it runs the checks below and starts the MLX web server with the persona. |
| `check_mac()`, `sysctl()` | The same checks as example3: a Mac with Apple Silicon, the MLX port installed in this Python, and enough memory (warns below 16 GB) |
| `check_model_access()` | Downloads the model's small `config.json` to confirm your Hugging Face login works, and prints the login steps if access is denied |

### How the BM25 score works

For each word in your topic that appears in a card (its text, plus its keyword line counted `KEYWORD_WEIGHT` = 2 times), BM25 adds:

- **More for rare words.** "Buttermilk" appears in one card, so it points strongly to that card; "salt" appears in several, so it counts less (the IDF term).
- **More for repeated words, with diminishing returns.** A second mention helps, a tenth barely does (`BM25_K1` = 1.5 controls how fast this levels off).
- **Less for long cards.** A long card doesn't win just by containing more words (`BM25_B` = 0.75 controls how much length is penalized).

These are BM25's standard parameter values. It's pure Python, with no extra packages.

### Pieces outside this folder

| Piece | Role |
|---|---|
| [`../../personaplex-mlx/`](../../personaplex-mlx) | The community MLX port that runs the model on the Mac's GPU with 4-bit weights. `serve.py` starts its web server, `personaplex_mlx.local_web`. |
| [`../../environment-mlx.yml`](../../environment-mlx.yml) | The `personaplex-mlx` conda environment that everything runs in |
| [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1) on Hugging Face | NVIDIA's model weights, the voice prompts, and the web page itself. They download on first use and are cached in `~/.cache/huggingface`. |
| The browser page | Where you paste the persona into **Text Prompt**, pick **Voice** `NATURAL_M0`, click **Connect**, and talk. The page's own Text Prompt box takes precedence over `serve.py`'s `--text-prompt`, which is why you paste the persona. |
| [example3](../example3) | The live setup this example builds on. Its chef persona is the generic baseline in the comparison below. |

## Running

Set up the `personaplex-mlx` environment as in [example3](../example3/README.md#setup-once), then, from the root of this repository:

```bash
conda activate personaplex-mlx
```

See which cards a topic retrieves, without starting anything:

```bash
python examples/example4/serve.py --topic "pasta with tomato sauce" --dry-run
```

Talk to the chef:

```bash
python examples/example4/serve.py --topic "pasta with tomato sauce"
```

In the page that opens: paste the persona into **Text Prompt** (⌘V), pick **Voice** `NATURAL_M0`, click **Connect**, and allow the microphone. Use headphones.

| Option | Effect |
|---|---|
| `--topic TEXT` | What you want to cook (required) |
| `--budget N` | Maximum persona length in characters (default `1500`) |
| `--knowledge FILE` | Another fact card file, e.g. for a different subject |
| `--dry-run` | Only show the retrieved cards and the persona |
| `--no-copy` | Don't copy the persona to the clipboard |
| `--quant 4\|8\|none` | Weight quantization (default `4`) |
| `--port N`, `--no-browser` | As in example3 |

### Adding knowledge

Add a paragraph to `knowledge/cooking.txt`:

```
[risotto] risotto, arborio, rice, stock, creamy
Toast arborio rice in butter for a minute, then add warm stock a ladle at a time, stirring, until it's creamy and just tender, about eighteen minutes.
```

The first line is `[id]` followed by keywords. The text after it is what the chef will be told. Use `--knowledge` to point at a different file entirely, such as a SenNet or HuBMAP fact file for the data guides.

## Does it help? A live comparison on a MacBook Pro M4 Max

I asked the same spoken question, "How much salt should I put in the pasta water, and should I add oil?", to two personas on the live server, streamed in real time, with three seeds each. The **generic chef** is example3's persona, which says only "salt pasta water well". The **retrieved persona** is example4's `pasta with tomato sauce` persona, whose cards say "one tablespoon of kosher salt per four quarts… don't add oil".

| Seed | Generic chef (example3) | Retrieved pasta persona (example4) |
|---|---|---|
| 1 | "a couple of tablespoons of oil will do the trick" ✗ | Talked about the sauce, using card facts ("a cup of starchy pasta water… a can of tomatoes, some garlic… fresh basil"), instead of answering |
| 2 | "20 grams per litre… about 2 tablespoons" (not from its prompt) | "You don't need oil… bring it to a rolling boil… Stir in the first minute" ✓, but "a tablespoon per quart" ✗ (the card says per *four* quarts) |
| 3 | Heat oil in a pan (off topic) | "It'll help the sauce cling" (a card phrase), still about the sauce |

**What this shows:**
- **Retrieval changes what the chef knows and says.** Phrases from the retrieved cards showed up in all three runs, and the one direct answer got "no oil" right where the generic chef said to add oil.
- **Retrieval doesn't make it reliable.** It garbled a number (per quart instead of per four quarts), and in two of three runs it answered about the sauce instead of the question. A 7B speech model answering in real time paraphrases loosely.
- **Treat it as a better-informed conversation partner, not a reference.** For exact quantities and food safety, check a recipe or use a thermometer. For dependable expert answers, option 3 above is the way.

This is a small, informal test (one question, three seeds) on the 4-bit MLX model, not a benchmark.

## Things to know

- **The persona length is a tradeoff.** More cards means more facts, but a longer prompt, and possibly a less coherent chef. 1,500 characters is a starting point, not a tested optimum. Try `--budget 1000` or `--budget 2500` and compare.
- **Retrieval is keyword-based.** BM25 matches words, not meaning: "spaghetti" finds the pasta cards because "spaghetti" is in their keywords, but a word that's in no card finds nothing. For larger knowledge bases, embedding search (e.g. a small sentence-embedding model) would catch synonyms, at the cost of an extra dependency.
- **One topic per session.** To change topics, stop the server and start it again with a new `--topic` (option 2 above).
