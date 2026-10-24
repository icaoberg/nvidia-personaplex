# Example 4: Giving the guide more expertise with retrieval (RAG)

Can retrieval-augmented generation (RAG) make PersonaPlex's guide more of an expert? This example answers that question for **high-performance computing**. There are two roles:
- **Penelope,** a student learning to use a cluster.
- **Ivan,** an HPC engineer, played by PersonaPlex.

Ivan's expertise comes from a knowledge base summarized from three sources:
- the [Slurm](https://slurm.schedmd.com/documentation.html) documentation, for running jobs
- the [Lmod](https://lmod.readthedocs.io/en/latest/) documentation, for loading software
- the [Bridges-2 User Guide](https://www.psc.edu/resources/bridges-2/user-guide/), for the specifics of PSC's Bridges-2 supercomputer

The example builds the approach that fits PersonaPlex best: **retrieve before the conversation, then prompt**. Penelope says what they need help with. The script picks the most relevant facts, builds Ivan's persona from them, and starts the live web page from [example3](../example3), running on your Mac.

```
--topic "submitting a GPU batch job"
        │
        ▼
knowledge/slurm.txt + lmod.txt + bridges2.txt ──BM25──► best fact cards ──► Ivan's persona (≤ 1,500 chars)
             (116 fact cards)                                                        │
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
| **1. Retrieve before connecting** (this example) | Keep a knowledge base of short fact cards. The student picks a topic, the closest cards are retrieved, and a focused persona is built and sent, within a length budget. | Moderate: the right facts for this session, not all facts | Small: one script plus card files; no model changes |
| **2. Re-prompt between sessions** | Same retrieval, but reconnect with a new prompt when the topic changes. Each connection reloads the persona in about 2 s on MLX. | Moderate, across topics | Small, but the conversation restarts |
| **3. PersonaPlex as the voice, an LLM as the brain** | Transcribe the student's speech (e.g. Whisper on MLX), answer with an LLM over a full RAG index of the documentation, then force the answer into PersonaPlex's text stream as example2 does. | High: real RAG over all the documentation | Large: you lose most of the natural full-duplex interplay and add latency |
| **4. Fine-tuning** | Train on domain conversations. | High within the domain | Very large: needs GPUs and conversation data. I haven't checked whether NVIDIA released training code. |

**Recommendation:** start with option 1. It fits how PersonaPlex is meant to be used and runs live on a Mac, and the right 800–1,500 characters of facts for *this student's problem* beat a generic persona. If you need real depth, such as answering from every Slurm and Lmod page, that requires option 3. At that point PersonaPlex is mainly the voice.

## How this example works

1. **Knowledge base:** every `.txt` file in [`knowledge/`](knowledge) is loaded and searched together:
   - [`slurm.txt`](knowledge/slurm.txt): 32 cards from the Slurm documentation (version 26.05). They cover:
     - concepts and architecture
     - `sbatch` and its directives; time, CPU, memory and GPU requests; partitions and accounts
     - output files, the working directory, and environment variables
     - `srun`, MPI, and interactive sessions with `salloc` and `srun --pty`
     - `squeue`, pending reasons, job states, `scancel`, `sinfo`, `sacct`, `scontrol`, `sstat`
     - job arrays, dependencies, constraints, email, and `sbcast`
   - [`lmod.txt`](knowledge/lmod.txt): 23 cards from the Lmod documentation (version 9.4.5). They cover:
     - what modules are; `module avail`, `list`, `load`, `unload`, `swap`, `purge` and `reset`
     - default versions; `module spider` and how it shows what to load first
     - the compiler/MPI software hierarchy, the one-version and family rules
     - `module show`, `help`, `whatis` and `keyword`; the `ml` shortcut
     - saving and restoring collections, personal modulefiles with `module use`, the `avail` legend, and startup files
   - [`bridges2.txt`](knowledge/bridges2.txt): 61 cards about PSC's Bridges-2: from its User Guide (read October 2026), `sinfo` output from the cluster, the ACCESS resource listing, and NVIDIA's GPU specifications. These are site-specific, about one cluster. They cover:
     - the hardware; logging in with SSH or OnDemand; login nodes versus compute nodes
     - file spaces (`$HOME`, `$PROJECT` on Ocean, `$LOCAL`, `$RAMDISK`), quotas, and file retention
     - file transfers through the data transfer node or Globus
     - allocations, `projects`, service-unit charging, and `-A` / `change_primary_group`
     - the partitions in detail: each one (RM, RM-shared, RM-512, EM, GPU, GPU-shared) with its limits, which to choose, their names, default and maximum times, how many nodes of each kind there are, example commands with their SU cost, and the GPU node types, including the DGX-2
     - `interact`, PSC's `slurm-tool`, Bridges-2 modules, community datasets, getting help, and acknowledgments
     - the GPU nodes, from `sinfo` on 2026-10-24: each type's node names, count, CPUs, memory and Slurm features; which GPU to choose; CPUs per GPU; and how to check the nodes yourself
     - the GPU hardware, from NVIDIA's specifications: H100 SXM5, L40S and V100 core counts, memory and bandwidth, peak FLOPS, BF16/FP8 support (V100 has neither), and the DGX-2's NVSwitch fabric
     - GPU partitions the user guide doesn't describe, found with `sinfo` across all partitions (GPU-small, GPU-dev with an A100, ROBO and ROBO-8 with more H100s, HACC with AMD MI210s), which are probably reserved for specific projects, plus A100 and MI210 specs

   Each card is one or two spoken-style sentences, because the model follows its prompt as instructions and says things aloud. So commands are spelled the way an engineer would say them: "dash dash gres equals gpu colon two".
2. **Retrieval:** [`serve.py`](serve.py) ranks the cards against the topic with [BM25](https://en.wikipedia.org/wiki/Okapi_BM25), a standard keyword-ranking formula, in pure Python with no extra packages. Each card's keyword line counts double. Filler words ("I", "want", "my", "why") are ignored. Letters and digits stay together, so GPU names like "H100" and "A100" match as words. Simple plurals and "-ing" forms are matched: "dependencies" ↔ "dependency", "loading" ↔ "load".
3. **Persona:** a short base persona gets the best cards added in rank order. In it, Ivan, an HPC engineer on a cluster with Slurm and Lmod, talks with Penelope. Ivan asks what Penelope is running, explains one step at a time, gives exact commands, checks in, and admits uncertainty. Adding stops when the next card would exceed `--budget` (1,500 characters by default), or when the remaining cards score below 40% of the best one, which usually means they only matched a common word like "job" or "module". For a topic the knowledge base doesn't cover, the script says so and uses the base persona alone.
4. **Live conversation:** the persona is printed and copied to the clipboard, and the community MLX server starts, exactly as in example3.

Examples of what gets retrieved:

| Topic | Cards retrieved | Persona length |
|---|---|---|
| `submitting a GPU batch job` | sbatch, gpus, b2-login-nodes, b2-partition-chooser | 1,417 |
| `why is my job pending` | squeue, pending-reasons | 1,102 |
| `job arrays and dependencies` | job-arrays, dependencies, scancel, array-limits | 1,482 |
| `my job ran out of memory` | memory, output-files, working-directory, sstat, b2-partition-rm | 1,451 |
| `interactive session on a compute node` | salloc, srun-pty, b2-interact, b2-login-nodes | 1,432 |
| `loading a compiler module` | hierarchy, b2-modules, spider-prerequisites, module-swap | 1,439 |
| `module not found` | module-spider, module-use, module-list, module-unload, lmod-overview | 1,397 |
| `saving my modules so they load every time` | startup-files, collections-restore, collections-save, module-load, module-reset | 1,479 |
| `loading Python and submitting a batch job` | sbatch, b2-login-nodes, module-avail, gpus (all three files) | 1,316 |
| `submitting a GPU job on Bridges-2` | sbatch, b2-partition-chooser, b2-login-nodes, b2-ondemand, b2-benchmark | 1,463 |
| `an interactive session on Bridges-2` | b2-interact, salloc, srun-pty, b2-interact-gpu | 1,483 |
| `transferring files to Bridges-2` | b2-transfer, b2-globus, b2-retention, sbcast | 1,150 |
| `where should I store large files on Bridges-2` | b2-project, b2-globus | 999 |
| `how many SUs do I have left` | b2-projects-command | 721 |
| `which partition should I use on Bridges-2` | b2-partition-chooser, b2-gpu-hw-compare, b2-partition-names, b2-partition-rm | 1,431 |
| `what is the maximum time limit on Bridges-2` | b2-partition-time-limits, time-limit, b2-partition-rm, b2-partition-rm-shared | 1,375 |
| `running a job that needs a lot of memory on Bridges-2` | b2-gpu-chooser, b2-partition-em, b2-gpu-cpus, sstat | 1,465 |
| `how much does a GPU job cost` | b2-gpu-partition-example, b2-gpu-chooser, b2-gpu-shared-example | 1,407 |
| `which GPU should I use on Bridges-2` | b2-gpu-hw-compare, b2-gpu-chooser, b2-gpu-types | 1,452 |
| `how fast are the H100s` | b2-gpu-hw-h100-speed, b2-gpu-h100 | 1,070 |
| `does the V100 support bf16` | b2-gpu-hw-v100-precision, b2-gpu-hw-h100, b2-gpu-hw-l40s-speed | 1,273 |
| `using the DGX-2` | b2-gpu-hw-dgx2, b2-gpu-dgx2, b2-gpu-nodes | 1,464 |
| `is there an A100` | b2-gpu-hw-a100, b2-other-gpu-partitions | 1,255 |
| `AMD GPUs on Bridges-2` | b2-gpu-hw-mi210, b2-other-gpu-partitions | 1,180 |
| `kubernetes` | none: the knowledge base is about Slurm, Lmod and Bridges-2, so it uses the base persona | 588 |

Cards whose ids start with `b2-` come from `bridges2.txt`. Because all files are searched by default, general topics pick up Bridges-2 specifics too: "submitting a GPU batch job" gets Slurm's general GPU card *and* the Bridges-2 partition chooser. That's what a Bridges-2 student wants. For any other cluster, search only the general files: `--knowledge examples/example4/knowledge/slurm.txt examples/example4/knowledge/lmod.txt`.

## The pieces

### The flow

```
--topic "submitting a GPU batch job"
   → tokenize → BM25 ranking over 116 fact cards (all knowledge/*.txt) → best cards
   → base persona + cards (≤ 1,500 chars) → printed and copied
   → personaplex_mlx.local_web → browser at localhost:8998 (mic in, voice out)
```

### Files in this folder

| File | What it is |
|---|---|
| [`README.md`](README.md) | This explanation: why plain RAG doesn't fit, the alternatives, how to run and extend the example, and the live test |
| [`knowledge/slurm.txt`](knowledge/slurm.txt) | 32 Slurm fact cards, one per paragraph |
| [`knowledge/lmod.txt`](knowledge/lmod.txt) | 23 Lmod fact cards, one per paragraph |
| [`knowledge/bridges2.txt`](knowledge/bridges2.txt) | 61 Bridges-2 fact cards, one per paragraph, specific to PSC's cluster |
| [`serve.py`](serve.py) | The program: retrieves cards, builds the persona, and starts the live server |

### The knowledge files: `knowledge/*.txt`

Each card is one paragraph:

```
[gpus] gpu, gpus, gres, cuda, accelerator, gpu job
Request GPUs with dash dash gres equals gpu colon two for two GPUs per node, or with dash dash gpus...

[module-spider] module spider, spider, search, find software, not found, hidden
module spider searches every module on the system, including ones that module avail doesn't show yet...
```

- **First line:** `[id]` followed by keywords. The keywords are what retrieval matches against most strongly. Ids must be unique across all the files; `serve.py` stops with an error if two cards share one.
- **Body:** exactly what Ivan is told when the card is retrieved. It's written as one or two spoken-style sentences, with commands spelled out as they're said aloud.
- **Site-specific settings:** partition names, accounts, GPU types, which modules exist, default versions, and the letters in `module avail` all differ from cluster to cluster. In `slurm.txt` and `lmod.txt`, those cards tell the student to check their site's settings (`sinfo`, the `module avail` legend, the site documentation) instead of guessing. `bridges2.txt` *is* site documentation: its cards name Bridges-2 in every sentence, so the engineer doesn't present PSC's settings as universal.
- **Source:** the top of each file cites its documentation, version, and the pages used. The facts are summarized in our own words, not copied.
- **Formatting:** blank lines separate cards, and lines starting with `#` are comments.

### The program: `serve.py`

In the order they're used:

| Piece | What it does |
|---|---|
| Settings | `MODEL_REPO` (NVIDIA's gated model), the voice (`NATF2`, labeled "NATURAL_F2" in the page's menu), `MIN_MEMORY_GB`, the BM25 tuning numbers `BM25_K1`, `BM25_B` and `KEYWORD_WEIGHT`, and `MIN_RELATIVE_SCORE` (the 40% cutoff) |
| `STOPWORDS` | Filler words ignored in matching ("I", "want", "my", "why", "run", …). Without "run" and "running", "running MPI across nodes" also pulled in the `sstat` and `squeue` cards, which mention running jobs. |
| `BASE_PERSONA` | Ivan without facts: an HPC engineer on a cluster with Slurm and Lmod, talking with Penelope. The persona asks what Penelope is running, explains one step at a time, gives exact commands, checks in, and points to the documentation or support team when unsure. The topic is filled in, and it ends with "Use these facts:". |
| `singular()`, `tokenize()` | Turn text into lowercase words, dropping stopwords, bare numbers, plurals and "-ing": "Loading Job Arrays on the H100s" → `load`, `job`, `array`, `h100`. Letters and digits stay together, so GPU names (`h100`, `a100`, `l40s`, `mi210`) match as whole words; `NOT_PLURAL` keeps "l40s" from losing its "s". Forms the rule gets wrong ("saving" → "sav") are covered by listing them in a card's keywords. |
| `load_cards()` | Reads one knowledge file into a list of cards (id, keywords, text), and stops with an error naming any malformed card |
| `bm25_rank()` | Scores every card against the topic with BM25. See [How the BM25 score works](#how-the-bm25-score-works). Returns the cards that match at all, best first. |
| `build_persona()` | Adds the best cards to the base persona in rank order, within `--budget`. It stops at cards below 40% of the best score. A card that doesn't fit is skipped and smaller ones are still tried. |
| `main()` | Reads the options and loads every `--knowledge` file (all of `knowledge/*.txt` by default), checking that card ids are unique. Then it retrieves the cards, prints their scores and the finished persona, and copies it with `pbcopy`. With `--dry-run` it stops there. Otherwise it runs the checks below and starts the MLX web server with the persona. |
| `check_mac()`, `sysctl()` | The same checks as example3: a Mac with Apple Silicon, the MLX port installed in this Python, and enough memory (warns below 16 GB) |
| `check_model_access()` | Downloads the model's small `config.json` to confirm your Hugging Face login works, and prints the login steps if access is denied |

### How the BM25 score works

For each word in the topic that appears in a card (its text, plus its keyword line counted `KEYWORD_WEIGHT` = 2 times), BM25 adds:

- **More for rare words.** "Spider" appears in two cards, so it points strongly to them. "Job" and "module" appear in many cards, so they count for little (the IDF term).
- **More for repeated words, with diminishing returns.** A second mention helps, a tenth barely does (`BM25_K1` = 1.5 controls how fast this levels off).
- **Less for long cards.** A long card doesn't win just by containing more words (`BM25_B` = 0.75 controls how much length is penalized).

These are BM25's standard parameter values. It's pure Python, with no extra packages.

### Pieces outside this folder

| Piece | Role |
|---|---|
| [`../../personaplex-mlx/`](../../personaplex-mlx) | The community MLX port that runs the model on the Mac's GPU with 4-bit weights. `serve.py` starts its web server, `personaplex_mlx.local_web`. |
| [`../../environment-mlx.yml`](../../environment-mlx.yml) | The `personaplex-mlx` conda environment that everything runs in |
| [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1) on Hugging Face | NVIDIA's model weights, the voice prompts, and the web page itself. They download on first use and are cached in `~/.cache/huggingface`. |
| The browser page | Where the student pastes the persona into **Text Prompt**, picks **Voice** `NATURAL_F2`, clicks **Connect**, and talks. The page's own Text Prompt box takes precedence over `serve.py`'s `--text-prompt`, which is why you paste the persona. |
| [example3](../example3) | The live setup this example builds on |
| [Slurm documentation](https://slurm.schedmd.com/documentation.html), [Lmod documentation](https://lmod.readthedocs.io/en/latest/), [Bridges-2 User Guide](https://www.psc.edu/resources/bridges-2/user-guide/), [ACCESS resource listing](https://cider.access-ci.org/public/resources/RDR_000646), NVIDIA's [H100](https://www.nvidia.com/en-us/data-center/h100/) and [L40S](https://www.nvidia.com/en-us/data-center/l40s/) specifications and V100/DGX-2 datasheets, and `sinfo` on Bridges-2 | The sources of the knowledge base; `bridges2.txt` cites each one at the top |

## Running

Set up the `personaplex-mlx` environment as in [example3](../example3/README.md#setup-once), then, from the root of this repository:

```bash
conda activate personaplex-mlx
```

See which cards a topic retrieves, without starting anything:

```bash
python examples/example4/serve.py --topic "submitting a GPU batch job" --dry-run
```

Talk to Ivan:

```bash
python examples/example4/serve.py --topic "submitting a GPU batch job"
```

In the page that opens: paste the persona into **Text Prompt** (⌘V), pick **Voice** `NATURAL_F2`, click **Connect**, and allow the microphone. Use headphones.

| Option | Effect |
|---|---|
| `--topic TEXT` | What the student needs help with (required). Name everything you'll ask about, e.g. "a GPU batch job with a time limit". |
| `--budget N` | Maximum persona length in characters (default `1500`) |
| `--knowledge FILE [FILE …]` | Which fact card files to search (default: every `.txt` file in `knowledge/`), e.g. `--knowledge examples/example4/knowledge/lmod.txt` for Lmod only |
| `--dry-run` | Only show the retrieved cards and the persona |
| `--no-copy` | Don't copy the persona to the clipboard |
| `--quant 4\|8\|none` | Weight quantization (default `4`) |
| `--port N`, `--no-browser` | As in example3 |

### Adding knowledge

Add a paragraph to one of the files in `knowledge/`, or create a new `.txt` file there; it's picked up automatically. Facts about your own cluster are a good addition, since the general Slurm and Lmod cards can't know them:

```
[our-gpu-partition] gpu, partition, a100, our cluster
On our cluster, GPU jobs go to the partition called gpu, and you request an A100 with dash dash gres equals gpu colon a100 colon one.
```

The first line is `[id]` followed by keywords; the id must not be used by any other card. The text after it is what Ivan will be told.

## Does it help? A live comparison on a MacBook Pro M4 Max

I asked the same spoken question, "How do I ask for two GPUs and four hours in my batch script?", on the live server, streamed in real time, with three seeds each, to two personas:
- **Base persona:** Ivan without facts (587 characters).
- **Retrieved persona:** the persona for the topic `submitting a GPU batch job` (1,176 characters, with the sbatch, working-directory and gpus cards). This test ran before `bridges2.txt` was added; with it, the same topic now retrieves the Bridges-2 GPU cards instead (see the table above).

| Seed | Base persona (no facts) | Retrieved persona |
|---|---|---|
| 1 | "the command to request those resources is called slurm submit… the slurm flag -M and the number of GPUs" ✗ (there's no `slurm submit` command, and `-M` isn't a GPU option) | "run sbatch followed by your script name. That puts it in the queue… You can add a dash dash chdir if you want it in a different directory." Correct but off the GPU question, with one garbled step ("lmod your script name") |
| 2 | "What command have you tried so far?" (no command) | "you ask for dash dash gres equals gpu colon two. That gives you 2 gpus per node" ✓ |
| 3 | "slurm submit option G equals 2 and time equals 4" ✗: `-G 2` is a real GPU option, but there's no `slurm submit`, and a time of 4 means 4 minutes, not 4 hours | "dash dash gres equals gpu colon two for 2 GPUs per node, or dash dash gpus… name the GPU type like gpu colon type colon count, but that varies by cluster, so check your site's docs" ✓ |

**What this shows:**
- **Without facts, the engineer invents commands.** Knowing only that the cluster uses Slurm and Lmod, the base persona made up a `slurm submit` command in two of three runs.
- **With facts, it gives the real ones,** but not every time. The retrieved persona gave the correct `--gres=gpu:2` in two of three runs. In the third, it explained `sbatch` and `--chdir` correctly but skipped the GPU part and mixed up one step.
- **It only knows what was retrieved.** No run answered the "four hours" half correctly, because the topic didn't mention time, so the time-limit card wasn't retrieved. Name everything you'll ask about in `--topic`, e.g. "a GPU batch job with a time limit".
- **Treat it as a better-informed conversation partner, not a reference.** Site-specific settings (partitions, accounts, GPU types, module names) come from your cluster's documentation. For dependable answers across all the documentation, option 3 above is the way.

This is a small, informal test (one question, three seeds) on the 4-bit MLX model, not a benchmark. An earlier run of the same test, with a Slurm-only persona that didn't mention Lmod, gave `--gres=gpu:2` in all three retrieved runs, and no command in any base run.

## Things to know

- **The persona length is a tradeoff.** More cards means more facts, but a longer prompt, and possibly a less coherent engineer. 1,500 characters is a starting point, not a tested optimum. Try `--budget 1000` or `--budget 2500` and compare.
- **Retrieval is keyword-based.** BM25 matches words, not meaning: "GPU" finds the GPU card because "gpu" is in its keywords, but a word that's in no card finds nothing. For larger knowledge bases, embedding search (e.g. a small sentence-embedding model) would catch synonyms, at the cost of an extra dependency.
- **One topic per session.** To change topics, stop the server and start it again with a new `--topic` (option 2 above).
- **The knowledge base reflects Slurm 26.05, Lmod 9.4.5, and the Bridges-2 User Guide as of October 2026.** Commands rarely change, but Bridges-2 partitions, limits and SU rates do; PSC was also preparing to move Bridges-2 to a new datacenter. Check the current guide for anything that matters.
