# Example 1: Text prompts to audio on a MacBook Pro M4

PersonaPlex is speech-to-speech: it listens to audio and answers in audio. This example lets you write **your side of the conversation as text** on a Mac. The script speaks your lines with macOS `say`, feeds them to PersonaPlex running on the Mac's CPU, and saves what the agent says back.

```
prompts.txt ──► macOS say ──► input.wav ──► moshi.offline (CPU) ──► agent.wav + transcript.txt
```

## Files

| File | Purpose |
|---|---|
| [`prompts.txt`](prompts.txt) | The conversation: agent voice, role prompt, and your lines. The included scenario is a graduate student asking a [HuBMAP](https://portal.hubmapconsortium.org/) data guide how to find public datasets for computational biology, with 5-second pauses for the answers. |
| [`run_example.py`](run_example.py) | Builds the input audio, runs PersonaPlex, writes the results |

## Requirements

- **A MacBook Pro M4** (M4, M4 Pro or M4 Max) on macOS. The script stops on anything other than macOS on Apple Silicon, and warns, but continues, on non-M4 chips.
- **24 GB of unified memory or more.** The BF16 weights alone are ~15 GB. On a 16 GB Mac the script warns that it will likely swap heavily or run out of memory. 36–48 GB is comfortable.
- **About 20 GB of free disk space** for the model download, in the Hugging Face cache (`~/.cache/huggingface`).
- **The `personaplex` conda environment** from [`../environment.yml`](../environment.yml). It includes `libopus`, so there's no need for `brew install opus`.
- **`HF_TOKEN`** exported, with the [model license](https://huggingface.co/nvidia/personaplex-7b-v1) accepted.

## Setup

From the root of this repository:

```bash
git submodule update --init --recursive
```

```bash
conda env create -f environment.yml
```

```bash
conda activate personaplex
```

The model is gated on Hugging Face, so give it your account once:

1. Sign in at [huggingface.co](https://huggingface.co) and accept the license at [huggingface.co/nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1).
2. Create a **read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Log in. This saves the token, so later shells don't need `HF_TOKEN`:

   ```bash
   huggingface-cli login
   ```

   Or set it for the current shell only:

   ```bash
   export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
   ```

The script checks access before running the model. If the license isn't accepted or you aren't logged in, it stops with these steps instead of a `GatedRepoError` traceback.

## Writing prompts

Each line of `prompts.txt` is `key: value`. Lines starting with `#` are comments.

| Key | Meaning |
|---|---|
| `voice:` | Agent voice: `NATF0`–`NATF3`, `NATM0`–`NATM3`, `VARF0`–`VARF4`, `VARM0`–`VARM4` |
| `role:` | Role prompt. Use several `role:` lines for a long prompt; they're joined with spaces. |
| `pause:` | Seconds of silence after each following user turn, which is the agent's time to answer. You can change it between turns. |
| `user:` | A line you say, spoken with macOS `say` |
| `audio:` | A line you say, from a WAV file instead (path relative to `prompts.txt`) |

Example:

```
voice: NATF2
role: You enjoy having a good conversation. Have a casual conversation about favorite foods.
pause: 6
user: Hi! What's your favorite thing to cook?
user: That sounds great. Where did you learn to make it?
```

## Running

Check your prompts first. This only builds `input.wav`, takes seconds, and needs no model:

```bash
python example1/run_example.py --input-only
```

```bash
open example1/output/input.wav
```

Then run the model on the CPU:

```bash
python example1/run_example.py
```

Options:

| Option | Effect |
|---|---|
| `my_prompts.txt` (positional) | Use a different prompts file |
| `--device mps` | Try the Apple GPU. Untested upstream; it sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops fall back to the CPU. |
| `--say-voice NAME` | macOS voice for your lines, e.g. `Samantha`. List voices with `say -v '?'`. |
| `--gap SECONDS` | Silence between voices in `conversation.wav` (default `0.5`) |
| `--input-only` | Only build `input.wav`, without running the model |
| `--mix-only` | Remake `conversation.wav` and `conversation_live.wav` from an existing `agent.wav`, without rerunning the model. Use it to try a different `--gap`. |
| `--seed N` | Random seed (default `42424242`) |
| `--out-dir DIR` | Output folder (default `example1/output`) |

## Output

Everything goes to `example1/output/`, which git ignores:

| File | Contents |
|---|---|
| `input.wav` | Your lines as speech, with the pauses |
| `agent.wav` | **The agent's reply audio**, the same length as `input.wav` |
| `conversation.wav` | **The conversation one voice at a time:** each of your turns, 0.5 s of silence (`--gap`), the agent's full answer, 0.5 s, the next turn. Agent sounds made entirely while you were speaking ("okay", "right") are left out. |
| `conversation_live.wav` | Stereo, time-aligned, as the model actually heard and answered: you on the left, the agent on the right. The voices can overlap. |
| `transcript.txt` | What the agent said, as text |
| `agent.json` | The agent's raw text tokens from `moshi.offline` |

Listen with:

```bash
open example1/output/conversation.wav
```

## What to expect on an M4

- **It's slow.** The official code is built for NVIDIA GPUs, and the Mac runs it on the CPU. Processing takes much longer than the length of `input.wav`. This is offline batch processing, not a live conversation.
- **The first run downloads the model** from Hugging Face, about 17 GB. Later runs use the cache.
- **Voice prompts are copied for the CPU.** The published voice files (`NATF2.pt` and the rest) were saved from an NVIDIA GPU. Upstream loads them without moving them to the CPU, which crashes on a Mac with `Attempting to deserialize object on a CUDA device`. The script saves a CPU copy of your voice in `output/voices/` and passes it to `moshi.offline` with `--voice-prompt-dir`. The submodule is untouched.
- **The Apple GPU (`--device mps`) is experimental.** Upstream turned automatic MPS support off. It may fail with unsupported-operation errors, or run no faster than the CPU. If it errors, go back to the default CPU.
- **Shorter is faster.** Fewer turns and shorter `pause:` values mean less audio to process.
- **For live conversation, use an NVIDIA GPU.** See [PSC Bridges-2 GPUs](../README.md#psc-bridges-2-gpus).

## Tips

- **Too little time to answer?** PersonaPlex only talks while there is input audio. If the agent gets cut off, increase `pause:`.
- **The agent talks over you.** That's expected. It's full-duplex, so it can interrupt, run its answer into your next turn, or make "mm-hm" sounds while you speak, like a person would. `conversation.wav` takes the voices apart so they never overlap; `conversation_live.wav` keeps the real timing.
- **Different answers each run?** Change `--seed`. Keep it fixed to reproduce a result.
