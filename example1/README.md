# Example 1: Text prompts to audio

PersonaPlex is speech-to-speech: it listens to audio and answers in audio. This example lets you write **your side of the conversation as text**. The script speaks your lines with text-to-speech, feeds them to PersonaPlex, and saves what the agent says back.

```
prompts.txt ──► text-to-speech ──► input.wav ──► moshi.offline ──► agent.wav + transcript.txt
```

## Files

| File | Purpose |
|---|---|
| [`prompts.txt`](prompts.txt) | The conversation: agent voice, role prompt, and your lines |
| [`run_example.py`](run_example.py) | Builds the input audio, runs PersonaPlex, writes the results |

## Requirements

- The `personaplex` conda environment from [`../environment.yml`](../environment.yml).
- `HF_TOKEN` exported, with the [model license](https://huggingface.co/nvidia/personaplex-7b-v1) accepted.
- A text-to-speech command:
  - **macOS:** `say` (built in)
  - **Linux:** `espeak-ng` (`sudo apt install espeak-ng` or `sudo dnf install espeak-ng`)
  - **Neither:** use `audio:` lines with your own recordings instead.
- A GPU. See [Hardware and GPU compatibility](../README.md#hardware-and-gpu-compatibility). On a Mac or without a GPU, add `--device cpu`, which is slow.

## Writing prompts

Each line of `prompts.txt` is `key: value`. Lines starting with `#` are comments.

| Key | Meaning |
|---|---|
| `voice:` | Agent voice: `NATF0`–`NATF3`, `NATM0`–`NATM3`, `VARF0`–`VARF4`, `VARM0`–`VARM4` |
| `role:` | Role prompt. Use several `role:` lines for a long prompt; they're joined with spaces. |
| `pause:` | Seconds of silence after each following user turn, which is the agent's time to answer. You can change it between turns. |
| `user:` | A line you say, spoken with text-to-speech |
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

From the root of this repository:

```bash
conda activate personaplex
```

```bash
python example1/run_example.py
```

Useful options:

| Option | Effect |
|---|---|
| `my_prompts.txt` (positional) | Use a different prompts file |
| `--device cpu` | Run without an NVIDIA GPU (slow) |
| `--cpu-offload` | Offload model layers to CPU on small GPUs |
| `--input-only` | Only build `input.wav`, to check your prompts without running the model |
| `--tts espeak-ng` | Force a text-to-speech command |
| `--seed N` | Random seed (default `42424242`) |
| `--out-dir DIR` | Output folder (default `example1/output`) |

## Output

Everything goes to `example1/output/`, which git ignores:

| File | Contents |
|---|---|
| `input.wav` | Your lines as speech, with the pauses |
| `agent.wav` | **The agent's reply audio**, the same length as `input.wav` |
| `conversation.wav` | Stereo mix to listen to both sides: you on the left, the agent on the right |
| `transcript.txt` | What the agent said, as text |
| `agent.json` | The agent's raw text tokens from `moshi.offline` |

## Tips

- **Too little time to answer?** PersonaPlex only talks while there is input audio. If the agent gets cut off, increase `pause:`.
- **The agent talks over you.** That's expected. It's full-duplex, so it can interrupt or make "mm-hm" sounds while you speak, like a person would.
- **Different answers each run?** Change `--seed`. Keep it fixed to reproduce a result.
