# Example 3: Live conversation in a web page on a MacBook Pro M4

Open a page, allow the microphone, and talk to PersonaPlex while it talks back, full duplex, **entirely on your Mac**. There are two roles: **you're the student**, and the model plays **Chef Marco, a cooking teacher**. Data-guide personas for [SenNet](https://sennetconsortium.org/) and [HuBMAP](https://portal.hubmapconsortium.org/) are also included.

NVIDIA's own PyTorch code is too slow for live conversation on Apple Silicon (details below). This example instead runs [personaplex-mlx](https://github.com/mu-hashmi/personaplex-mlx), a **community MLX port** of PersonaPlex, included as the [`personaplex-mlx`](../personaplex-mlx) submodule. It quantizes NVIDIA's official weights to 4-bit and serves NVIDIA's own PersonaPlex web page.

```
browser (mic + speakers) ──WebSocket──► personaplex_mlx.local_web ──► PersonaPlex (MLX, 4-bit, Apple GPU)
   http://localhost:8998                   (started by example3/serve.py)
```

## Files

| File | Purpose |
|---|---|
| [`serve.py`](serve.py) | Checks your Mac and Hugging Face access, prints and copies the persona, starts the MLX web server |
| [`personas/chef.txt`](personas/chef.txt) | Default persona: Chef Marco, a patient cooking teacher (851 characters; the page allows 1,000) |
| [`personas/sennet.txt`](personas/sennet.txt) | SenNet data guide persona (833 characters) |
| [`personas/hubmap.txt`](personas/hubmap.txt) | HuBMAP data guide persona (852 characters) |
| [`../environment-mlx.yml`](../environment-mlx.yml) | The separate `personaplex-mlx` conda environment |

## Requirements

- **A Mac with Apple Silicon.** Tested on a MacBook Pro M4 Max with 48 GB. The 4-bit model needs much less memory than example1 and example2; `serve.py` warns below 16 GB.
- **The `personaplex-mlx` conda environment.** It's separate from `personaplex`, because the MLX port needs `sphn >= 0.2` and `numpy >= 2.1`, while NVIDIA's package pins `sphn < 0.2`.
- **A Hugging Face login with the [model license](https://huggingface.co/nvidia/personaplex-7b-v1) accepted.** It's the same model; the port reads NVIDIA's weights and voices.
- **Headphones.** Neither the page nor the server cancels echo, so through speakers the model hears itself.

## Setup (once)

From the root of this repository:

```bash
git submodule update --init --recursive
```

```bash
conda env create -f environment-mlx.yml
```

If you haven't logged in to Hugging Face in this environment yet:

```bash
conda activate personaplex-mlx && huggingface-cli login
```

## Running

```bash
conda activate personaplex-mlx
```

```bash
python example3/serve.py
```

`serve.py` prints the chef persona, copies it to the clipboard, loads the model and opens [http://localhost:8998](http://localhost:8998) in your browser. Then:

1. Paste the persona into **Text Prompt** (⌘V). The page starts with its own example text, which would otherwise be used.
2. Pick **Voice** `NATURAL_M0`. For the data guides, it's `NATURAL_F2` (SenNet) or `NATURAL_M1` (HuBMAP); `serve.py` prints the right one.
3. Click **Connect** and allow the microphone.
4. Talk. Tell Chef Marco what you'd like to cook, and he'll ask what you have and teach it step by step. The first couple of seconds are slower while the model warms up.

Press Ctrl+C in the terminal to stop the server.

| Option | Effect |
|---|---|
| `--persona sennet` / `--persona hubmap` | A SenNet or HuBMAP data guide instead of the chef |
| `--quant 8` | 8-bit weights: closer to the original model, slightly slower |
| `--quant none` | Full bf16 weights: too slow for live use on an M4 Max (see below) |
| `--port N` | Serve on another port (default `8998`) |
| `--no-browser` | Don't open the page automatically |

## Does it keep up? Measurements on a MacBook Pro M4 Max (48 GB)

Live conversation needs each 80 ms audio frame processed in under 80 ms.

**MLX port** (this example), offline timing over a 76.3 s recording, model already loaded:

| Weights | Time per 80 ms frame | Compared with real time |
|---|---|---|
| 4-bit (default) | 72 ms | 0.90×: keeps up |
| 8-bit | 75 ms | 0.94×: keeps up |

**Live server test.** I streamed 30 s of speech to the running server over its WebSocket, in real time and Opus-encoded, as the browser does:
- The persona loaded in 1.8 s.
- After a slow start (only 0.1 s of reply audio by the 5 s mark), the model caught up by 10 s.
- From then on it stayed 0.08 s (one frame) behind the input through the end. It returned 30.0 s of audio for 30.0 s sent, answering from the SenNet persona.
- The same test with the chef persona and a spoken question gave the same timing ("I want to make pasta with tomato sauce tonight… Where do I start?"). Marco answered in character: "Hi, this is Chef Marco. Great! What have you got? … start with boiling water, add a good pinch of salt…"

**NVIDIA's PyTorch code, for comparison** (why this example doesn't use it):

| Device | Time per 80 ms frame | Why |
|---|---|---|
| CPU | 136 ms (1.7× too slow) | ~85 ms in the main transformer, plus the depth transformer and the audio codec |
| Apple GPU (MPS) | 735 ms (9× too slow) | `aten::index_copy` isn't implemented on MPS in PyTorch 2.4, so the attention cache falls back to the CPU on every step |
| MPS, with `index_copy` replaced | 229 ms (2.9× too slow) | The depth transformer's many small operations cost 136 ms per frame on the GPU, and bf16/fp16 made no difference |

The margin is thin: about 10% at 4-bit. If other heavy apps are using the GPU or CPU, the reply audio may stutter, so close them while you talk.

## Things to know

- **The MLX port is community code, not NVIDIA's.** It's [mu-hashmi/personaplex-mlx](https://github.com/mu-hashmi/personaplex-mlx), MIT-licensed, pinned at commit `7b29cfc` (February 2026). Before adding it, I read its code: it downloads only from NVIDIA's Hugging Face repository, makes no other network calls, and runs no shell commands. Its outputs may differ slightly from NVIDIA's reference implementation, especially at 4-bit.
- **The server only listens on `localhost`,** so other devices on your network can't reach it. NVIDIA's own server listens on every interface.
- **The page's text prompt can't be longer than 1,000 characters,** which is why these personas are shorter than example2's guide prompt.
- **The facts come from the persona.** PersonaPlex can still get things wrong or make them up. For cooking, use a food thermometer: the chef persona gives safe temperatures for chicken (165 °F / 74 °C) and ground beef (160 °F / 71 °C), but the model may still misspeak. For the data guides, check anything important on the [SenNet](https://data.sennetconsortium.org/) or [HuBMAP](https://portal.hubmapconsortium.org/) portal.
