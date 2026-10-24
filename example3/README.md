# Example 3: Live conversation in a web page on a MacBook Pro M4

Open a page, allow the microphone, and talk to PersonaPlex while it talks back, full duplex, **entirely on your Mac**. The model plays a [SenNet](https://sennetconsortium.org/) or [HuBMAP](https://portal.hubmapconsortium.org/) data guide.

NVIDIA's own PyTorch code is too slow for live conversation on Apple Silicon (details below). This example instead runs [personaplex-mlx](https://github.com/mu-hashmi/personaplex-mlx), a **community MLX port** of PersonaPlex, included as the [`personaplex-mlx`](../personaplex-mlx) submodule. It quantizes NVIDIA's official weights to 4-bit and serves NVIDIA's own PersonaPlex web page.

```
browser (mic + speakers) ──WebSocket──► personaplex_mlx.local_web ──► PersonaPlex (MLX, 4-bit, Apple GPU)
   http://localhost:8998                   (started by example3/serve.py)
```

## Files

| File | Purpose |
|---|---|
| [`serve.py`](serve.py) | Checks your Mac and Hugging Face access, prints and copies the persona, starts the MLX web server |
| [`personas/sennet.txt`](personas/sennet.txt) | SenNet data guide persona (833 characters; the page allows 1,000) |
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

`serve.py` prints the SenNet persona, copies it to the clipboard, loads the model and opens [http://localhost:8998](http://localhost:8998) in your browser. Then:

1. Paste the persona into **Text Prompt** (⌘V). The page starts with its own example text, which would otherwise be used.
2. Pick **Voice** `NATURAL_F2` (`NATURAL_M1` for HuBMAP).
3. Click **Connect** and allow the microphone.
4. Talk. The first couple of seconds are slower while the model warms up.

Press Ctrl+C in the terminal to stop the server.

| Option | Effect |
|---|---|
| `--persona hubmap` | HuBMAP guide instead of SenNet |
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
- **The facts come from the persona.** PersonaPlex can still get things wrong or make them up, so check anything important on the [SenNet](https://data.sennetconsortium.org/) or [HuBMAP](https://portal.hubmapconsortium.org/) portal.
