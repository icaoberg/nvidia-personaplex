# Example 3: Live conversation in a web page (microphone in, voice out)

PersonaPlex ships a web UI: you open a page, allow the microphone, and talk to the model while it talks back, full duplex. This example starts that UI (upstream's `moshi.server`) with a ready-made [SenNet](https://sennetconsortium.org/) or [HuBMAP](https://portal.hubmapconsortium.org/) data-guide persona.

**Short answer: yes, it works in a browser with your microphone, but it needs an NVIDIA GPU to keep up in real time.** On a MacBook Pro M4 the page runs, but the model falls behind (measurements below). For a real conversation, run the server on a GPU, for example a Bridges-2 node, and open it in your laptop's browser through an SSH tunnel.

```
browser (mic + speakers) ──WebSocket──► moshi.server ──► PersonaPlex on GPU/CPU
   http://localhost:8998                  (this repo: example3/serve.py)
```

## Files

| File | Purpose |
|---|---|
| [`serve.py`](serve.py) | Starts `moshi.server` with the checks and fixes this repo needs, and prints the persona to paste |
| [`personas/sennet.txt`](personas/sennet.txt) | SenNet data guide persona (833 characters; the page allows 1,000) |
| [`personas/hubmap.txt`](personas/hubmap.txt) | HuBMAP data guide persona (852 characters) |
| [`bridges2.sbatch`](bridges2.sbatch) | Slurm job that runs the live server on one Bridges-2 L40S GPU |

`serve.py` adds the following to `python -m moshi.server`:
- **Checks Hugging Face access** before loading the 17 GB model, and prints the login steps if it can't.
- **Picks the device:** CUDA if available, otherwise CPU. It warns when the device can't keep up.
- **On CPU or MPS, makes CPU copies of all 18 voices** and passes them with `--voice-prompt-dir`. The published voices only load with CUDA (see the [main README](../README.md#apple-macbook-pro-m4)).
- **Prints the persona** and the voice to pick, and copies the persona to the clipboard on macOS.
- **On Linux,** prints the SSH tunnel command for a cluster node.

## Can it run live on a MacBook Pro M4?

Not in real time. The model has to process each 80 ms audio frame in under 80 ms. Measured on a MacBook Pro M4 Max (48 GB) with this repo's environment (PyTorch 2.4.1):

| Device | Time per 80 ms frame | Compared with real time |
|---|---|---|
| CPU | 136 ms | 1.7× too slow |
| Apple GPU (MPS) | 735 ms | 9× too slow: `aten::index_copy` isn't implemented on MPS in PyTorch 2.4, so it falls back to the CPU on every step |
| NVIDIA A100 / H100 / L40S | not measured here | NVIDIA's supported hardware for live use |

On the Mac, the page loads, the persona and voice load (about 30 s after you click Connect), and the model answers, but its replies lag and stutter. That's fine for seeing how the UI works, but not for a conversation.

## Running it on your Mac (to try the UI)

From the root of this repository:

```bash
conda activate personaplex
```

```bash
python example3/serve.py
```

When the log says `Access the Web UI`, open [http://localhost:8998](http://localhost:8998). Then:

1. Paste the persona into **Text Prompt**. `serve.py` copies it to the clipboard on macOS, so just press ⌘V. Use `--persona hubmap` for the HuBMAP guide.
2. Pick **Voice** `NATURAL_F2` (`NATURAL_M1` for HuBMAP).
3. Click **Connect** and allow the microphone.
4. Wait for the model to load the persona, about 30 s on the CPU, then talk.

**Use headphones.** Through speakers, the model hears its own voice and reacts to it. Press Ctrl+C in the terminal to stop the server.

## Running it live on Bridges-2

**Setup (once).** On Bridges-2, clone this repository, create the environment and log in to Hugging Face:

```bash
git clone --recurse-submodules https://github.com/icaoberg/nvidia-personaplex.git
```

```bash
cd nvidia-personaplex && conda env create -f environment.yml
```

```bash
conda activate personaplex && huggingface-cli login
```

**Each session:**

1. From the repository root on Bridges-2, submit the job, which asks for one L40S for one hour:
   ```bash
   sbatch example3/bridges2.sbatch
   ```
2. When it starts, the top of `personaplex-live-<jobid>.log` shows the compute node and the tunnel command. On your laptop, run that command; it looks like:
   ```bash
   ssh -L 8998:<compute-node>:8998 <user>@bridges2.psc.edu
   ```
3. Open [http://localhost:8998](http://localhost:8998) on your laptop, and continue from step 1 of the Mac instructions. The persona is printed in the job log (there's no clipboard on the cluster).
4. When you're done, end the job so it stops using SUs:
   ```bash
   scancel <jobid>
   ```

**Notes:**
- To use an H100 instead, change `--gpus=l40s-48:1` to `--gpus=h100-80:1`. It costs 2 SUs per GPU-hour instead of 1; see [PSC Bridges-2 GPUs](../README.md#psc-bridges-2-gpus). Don't use V100s: they lack native BF16 and are too slow for live use.
- **The batch script is untested.** It follows PSC's documented partitions, GPU types and Slurm options, but I haven't run it on Bridges-2, and your account may need `-A <allocation>` or a different conda module.

## Things to know

- **The page is served over plain HTTP on localhost.** Browsers allow the microphone on `http://localhost`, including through an SSH tunnel, so no certificates are needed. Upstream's `--ssl` option only creates certificates if `mkcert` is installed.
- **Anyone on the same network can reach the server.** Upstream listens on every network interface, whatever `--host` says, and the startup log shows your LAN address. Stop it when you're done, especially on shared Wi-Fi or a shared cluster node.
- **The text prompt can't be longer than 1,000 characters,** which is why these personas are shorter than example2's guide prompt.
- **The facts come from the persona.** PersonaPlex can still get things wrong or make them up, so check anything important on the [SenNet](https://data.sennetconsortium.org/) or [HuBMAP](https://portal.hubmapconsortium.org/) portal.
