#!/bin/bash
# Run PersonaPlex's live web UI on one Bridges-2 GPU.
#
# Submit from the root of this repository:
#   sbatch example3/bridges2.sbatch
# Then follow the tunnel instructions printed at the top of the job log
# (personaplex-live-<jobid>.log).
#
# One-time setup on Bridges-2, before the first job:
#   git clone --recurse-submodules https://github.com/icaoberg/nvidia-personaplex.git
#   cd nvidia-personaplex && conda env create -f environment.yml
#   conda activate personaplex && huggingface-cli login
#
# GPU choice (see the main README): l40s-48 (1 SU/GPU-hour) or h100-80 (2 SUs).
# V100s lack native BF16 and are too slow for live conversation.

#SBATCH --job-name=personaplex-live
#SBATCH --partition=GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH --time=01:00:00
#SBATCH --output=personaplex-live-%j.log

set -euo pipefail

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate personaplex

PORT=8998
echo "PersonaPlex live web UI on $(hostname), port ${PORT}"
echo "From your laptop, open a tunnel through a login node:"
echo "  ssh -L ${PORT}:$(hostname):${PORT} ${USER}@bridges2.psc.edu"
echo "Then open http://localhost:${PORT} in your browser."
echo

python example3/serve.py --device cuda --port "${PORT}" --persona sennet
