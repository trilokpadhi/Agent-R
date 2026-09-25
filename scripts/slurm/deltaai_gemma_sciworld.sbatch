#!/usr/bin/env bash
# Agent-R: Gemma-3-12B on ScienceWorld (Co-Evolving / ETO), on NCSA DeltaAI.
#
#   sbatch --account=biqx-dtai-gh scripts/slurm/deltaai_gemma_sciworld.sbatch
#
# One node, 4x GH200. The pipeline places work by local GPU index (one vLLM server and one
# ScienceWorld server per GPU), so it does not span nodes.
#
# CHARGING: 1 SU = one GH200-hour, billed on the LARGER of GPUs, cores/72 and memory/110GB, for the
# time reserved. 4 GPUs + 288 cores + 440 GB is therefore 4 SU/hour - asking for more cores or
# memory than that would bill extra GPUs for no gain. Do not raise these without recomputing.
#
# RESUMING: ghx4 caps a job at 48 h and three iterations may exceed that. run_agentr.sh writes a
# .done marker per step and skips finished trees and eval items, so resubmitting this exact file
# continues where it stopped. That is the intended way to cross the wall, not a longer limit.
#
#SBATCH --job-name=agentr-gemma-sciworld
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=440g
#SBATCH --time=48:00:00
#SBATCH --output=logs/agentr-gemma-sciworld-%j.out
#SBATCH --error=logs/agentr-gemma-sciworld-%j.out

set -uo pipefail

BASE=/work/nvme/biqx/$USER/agentr
CODE=$BASE/code
CONFIG=$CODE/pipeline/config-gemma-sciworld.yaml     # Unseen (test split), eval at 100 rounds

echo "node    : $(hostname)"
echo "gpus    : ${SLURM_GPUS_ON_NODE:-?}"
echo "config  : $CONFIG"
echo "started : $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Only for its Python 3.12 interpreter; the three venvs carry every package.
module load python/miniforge3_pytorch/2.12.0 >/dev/null 2>&1 || true

# CUDA compiler mismatch: the module sets CUDA_HOME to 13.1 while this stack is 12.9, and vLLM
# JIT-compiles kernels at startup ("CUDA compiler and CUDA toolkit headers are incompatible").
# `module load cuda/12.9` does not override it. Verified on this machine by the pmpo project.
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9
export PATH="$CUDA_HOME/bin:$PATH"

export AGENTR_SITE_ENV=$CODE/scripts/site.env
export HF_HOME=$BASE/hf
export HF_HUB_OFFLINE=1          # the model is already local; never reach out mid-run
export TOKENIZERS_PARALLELISM=false

# A long-lived environment server per GPU holds many sockets open.
ulimit -n 65536 2>/dev/null || true

cd "$CODE"
exec bash scripts/run_agentr.sh "$CONFIG"
