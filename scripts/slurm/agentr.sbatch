#!/usr/bin/env bash
# Agent-R pipeline as a single Slurm job on one multi-GPU node.
#
#   sbatch --export=ALL,CONFIG=pipeline/config.yaml scripts/slurm/agentr.sbatch
#
# One node, because every step is placed by GPU index on the local machine: search and revise run
# one vLLM server per GPU, and SFT uses all GPUs through DeepSpeed on that node. Multi-node would
# need a launcher change, not just more --nodes.
#
# The job is designed to be resubmitted. Each step writes .done and is skipped on the next run, and
# within a step finished trees / eval items are skipped, so hitting the time limit is not a loss -
# resubmit the same command and it continues. Budget generously anyway: the full three iterations
# took ~40 h on 7xH100 for WebShop.
#
#SBATCH --job-name=agent-r
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-agent-r-%j.out
#SBATCH --error=slurm-agent-r-%j.out

set -uo pipefail

CONFIG=${CONFIG:?set CONFIG, e.g. sbatch --export=ALL,CONFIG=pipeline/config.yaml scripts/slurm/agentr.sbatch}
REPO=${REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO" || exit 1

echo "node:    $(hostname)"
echo "gpus:    ${SLURM_GPUS_ON_NODE:-unknown} (config gpus: must match, or edit pipeline/config.yaml)"
echo "config:  $CONFIG"
echo "commit:  $(git rev-parse --short=12 HEAD 2>/dev/null || echo 'not a git checkout')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# The config's `gpus:` decides the sharding; a mismatch with the allocation would leave GPUs idle
# or oversubscribe them, so fail early rather than run a wrong-shaped job.
CFG_GPUS=$(python3 scripts/config_env.py "$CONFIG" | sed -n 's/^GPUS=//p' | tr -d "'")
if [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [ "$CFG_GPUS" != "$SLURM_GPUS_ON_NODE" ]; then
  echo "ERROR: config gpus=$CFG_GPUS but Slurm allocated $SLURM_GPUS_ON_NODE." >&2
  echo "Set --gres=gpu:$CFG_GPUS, or change gpus: in $CONFIG." >&2
  exit 1
fi

# A single long-lived environment server per GPU holds open sockets for hours; raise the limit if
# the site default is low.
ulimit -n 65536 2>/dev/null || true

exec bash scripts/run_agentr.sh "$CONFIG"
