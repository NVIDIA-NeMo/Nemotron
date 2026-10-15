#!/bin/bash
# Run one step of the cookbook as a Slurm job, inside the container (Pyxis), once per node.
# Submit from the cookbook directory; see the README for the resources each step needs:
#   sbatch [--account=... --partition=... --time=...] <resources> slurm/step.sbatch <step>
#SBATCH --job-name=super-vl-lora
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%j.log

set -euo pipefail
source config.env
mkdir -p "$WORKSPACE"  # Pyxis can only mount directories that exist

# The first node coordinates multi-node training (read by steps/train.sh).
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

srun --container-image="$CONTAINER_IMAGE" --container-mounts="$PWD:$PWD,$WORKSPACE:$WORKSPACE" \
  --container-workdir="$PWD" --no-container-mount-home \
  bash "steps/${1:?usage: sbatch slurm/step.sbatch <download|dataprep|convert|train|merge|try> [args]}.sh" "${@:2}"
