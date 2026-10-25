#!/bin/bash
#SBATCH --job-name=pernaf_ablation
#SBATCH --output=scc/logs/pernaf_%A_%a.out
#SBATCH --error=scc/logs/pernaf_%A_%a.err
#SBATCH --array=0-49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00

set -euo pipefail

mkdir -p scc/logs scc/outputs

echo "=== Starting PER-NAF 2x2 Factorial Array Task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) ==="
date

# Activate Python environment if present
if [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export JAX_PLATFORMS=cpu

python3 scc/run_cluster_job.py --task-id "${SLURM_ARRAY_TASK_ID:-0}" --out-dir scc/outputs

echo "=== Finished Task ${SLURM_ARRAY_TASK_ID:-0} ==="
date
