#!/bin/bash
#SBATCH --job-name=pernaf_ablation
#SBATCH --partition=base
#SBATCH --output=/home/scc_e_994694/PER-NAF/scc/logs/pernaf_%A_%a.out
#SBATCH --error=/home/scc_e_994694/PER-NAF/scc/logs/pernaf_%A_%a.err
#SBATCH --array=0-49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00

set -euo pipefail

cd /home/scc_e_994694/PER-NAF
mkdir -p scc/logs scc/outputs

echo "=== Starting PER-NAF 2x2 Factorial Array Task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) ==="
date

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"

python3 -u scc/run_cluster_job.py --task-id "${SLURM_ARRAY_TASK_ID:-0}" --out-dir /home/scc_e_994694/PER-NAF/scc/outputs

echo "=== Finished Task ${SLURM_ARRAY_TASK_ID:-0} ==="
date
