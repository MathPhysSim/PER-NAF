#!/bin/bash
#SBATCH --job-name=pernaf_collect
#SBATCH --partition=base
#SBATCH --output=/home/scc_e_994694/PER-NAF/scc/logs/collect_%j.out
#SBATCH --error=/home/scc_e_994694/PER-NAF/scc/logs/collect_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail

cd /home/scc_e_994694/PER-NAF
export JAX_PLATFORMS=cpu
python3 -u scc/run_cluster_job.py --aggregate --out-dir /home/scc_e_994694/PER-NAF/scc/outputs
echo "FERTIG pernaf_collect $(date -Iseconds)"
