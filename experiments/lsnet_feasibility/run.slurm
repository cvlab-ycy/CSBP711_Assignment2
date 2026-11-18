#!/usr/bin/env bash
#SBATCH --job-name=lsnet-feasibility
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/probe-%j.out
#SBATCH --error=logs/probe-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCH_HOME="$PWD/cache/torch"
export XDG_CACHE_HOME="$PWD/cache"
PROBE_PYTHON="${PROBE_PYTHON:-/home/700053286/miniconda3/envs/advanced-ai/bin/python}"
mkdir -p cache results
"$PROBE_PYTHON" -u probe.py \
  --data ../audited_v2/data/prepared/train_val.npz \
  --output "results/gpu-${SLURM_JOB_ID}" --device cuda --backend official \
  --scale tiny_adapted --train-per-class 128 --val-per-class 50 \
  --batch 64 --epochs 3
