#!/usr/bin/env bash
#SBATCH --job-name=fashion-spatial
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fashion-%j.out
#SBATCH --error=logs/fashion-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
mkdir -p outputs reports
exec 9>outputs/.experiment.lock
flock -n 9 || { echo 'Another experiment process holds the project lock'; exit 1; }
STUDY_PYTHON="${STUDY_PYTHON:-$HOME/miniconda3/envs/advanced-ai/bin/python}"
export STUDY_CPU_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$STUDY_CPU_THREADS"
export MKL_NUM_THREADS="$STUDY_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$STUDY_CPU_THREADS"
export PYTHONHASHSEED=711
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$PWD/outputs/matplotlib-cache"
date -u
hostname
nvidia-smi
"$STUDY_PYTHON" -m pip freeze > reports/environment-lock.txt
"$STUDY_PYTHON" -m pytest -q
"$STUDY_PYTHON" -u src/study.py all --device cuda
date -u
