#!/usr/bin/env bash
#SBATCH --job-name=CSBP711-A2-recipe
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/recipe-%j.out
#SBATCH --error=logs/recipe-%j.err
# Formal run: sbatch run.slurm
# Smoke test (2 epochs, seed 42, separate outputs): sbatch --time=01:00:00 --export=ALL,SMOKE=1 run.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=711 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/../lsnet_feasibility/deps${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCH_HOME="$PWD/cache/torch"
export XDG_CACHE_HOME="$PWD/cache"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
STUDY_PYTHON="${STUDY_PYTHON:-$HOME/miniconda3/envs/advanced-ai/bin/python}"
A1_ROOT="${A1_ROOT:-$PWD/../audited_v2}"
mkdir -p cache logs reports
"$STUDY_PYTHON" -m pip freeze > reports/base-environment.txt
"$STUDY_PYTHON" -u recipe_study.py --a1-root "$A1_ROOT" ${SMOKE:+--smoke}
echo "Recipe check finished; all GPU work was in job ${SLURM_JOB_ID}."
