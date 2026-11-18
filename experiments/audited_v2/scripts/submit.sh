#!/usr/bin/env bash
set -euo pipefail
STUDY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STUDY_ROOT"
module load slurm/23.02.08
mkdir -p logs outputs reports
sbatch --parsable scripts/run.slurm | tee logs/latest-job-id.txt
