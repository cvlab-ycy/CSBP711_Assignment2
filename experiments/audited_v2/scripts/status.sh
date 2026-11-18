#!/usr/bin/env bash
set -euo pipefail
STUDY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STUDY_ROOT"
module load slurm/23.02.08
STUDY_JOB_ID="$(cat logs/latest-job-id.txt)"
squeue -j "$STUDY_JOB_ID" -o '%.18i %.12P %.20j %.10T %.12M %.30R'
sacct -j "$STUDY_JOB_ID" --format=JobID,JobName,State,Elapsed,ExitCode,AllocTRES --parsable2
if test -f "logs/fashion-$STUDY_JOB_ID.out"; then
    tail -n 18 "logs/fashion-$STUDY_JOB_ID.out"
fi
if test -s "logs/fashion-$STUDY_JOB_ID.err"; then
    tail -n 18 "logs/fashion-$STUDY_JOB_ID.err"
fi
