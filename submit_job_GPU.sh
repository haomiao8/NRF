#!/bin/bash
# Usage: ./submit_job_GPU.sh <job_name> <script.sh>
# Adjust partition, node, and resource settings for your cluster.

mkdir -p arch_diagnose
mkdir -p arch_diagnose_err

sbatch --requeue \
    -p YOUR_PARTITION \
    -t 72:00:00 \
    -n 8 \
    -N 1 \
    --mem=80G \
    --gres=gpu:1 \
    --job-name="$1" \
    -e "./arch_diagnose_err/%j-$1.err" \
    -o "./arch_diagnose/%j-$1.out" \
    "$2"
