#!/bin/bash

mkdir -p arch_diagnose
mkdir -p arch_diagnose_err

sbatch --requeue \
    -p sablab-gpu \
    -t 72:00:00 \
    -n 8 \
    -N 1 \
    --nodelist ai-gpu03 \
    --mem=80G \
    --gres=gpu:1 \
    --mail-type=ALL \
    --mail-user=hc872@cornell.edu \
    --job-name="$1" \
    -e "./arch_diagnose_err/%j-$1.err" \
    -o "./arch_diagnose/%j-$1.out" \
    "$2"
    #--wrap "$2 2>&1 | tee ./test_out/%j-$1.out"
