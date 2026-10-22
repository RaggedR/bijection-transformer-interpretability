#!/bin/bash
# Embedding ablation sweep for n=10
# Baseline (full model) should already exist in checkpoints/encoder_n10/
# If not, run without --ablate first.

cd "$(dirname "$0")"
COMMON="--model encoder --n 10 --source sample --train-size 500000 --batch-size 256 --device mps"

echo "=== Embedding ablation sweep: n=10 ==="
echo "Start: $(date)"

for ABLATE in drop-row drop-col drop-tab drop-row-col 1d-pos concat; do
    echo ""
    echo "--- Starting ablation: $ABLATE ($(date)) ---"
    PYTHONUNBUFFERED=1 python3 train.py $COMMON --ablate $ABLATE
    echo "--- Finished: $ABLATE ($(date)) ---"
done

echo ""
echo "=== Sweep complete: $(date) ==="
