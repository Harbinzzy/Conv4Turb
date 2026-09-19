#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
MODEL_TYPE="${MODEL_TYPE:-edsr}"
APPROX_PARAM="${APPROX_PARAM:-0.5M}"
UPSCALE="${UPSCALE:-8}"
DATA_PATH="${DATA_PATH:-/data1t/blastnet}"
DEVICES="${DEVICES:-0,1,2,3}"

exec "${PYTHON}" train.py \
    --model-type "${MODEL_TYPE}" \
    --approx-param "${APPROX_PARAM}" \
    --upscale "${UPSCALE}" \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --loss-type mse_grad \
    --num-workers 20 \
    --accumulate-grad-batches 16 \
    --train-meta "${DATA_PATH}/train_data_summary.csv" \
    --val-meta "${DATA_PATH}/test_data_summary.csv" \
    --data-path "${DATA_PATH}" \
    --checkpoint-dir "checkpoints/${MODEL_TYPE}_${APPROX_PARAM}_${UPSCALE}x" \
    --accelerator gpu \
    --devices "${DEVICES}" \
    "$@"
