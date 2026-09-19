#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
MODEL_TYPE="${MODEL_TYPE:-edsr}"
APPROX_PARAM="${APPROX_PARAM:-0.5M}"
UPSCALE="${UPSCALE:-8}"
DATA_PATH="${DATA_PATH:-/data1t/blastnet}"
DEVICES="${DEVICES:-0}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exec "${PYTHON}" test.py --help
fi
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 CHECKPOINT [test.py options]" >&2
    exit 2
fi

CHECKPOINT="$1"
shift

exec "${PYTHON}" test.py \
    --checkpoint "${CHECKPOINT}" \
    --model-type "${MODEL_TYPE}" \
    --approx-param "${APPROX_PARAM}" \
    --upscale "${UPSCALE}" \
    --test-data all \
    --data-path "${DATA_PATH}" \
    --test-meta "${DATA_PATH}/test_data_summary.csv" \
    --paramvar-meta "${DATA_PATH}/paramvar_data_summary.csv" \
    --forcedhit-meta "${DATA_PATH}/forcedhit_data_summary.csv" \
    --output-dir "eval/${MODEL_TYPE}_${APPROX_PARAM}_${UPSCALE}x" \
    --accelerator gpu \
    --devices "${DEVICES}" \
    "$@"
