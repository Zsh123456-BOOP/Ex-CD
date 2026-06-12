#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/RetriCD}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/xph_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data_retricd}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/retricd_fidelity_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
DATASETS=(${DATASETS:-assist_09 assist_17 junyi nips34_retricd_small})

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$ROOT/logs"
bash scripts/ensure_server_runtime.sh

for dataset in "${DATASETS[@]}"; do
  gpu="${GPU:-2}"
  log="$ROOT/logs/retricd_${dataset}_fidelity_$(date +%Y%m%d_%H%M%S).log"
  echo "[RUN] dataset=$dataset fidelity-full gpu=$gpu log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m retricd.train \
    --dataset "$dataset" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT_ROOT" \
    --variant full \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    >"$log" 2>&1
done

echo "[ALL_DONE] outputs=$OUT_ROOT"
