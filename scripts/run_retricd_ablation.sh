#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/RetriCD}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/xph_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data_retricd}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/retricd_ablation_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
DATASETS=(${DATASETS:-assist_09 junyi})
VARIANTS=(no_retrieval random_retrieval recency_only overlap_only difficulty_only latent_only)

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$ROOT/logs"
bash scripts/ensure_server_runtime.sh

for dataset in "${DATASETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    gpu="${GPU:-2}"
    log="$ROOT/logs/retricd_${dataset}_${variant}_$(date +%Y%m%d_%H%M%S).log"
    echo "[RUN] dataset=$dataset variant=$variant gpu=$gpu log=$log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m retricd.train \
      --dataset "$dataset" \
      --data-root "$DATA_ROOT" \
      --output-dir "$OUT_ROOT" \
      --variant "$variant" \
      --seed "$SEED" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      >"$log" 2>&1
  done
done

echo "[ALL_DONE] outputs=$OUT_ROOT"
