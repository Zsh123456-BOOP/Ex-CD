#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/RetriCD}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/xph_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data_retricd}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/retricd_full_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
DATASETS=(assist_09 assist_17 junyi nips34_retricd_small)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] missing python: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$ROOT/logs"
bash scripts/ensure_server_runtime.sh

run_one() {
  local dataset="$1"
  local gpu="$2"
  local log="$ROOT/logs/retricd_${dataset}_full_$(date +%Y%m%d_%H%M%S).log"
  echo "[RUN] dataset=$dataset gpu=$gpu log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m retricd.train \
    --dataset "$dataset" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT_ROOT" \
    --variant full \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --topk 16 \
    --max-history-len 128 \
    --embed-dim 64 \
    --fidelity-margin-weight 0.2 \
    --export-support-limit 50000 \
    >"$log" 2>&1
  echo "[DONE] dataset=$dataset log=$log"
}

run_one "${DATASETS[0]}" 2 &
pid_a=$!
run_one "${DATASETS[1]}" 3 &
pid_b=$!
wait "$pid_a" "$pid_b"

run_one "${DATASETS[2]}" 2 &
pid_c=$!
run_one "${DATASETS[3]}" 3 &
pid_d=$!
wait "$pid_c" "$pid_d"

echo "[ALL_DONE] outputs=$OUT_ROOT"
