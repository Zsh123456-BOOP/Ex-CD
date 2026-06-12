#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/RetriCD}"
SRC_ROOT="${SRC_ROOT:-$HOME/ConceptSkillCDM/data}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data_retricd_extra_smoke}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/xph_env/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/retricd_extra_smoke_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EXPORT_SUPPORT_LIMIT="${EXPORT_SUPPORT_LIMIT:-5000}"
SEED="${SEED:-42}"
REFERENCE_DATASET="${REFERENCE_DATASET:-junyi}"

DATASETS=(
  assist_09_chold
  assist_12
  assist_12_clean15_item50
  assist_15
  assist_17_chold
  cdbd_a0910
  cdbd_lsat
  ednet_kt1
  ednet_kt1_gap
  ednet_kt1_gap_probe
  ednet_kt1_gap_small_t50u2600
  ednet_kt1_gap_small_t60u2200
  ednet_kt1_gap_small_t65u1200_long
  ednet_kt1_gap_small_t70u2000
  ednet_kt1_gap_t50u5000
  ednet_kt1_gap_t60u5000
  ednet_kt1_gap_t65u3000_long
  ednet_kt1_gap_t70u5000
  frcsub
  junyi_chold
  junyi_long
  junyi_sample
  math2
  nips34_l3
)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] missing python: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT"
mkdir -p "$DATA_ROOT" "$OUT_ROOT" "$ROOT/logs"
bash scripts/ensure_server_runtime.sh
export ROOT SRC_ROOT DATA_ROOT PYTHON_BIN OUT_ROOT EPOCHS BATCH_SIZE EXPORT_SUPPORT_LIMIT SEED REFERENCE_DATASET
export DATASETS_JOINED="${DATASETS[*]}"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from retricd.preprocess import concatenate_splits, load_source_splits
from retricd.splitters import build_dataset

root = Path(os.environ.get("ROOT", str(Path.home() / "RetriCD")))
source_root = os.environ.get("SRC_ROOT", str(Path.home() / "ConceptSkillCDM" / "data"))
output_root = os.environ.get("DATA_ROOT", str(root / "data_retricd_extra_smoke"))
seed = int(os.environ.get("SEED", "42"))
reference_dataset = os.environ.get("REFERENCE_DATASET", "junyi")
datasets = os.environ["DATASETS_JOINED"].split()
reference_rows = len(concatenate_splits(load_source_splits(source_root, reference_dataset)))
print(f"[BUILD] reference_dataset={reference_dataset} reference_rows={reference_rows}")
for dataset in datasets:
    rows = len(concatenate_splits(load_source_splits(source_root, dataset)))
    target_rows = reference_rows if rows > reference_rows else 0
    output_name = f"{dataset}_retricd_small" if target_rows else dataset
    path = build_dataset(
        source_root=source_root,
        output_root=output_root,
        dataset=dataset,
        output_name=output_name,
        split_mode="chronological_per_student",
        ratio=(0.7, 0.1, 0.2),
        seed=seed,
        target_rows=target_rows,
    )
    print(f"[OK] built source={dataset} rows={rows} target_rows={target_rows} output={path.name}")
PY

run_one() {
  local dataset="$1"
  local gpu="$2"
  local log="$ROOT/logs/retricd_${dataset}_extra_smoke_$(date +%Y%m%d_%H%M%S).log"
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
    --export-support-limit "$EXPORT_SUPPORT_LIMIT" \
    >"$log" 2>&1
  echo "[DONE] dataset=$dataset log=$log"
}

OUTPUT_DATASETS=()
for dataset in "${DATASETS[@]}"; do
  if [[ -d "$DATA_ROOT/${dataset}_retricd_small" ]]; then
    OUTPUT_DATASETS+=("${dataset}_retricd_small")
  else
    OUTPUT_DATASETS+=("$dataset")
  fi
done

idx=0
while (( idx < ${#OUTPUT_DATASETS[@]} )); do
  first="${OUTPUT_DATASETS[$idx]}"
  second=""
  if (( idx + 1 < ${#OUTPUT_DATASETS[@]} )); then
    second="${OUTPUT_DATASETS[$((idx + 1))]}"
  fi
  run_one "$first" 2 &
  pid_a=$!
  if [[ -n "$second" ]]; then
    run_one "$second" 3 &
    pid_b=$!
    wait "$pid_a" "$pid_b"
  else
    wait "$pid_a"
  fi
  idx=$((idx + 2))
done

echo "[ALL_DONE] outputs=$OUT_ROOT"
