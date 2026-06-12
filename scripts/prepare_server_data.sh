#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-$HOME/ConceptSkillCDM/data}"
RAW_ROOT="${2:-$HOME/RetriCD/data_source}"
PREP_ROOT="${3:-$HOME/RetriCD/data_retricd}"
DATASETS=(assist_09 assist_17 junyi nips34)

mkdir -p "$RAW_ROOT" "$PREP_ROOT"
for dataset in "${DATASETS[@]}"; do
  src="$SRC_ROOT/$dataset"
  dst="$RAW_ROOT/$dataset"
  if [[ ! -d "$src" ]]; then
    echo "[ERROR] missing source dataset: $src" >&2
    exit 1
  fi
  mkdir -p "$dst"
  rsync -a "$src/" "$dst/"
  echo "[OK] copied $src -> $dst"
done

cd "$HOME/RetriCD"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON_BIN:-$HOME/anaconda3/envs/xph_env/bin/python}" tools/build_retricd_splits.py \
  --source-root "$RAW_ROOT" \
  --output-root "$PREP_ROOT" \
  --split-mode chronological_per_student \
  --ratio 0.7 0.1 0.2 \
  --nips34-match-junyi
