#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-$HOME/ConceptSkillCDM/data}"
DST_ROOT="${2:-$HOME/RetriCD/data}"
DATASETS=(assist_09 assist_17 junyi nips34)

mkdir -p "$DST_ROOT"
for dataset in "${DATASETS[@]}"; do
  src="$SRC_ROOT/$dataset"
  dst="$DST_ROOT/$dataset"
  if [[ ! -d "$src" ]]; then
    echo "[ERROR] missing source dataset: $src" >&2
    exit 1
  fi
  mkdir -p "$dst"
  rsync -a "$src/" "$dst/"
  echo "[OK] copied $src -> $dst"
done

