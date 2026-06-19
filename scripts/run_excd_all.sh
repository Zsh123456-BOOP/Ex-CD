#!/usr/bin/env bash
# Ex-CD main experiment sweep: backbone x debiasing variant across the 4 primary datasets.
# Produces the Figure-1 cliff (vanilla exposure-stratified DOA) AND the method result
# (ips/dr tail-gap vs vanilla). Logs stream to logs/ for later pull-back.
#
# Usage:
#   bash scripts/run_excd_all.sh [PYTHON] [DEVICE]
# Examples:
#   bash scripts/run_excd_all.sh                                  # python, auto device
#   bash scripts/run_excd_all.sh /home/zsh/anaconda3/envs/xph_env/bin/python cuda:2
set -u

PY="${1:-python}"
DEVICE="${2:-auto}"
DATA_ROOT="${DATA_ROOT:-data_retricd}"
OUT_ROOT="${OUT_ROOT:-outputs}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_OUT="${OUT_ROOT}/excd_${TS}"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${RUN_OUT}"

DATASETS=("assist_09" "assist_17" "junyi" "nips34_retricd_small")
MODELS=("ncdm")            # add "kancd" to compare backbones
VARIANTS=("vanilla" "ips") # add "dr" for the doubly-robust variant

echo "Ex-CD sweep ${TS} | python=${PY} device=${DEVICE} out=${RUN_OUT}"
for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      LOG="${LOG_DIR}/excd_${ds}_${model}_${variant}_${TS}.log"
      echo ">>> ${ds} | ${model} | ${variant}  -> ${LOG}"
      "${PY}" -m excd.train \
        --dataset "${ds}" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${RUN_OUT}" \
        --model "${model}" \
        --variant "${variant}" \
        --device "${DEVICE}" \
        2>&1 | tee "${LOG}"
    done
  done
done

echo ">>> summarizing"
"${PY}" -m excd.summarize --output-dir "${RUN_OUT}" --out "${RUN_OUT}/summary.csv" 2>&1 | tee "${LOG_DIR}/excd_summary_${TS}.log"
echo "done: ${RUN_OUT}/summary.csv"
