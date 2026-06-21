#!/usr/bin/env bash
# CertCD full experiment: real-data selective-prediction sweep (the kill-switch) + synthetic
# certificate validation (primary novelty evidence) + CAT certification efficiency (decision task).
# Streams logs to logs/ for pull-back.
#
# Usage: bash scripts/run_certcd_all.sh [PYTHON] [DEVICE]
#   bash scripts/run_certcd_all.sh /home/zsh/anaconda3/envs/xph_env/bin/python cuda:1
set -u

PY="${1:-python}"
DEVICE="${2:-auto}"
DATA_ROOT="${DATA_ROOT:-data_retricd}"
OUT_ROOT="${OUT_ROOT:-outputs}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_OUT="${OUT_ROOT}/certcd_${TS}"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}" "${RUN_OUT}"

DATASETS=("assist_09" "assist_17" "junyi" "nips34_retricd_small")
MODELS=("ncdm" "kancd")

echo "CertCD sweep ${TS} | python=${PY} device=${DEVICE} out=${RUN_OUT}"

# 1) real-data selective prediction (kill-switch: excess AURC vs count AND mc_dropout must be > 0)
for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    LOG="${LOG_DIR}/certcd_${ds}_${model}_${TS}.log"
    echo ">>> real ${ds} | ${model} -> ${LOG}"
    "${PY}" -m certcd.run --dataset "${ds}" --data-root "${DATA_ROOT}" \
      --output-dir "${RUN_OUT}" --model "${model}" --device "${DEVICE}" 2>&1 | tee "${LOG}"
  done
done

# 2) synthetic certificate validation (precision/recall vs true recoverability)
for model in "${MODELS[@]}"; do
  LOG="${LOG_DIR}/certcd_synth_${model}_${TS}.log"
  echo ">>> synthetic | ${model} -> ${LOG}"
  "${PY}" -m certcd.run_synthetic --output-dir "${RUN_OUT}/synth" --model "${model}" --device "${DEVICE}" 2>&1 | tee "${LOG}"
done

# 3) CAT certification efficiency (decision task) on the two real concept-structured datasets
for ds in "assist_09" "assist_17" "nips34_retricd_small"; do
  LOG="${LOG_DIR}/certcd_cat_${ds}_${TS}.log"
  echo ">>> cat ${ds} -> ${LOG}"
  "${PY}" -m certcd.cat --dataset "${ds}" --data-root "${DATA_ROOT}" \
    --output-dir "${RUN_OUT}/cat" --model ncdm --device "${DEVICE}" 2>&1 | tee "${LOG}"
done

# 4) summary dashboard
"${PY}" -m certcd.summarize --output-dir "${RUN_OUT}" --out "${RUN_OUT}/certcd_summary.csv" 2>&1 | tee "${LOG_DIR}/certcd_summary_${TS}.log"
echo "done: ${RUN_OUT}/certcd_summary.csv"
