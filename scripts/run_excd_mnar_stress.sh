#!/usr/bin/env bash
# Ex-CD Phase-2 closed loop: synthetic exposure-skew sweep. Thins rare-concept train
# responses at increasing gamma and checks that vanilla's tail-DOA gap grows while ips/dr
# keeps it smaller. Run on the datasets with a real concept hierarchy (assist_09, assist_17).
#
# Usage: bash scripts/run_excd_mnar_stress.sh [PYTHON] [DEVICE]
set -u

PY="${1:-python}"
DEVICE="${2:-auto}"
DATA_ROOT="${DATA_ROOT:-data_retricd}"
OUT_ROOT="${OUT_ROOT:-outputs}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

for ds in "assist_09" "assist_17"; do
  LOG="${LOG_DIR}/excd_mnar_stress_${ds}_${TS}.log"
  echo ">>> MNAR-stress ${ds} -> ${LOG}"
  "${PY}" -m excd.mnar_stress \
    --dataset "${ds}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_ROOT}" \
    --model ncdm \
    --variants vanilla ips \
    --gammas 0.0 0.3 0.6 \
    --bottom-deciles 3 \
    --device "${DEVICE}" \
    2>&1 | tee "${LOG}"
done
echo "done"
