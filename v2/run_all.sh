#!/usr/bin/env bash
# Complete pipeline of the thesis MIS24011 (steps 00a–10). Run from the repository root:
#   bash v2/run_all.sh
set -euo pipefail
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=${PYTHON:-python3}
mkdir -p outputs_v2
for S in 00a_REDOWNLOAD_v3 00b_REPAIR_DATA 00c_VERIFY_ANNUAL 01_FEATURES_v2 02_ML_v2 02c_SEASONAL_BY_TYPE_v2 \
         03_CLIMATE_v2 04_ENERGY_BURDEN_v2 05_EDA_v2 06_ESG_DF_DECARB_v2 07_FIGURES_v2 \
         08_SIGNIFICANCE_v2 09_DASHBOARD_v2 10_SUPPLEMENT_v2; do
  echo "===== $S  started $(date +%T)"
  "$PY" "v2/$S.py" > "outputs_v2/log_$S.txt" 2>&1 || { echo "*** $S FAILED - see outputs_v2/log_$S.txt"; exit 1; }
  echo "===== $S  finished $(date +%T)"
done
echo "ALL STEPS OK - results in outputs_v2/tables and outputs_v2/figures"
