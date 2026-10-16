@echo off
REM ================================================================
REM  RUN_ALL — complete pipeline of the thesis MIS24011 (steps 00a–10)
REM  Run from the repository root:   v2\RUN_ALL.bat
REM  Every step writes outputs_v2\log_<step>.txt and stops the chain if it fails.
REM  00a is resumable: if the download stops, run this file again.
REM ================================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
call v2\find_python.bat || exit /b 1
"%PY%" -c "import pandas, pyarrow, boto3, numpy, sklearn, xgboost, optuna, scipy, matplotlib; print('packages OK')" || (
  echo *** Missing package. Run:  "%PY%" -m pip install -r requirements.txt
  exit /b 1
)
if not exist outputs_v2 mkdir outputs_v2
for %%S in (00a_REDOWNLOAD_v3 00b_REPAIR_DATA 00c_VERIFY_ANNUAL 01_FEATURES_v2 02_ML_v2 02c_SEASONAL_BY_TYPE_v2 03_CLIMATE_v2 04_ENERGY_BURDEN_v2 05_EDA_v2 06_ESG_DF_DECARB_v2 07_FIGURES_v2 08_SIGNIFICANCE_v2 09_DASHBOARD_v2 10_SUPPLEMENT_v2) do (
    echo ===== %%S  started !DATE! !TIME!
    "%PY%" v2\%%S.py > outputs_v2\log_%%S.txt 2>&1
    if errorlevel 1 (
        echo *** %%S FAILED - see outputs_v2\log_%%S.txt
        exit /b 1
    )
    echo ===== %%S  finished !TIME!
)
echo.
echo ALL STEPS OK - results in outputs_v2\tables and outputs_v2\figures
