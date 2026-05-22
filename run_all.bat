@echo off
set PY=python

echo ============================================================
echo  STEP 1: Preprocess
echo ============================================================
%PY% preprocess.py
if errorlevel 1 goto error

echo ============================================================
echo  STEP 2: Train
echo ============================================================
%PY% train.py
if errorlevel 1 goto error

echo ============================================================
echo  STEP 3: Report Results
echo ============================================================
%PY% report_results.py
if errorlevel 1 goto error

echo ============================================================
echo  STEP 4: FGSM Robustness
echo ============================================================
%PY% compare_fgsm_baselines.py
if errorlevel 1 goto error

echo ============================================================
echo  STEP 5: Gaussian Noise Robustness
echo ============================================================
%PY% evaluate_robustness_all.py
if errorlevel 1 goto error

echo ============================================================
echo  STEP 5b: PGD Adversarial Attack
echo ============================================================
%PY% evaluate_pgd.py
if errorlevel 1 goto error

echo ============================================================
echo  STEP 6: Calibration (ECE + Brier + Reliability Diagram)
echo ============================================================
%PY% evaluate_calibration.py
if errorlevel 1 goto error

echo ============================================================
echo  STEP 7: Generate Tables + Summary Figure
echo ============================================================
%PY% generate_results_tables.py
if errorlevel 1 goto error

echo.
echo ============================================================
echo  ALL DONE! Check results/ folder.
echo ============================================================
goto end

:error
echo.
echo [ERROR] Script failed at the step above. Check output above.
exit /b 1

:end
