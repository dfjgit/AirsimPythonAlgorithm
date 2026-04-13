@echo off
setlocal

set "ROOT=%~dp0.."
set "EVAL_CSV=%ROOT%\analysis_results\four_group_benchmark\four_group_eval_episodes.csv"
set "REGISTRY=%ROOT%\multirotor\benchmark_registry.json"
set "OUT=%ROOT%\analysis_results\family_comparisons"
set "PYTHON_EXE=python"
if exist "%ROOT%\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\myvenv\Scripts\python.exe"
if exist "%ROOT%\..\..\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\..\..\myvenv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo ============================================================
echo Generating family comparison reports
echo Input:    %EVAL_CSV%
echo Registry: %REGISTRY%
echo Output:   %OUT%
echo ============================================================

"%PYTHON_EXE%" "%ROOT%\multirotor\Algorithm\family_analysis.py" --eval-csv "%EVAL_CSV%" --registry "%REGISTRY%" --out "%OUT%" %*
exit /b %ERRORLEVEL%
