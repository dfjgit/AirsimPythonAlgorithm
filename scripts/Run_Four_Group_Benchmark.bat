@echo off
setlocal

set "ROOT=%~dp0.."
set "OUT=%ROOT%\analysis_results\four_group_benchmark"
set "PYTHON_EXE=python"
if exist "%ROOT%\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\myvenv\Scripts\python.exe"
if exist "%ROOT%\..\..\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\..\..\myvenv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo ============================================================
echo Running four-group frozen benchmark
echo Output: %OUT%
echo ============================================================

"%PYTHON_EXE%" "%ROOT%\multirotor\Algorithm\four_group_benchmark_runner.py" --unity-timeout 15 --out "%OUT%" %*
exit /b %ERRORLEVEL%
