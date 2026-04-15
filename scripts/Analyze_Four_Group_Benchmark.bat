@echo off
setlocal

set "ROOT=%~dp0.."
set "EVAL_CSV=%ROOT%\analysis_results\four_group_benchmark\four_group_eval_episodes.csv"
set "OUT=%ROOT%\analysis_results\four_group_benchmark"
set "PYTHON_EXE=python"
if exist "%ROOT%\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\myvenv\Scripts\python.exe"
if exist "%ROOT%\..\..\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\..\..\myvenv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

if /i "%AIRSIM_UI_LANG%"=="zh" (
    echo ============================================================
    echo 正在生成四组主结果分析
    echo 输入文件: %EVAL_CSV%
    echo 输出目录: %OUT%
    echo ============================================================
) else (
    echo ============================================================
    echo Generating four-group benchmark report
    echo Input:  %EVAL_CSV%
    echo Output: %OUT%
    echo ============================================================
)

"%PYTHON_EXE%" "%ROOT%\multirotor\Algorithm\four_group_benchmark_analyzer.py" --eval-csv "%EVAL_CSV%" --out "%OUT%" %*
exit /b %ERRORLEVEL%
