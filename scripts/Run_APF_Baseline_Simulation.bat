@echo off
setlocal

set "ROOT=%~dp0.."
set "OUT=%ROOT%\analysis_results\apf_baseline_sim"
set "PYTHON_EXE=python"
if exist "%ROOT%\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\myvenv\Scripts\python.exe"
if exist "%ROOT%\.venv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\.venv\Scripts\python.exe"
if exist "%ROOT%\..\..\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\..\..\myvenv\Scripts\python.exe"
set "EPISODES_ARG="
if not "%AIRSIM_QUICK_APF_BASELINE_EPISODES%"=="" set "EPISODES_ARG=--episodes %AIRSIM_QUICK_APF_BASELINE_EPISODES%"

if /i "%AIRSIM_UI_LANG%"=="en" (
    echo ============================================================
    echo APF Baseline Multi-Episode Simulation Stage
    echo Output: %OUT%
    echo ============================================================
    echo   [1] fixed APF (fixed-policy baseline, no training stage)
    echo   [2] random APF (random-policy baseline, no training stage)
    echo.
) else (
    echo ============================================================
    echo APF 基线多轮仿真阶段
    echo 输出目录: %OUT%
    echo ============================================================
    echo   [1] fixed APF（固定策略基线，不参加训练）
    echo   [2] random APF（随机策略基线，不参加训练）
    echo.
)

"%PYTHON_EXE%" "%ROOT%\multirotor\Algorithm\apf_baseline_sim_runner.py" --out "%OUT%" %EPISODES_ARG% %*
exit /b %ERRORLEVEL%
