@echo off
setlocal

set "ROOT=%~dp0.."
set "OUT=%ROOT%\analysis_results\apf_baseline_sim"
set "RAW_LOG_DIR="
set "PYTHON_EXE=python"
if exist "%ROOT%\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\myvenv\Scripts\python.exe"
if exist "%ROOT%\.venv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\.venv\Scripts\python.exe"
if exist "%ROOT%\..\..\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\..\..\myvenv\Scripts\python.exe"
set "EPISODES_ARG="
if not "%AIRSIM_QUICK_APF_BASELINE_EPISODES%"=="" set "EPISODES_ARG=--episodes %AIRSIM_QUICK_APF_BASELINE_EPISODES%"
set "FORWARD_ARGS=%*"

:parse_args
if "%~1"=="" goto args_parsed
if /i "%~1"=="--out" (
    if not "%~2"=="" set "OUT=%~2"
    shift
) else if /i "%~1"=="--raw-log-dir" (
    if not "%~2"=="" set "RAW_LOG_DIR=%~2"
    shift
)
shift
goto parse_args

:args_parsed

if /i "%AIRSIM_UI_LANG%"=="en" (
    echo ============================================================
    echo APF Baseline Multi-Episode Simulation Stage
    echo Output: %OUT%
    if defined RAW_LOG_DIR echo Raw logs: %RAW_LOG_DIR%
    echo ============================================================
    echo   [1] fixed APF (fixed-policy baseline, no training stage)
    echo   [2] random APF (random-policy baseline, no training stage)
    echo.
) else (
    echo ============================================================
    echo APF 基线多轮仿真阶段
    echo 输出目录: %OUT%
    if defined RAW_LOG_DIR echo 原始日志目录: %RAW_LOG_DIR%
    echo ============================================================
    echo   [1] fixed APF（固定策略基线，不参加训练）
    echo   [2] random APF（随机策略基线，不参加训练）
    echo.
)

"%PYTHON_EXE%" "%ROOT%\multirotor\Algorithm\apf_baseline_sim_runner.py" %EPISODES_ARG% %FORWARD_ARGS%
exit /b %ERRORLEVEL%
