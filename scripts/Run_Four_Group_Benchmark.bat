@echo off
setlocal

set "ROOT=%~dp0.."
set "OUT=%ROOT%\analysis_results\four_group_benchmark"
set "PYTHON_EXE=python"
if exist "%ROOT%\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\myvenv\Scripts\python.exe"
if exist "%ROOT%\..\..\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\..\..\myvenv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

if /i "%AIRSIM_UI_LANG%"=="zh" (
    echo ============================================================
    echo 正在执行四组仿真评测（冻结策略）
    echo 输出目录: %OUT%
    echo 本阶段将依次在 Unity/AirSim 中评测以下四组：
    echo   [1] fixed APF（固定策略基线，不参加训练）
    echo   [2] random APF（随机策略基线，不参加训练）
    echo   [3] DDPG+APF（使用已训练模型，冻结策略）
    echo   [4] Pure DQN（使用已训练模型，冻结策略）
    echo ============================================================
) else (
    echo ============================================================
    echo Running four-group frozen benchmark
    echo Output: %OUT%
    echo ============================================================
)

"%PYTHON_EXE%" "%ROOT%\multirotor\Algorithm\four_group_benchmark_runner.py" --unity-timeout 15 --out "%OUT%" %*
exit /b %ERRORLEVEL%
