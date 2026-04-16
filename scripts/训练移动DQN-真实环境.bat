@echo off
chcp 65001 >nul
cls

echo ============================================================
echo DQN控制训练 (真实AirSim环境)
echo ============================================================
echo.
echo 本脚本将在 Unity AirSim 仿真环境中训练 DQN 控制模型
echo.
echo 重要提示:
echo   1. 请先启动 Unity AirSim 仿真场景
echo   2. 请确认 Unity 环境已启动并正常运行
echo   3. DQN 将通过 AlgorithmServer 控制无人机移动
echo   4. 训练完成后，模型将保存到 multirotor\DQN_Movement\models\
echo.
echo ============================================================
echo.
echo 按任意键开始训练...
pause >nul
echo.

REM 激活虚拟环境
echo [1/3] 激活 Python 虚拟环境...
if exist "%~dp0..\myvenv\Scripts\activate.bat" (
    call "%~dp0..\myvenv\Scripts\activate.bat"
    if %ERRORLEVEL% NEQ 0 (
        echo [!] 虚拟环境激活失败，将使用系统 Python
    ) else (
        echo [OK] 虚拟环境已激活
    )
) else (
    echo [!] 虚拟环境不存在，将使用系统 Python
)
echo.

REM 检查训练脚本
echo [2/3] 检查训练脚本...
if exist "%~dp0..\multirotor\DQN_Movement\scripts\train_movement_with_airsim.py" (
    echo [OK] 训练脚本已找到
) else (
    echo [错误] 训练脚本不存在
    pause
    exit /b 1
)
echo.

REM 运行训练
echo [3/3] 开始训练（连接 AirSim）...
echo.
echo ============================================================
echo 提示: 按 Ctrl+C 可随时中断训练
echo ============================================================
echo.
python "%~dp0..\multirotor\DQN_Movement\scripts\train_movement_with_airsim.py"
set "TRAIN_EXIT_CODE=%ERRORLEVEL%"

echo.
if %TRAIN_EXIT_CODE% NEQ 0 (
    echo ============================================================
    echo [X] 训练失败，错误码: %TRAIN_EXIT_CODE%
    echo ============================================================
) else (
    echo ============================================================
    echo 训练完成
    echo ============================================================
)
echo.
echo 按任意键退出...
pause >nul
exit /b %TRAIN_EXIT_CODE%
