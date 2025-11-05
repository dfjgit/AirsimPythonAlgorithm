@echo off
chcp 65001 >nul
cls

echo ============================================================
echo DQN移动控制训练 (真实AirSim环境)
echo ============================================================
echo.
echo 本脚本将使用真实的Unity AirSim仿真环境训练DQN移动控制模型
echo.
echo 重要提示:
echo   1. 请先启动Unity AirSim仿真场景
echo   2. 确保Unity环境已启动并运行
echo   3. 训练时间较长，请保持Unity运行
echo   4. 训练完成后，模型将保存到 multirotor\DQN_Movement\models\
echo.
echo ============================================================
echo.
echo 按任意键开始训练...
pause >nul
echo.

REM 激活虚拟环境
echo [1/3] 激活虚拟环境...
if exist "%~dp0..\.venv\Scripts\activate.bat" (
    call "%~dp0..\.venv\Scripts\activate.bat"
    if %ERRORLEVEL% NEQ 0 (
        echo [!] 虚拟环境激活失败，使用系统Python
    ) else (
        echo [OK] 虚拟环境已激活
    )
) else (
    echo [!] 虚拟环境不存在，使用系统Python
)
echo.

REM 检查训练脚本
echo [2/3] 检查训练脚本...
if exist "%~dp0..\multirotor\DQN_Movement\train_movement_with_airsim.py" (
    echo [OK] 训练脚本已找到
) else (
    echo [!] 错误: 训练脚本不存在
    pause
    exit /b 1
)
echo.

REM 运行训练
echo [3/3] 开始训练 (连接AirSim)...
echo.
echo ============================================================
echo 提示: 按 Ctrl+C 可随时中断训练
echo ============================================================
echo.
python %~dp0..\multirotor\DQN_Movement\train_movement_with_airsim.py

echo.
echo ============================================================
echo 训练结束
echo ============================================================
echo.
echo 按任意键退出...
pause >nul
