@echo off
chcp 65001 >nul
echo ========================================
echo DQN无人机移动训练 (AirSim集成)
echo ========================================
echo.
echo 警告: 此脚本需要Unity客户端运行！
echo 请确保Unity环境已启动并运行
echo.
echo 按任意键继续...
pause >nul

REM 激活虚拟环境
echo [1/3] 激活虚拟环境...
call %~dp0..\.venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 虚拟环境激活失败
    pause
    exit /b 1
)
echo [OK] 虚拟环境已激活
echo.

REM 检查依赖
echo [2/3] 检查依赖...
python -c "import torch; import stable_baselines3; print('[OK] 所有依赖已安装')" 2>nul
if errorlevel 1 (
    echo 依赖未完全安装，正在安装...
    pip install torch stable-baselines3 numpy gym -i https://pypi.tuna.tsinghua.edu.cn/simple
)

REM 运行训练
echo.
echo [3/3] 开始训练 (连接AirSim)...
echo.
echo 提示: 按 Ctrl+C 可随时中断训练
echo.
python %~dp0..\multirotor\DQN_Movement\train_movement_with_airsim.py

echo.
echo ========================================
echo 训练结束
echo ========================================
echo.
echo 按任意键退出...
pause >nul
