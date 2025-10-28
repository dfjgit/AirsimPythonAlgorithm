@echo off
chcp 65001 >nul 2>&1
cls

echo ============================================================
echo DQN训练 - 使用真实AirSim环境
echo ============================================================
echo.
echo 本脚本将使用真实的Unity AirSim仿真环境训练DQN模型
echo.
echo 重要提示:
echo   1. 请先启动Unity AirSim仿真场景
echo   2. 确保Unity场景中有4台无人机 (UAV1-UAV4) 和环境
echo   3. 训练时间约33分钟，请保持Unity运行
echo.
echo ============================================================
echo.

REM 激活虚拟环境（如果存在）
if exist "%~dp0..\.venv\Scripts\activate.bat" (
    call "%~dp0..\.venv\Scripts\activate.bat"
)

REM 切换到训练脚本目录并运行
cd /d "%~dp0..\multirotor\DQN_Weight"
python train_with_airsim_improved.py

echo.
echo 按任意键退出...
pause >nul
