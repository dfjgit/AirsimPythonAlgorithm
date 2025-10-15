@echo off
chcp 65001 >nul
echo ========================================
echo DQN无人机移动模型测试
echo ========================================
echo.

REM 激活虚拟环境
echo [1/2] 激活虚拟环境...
call %~dp0..\.venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 虚拟环境激活失败
    pause
    exit /b 1
)
echo [OK] 虚拟环境已激活
echo.

REM 运行测试
echo [2/2] 开始测试...
echo.
python %~dp0..\multirotor\DQN_Movement\test_movement_dqn.py

echo.
echo ========================================
echo 测试结束
echo ========================================
echo.
echo 按任意键退出...
pause >nul
