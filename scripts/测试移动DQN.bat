@echo off
chcp 65001 >nul
cls

echo ============================================================
echo 测试移动DQN模型
echo ============================================================
echo.
echo 本脚本将测试训练好的DQN移动控制模型
echo.
echo 重要提示:
echo   1. 确保已经训练好DQN模型
echo   2. 模型文件位置: multirotor\DQN_Movement\models\
echo   3. 如果要在Unity环境中测试，请先启动Unity AirSim
echo.
echo ============================================================
echo.

REM 激活虚拟环境
echo [1/3] 激活虚拟环境...
if exist "%~dp0..\myvenv\Scripts\activate.bat" (
    call "%~dp0..\myvenv\Scripts\activate.bat"
    if %ERRORLEVEL% NEQ 0 (
        echo [!] 虚拟环境激活失败，使用系统Python
    ) else (
        echo [OK] 虚拟环境已激活
    )
) else (
    echo [!] 虚拟环境不存在，使用系统Python
)
echo.

REM 检查模型文件
echo [2/3] 检查DQN模型...
if exist "%~dp0..\multirotor\DQN_Movement\models\movement_dqn_final.zip" (
    echo [OK] 找到训练好的DQN模型
) else (
    echo [!] 警告: 未找到训练好的模型，请先训练模型
    echo     可以使用主菜单选项 [8] 训练移动DQN
    echo.
    pause
    exit /b 1
)
echo.

REM 检查测试脚本
echo [3/3] 检查测试脚本...
if exist "%~dp0..\multirotor\DQN_Movement\test_movement_dqn.py" (
    echo [OK] 测试脚本已找到
) else (
    echo [!] 错误: 测试脚本不存在
    pause
    exit /b 1
)
echo.

echo ============================================================
echo 开始测试 DQN 模型...
echo ============================================================
echo.
echo 提示: 按 Ctrl+C 可随时中断测试
echo.

python "%~dp0..\multirotor\DQN_Movement\test_movement_dqn.py"

echo.
echo ============================================================
echo 测试结束
echo ============================================================
echo.
echo 按任意键退出...
pause >nul
