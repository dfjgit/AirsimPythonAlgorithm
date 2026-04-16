@echo off
chcp 65001 >nul
cls

echo ============================================================
echo DQN 控制模型验证
echo ============================================================
echo.
echo 本脚本将验证已训练的 DQN 控制模型
echo.
echo 重要提示:
echo   1. 请确认已完成 DQN 模型训练
echo   2. 模型文件位置: multirotor\DQN_Movement\models\
echo   3. 如需在 Unity 环境中验证，请先启动 Unity AirSim
echo.
echo ============================================================
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

REM 检查模型文件
echo [2/3] 检查DQN模型...
if exist "%~dp0..\multirotor\DQN_Movement\models\movement_dqn_final.zip" (
    echo [OK] 找到训练好的DQN模型
) else (
    echo [!] 警告: 未找到已训练的模型，请先完成训练
    echo     可通过主菜单选项 [7] 或 [8] 进入 DQN 控制训练
    echo.
    pause
    exit /b 1
)
echo.

REM 检查测试脚本
echo [3/3] 检查测试脚本...
if exist "%~dp0..\multirotor\DQN_Movement\tests\test_movement_dqn.py" (
    echo [OK] 测试脚本已找到
) else (
    echo [!] 错误: 测试脚本不存在
    pause
    exit /b 1
)
echo.

echo ============================================================
echo 开始验证 DQN 模型...
echo ============================================================
echo.
echo 提示: 按 Ctrl+C 可随时中断验证
echo.

python "%~dp0..\multirotor\DQN_Movement\tests\test_movement_dqn.py"

echo.
echo ============================================================
echo 验证结束
echo ============================================================
echo.
echo 按任意键退出...
pause >nul
