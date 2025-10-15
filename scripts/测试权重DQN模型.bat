@echo off
chcp 65001 >nul
echo ============================================================
echo 测试DQN训练模型
echo ============================================================
echo.

REM 激活虚拟环境
echo 正在激活Python虚拟环境...
call %~dp0..\.venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 虚拟环境激活失败
    pause
    exit /b 1
)
echo [OK] 虚拟环境已激活
echo.

REM 检查模型文件是否存在
if not exist "%~dp0..\multirotor\DQN_Weight\models\weight_predictor_simple.zip" (
    echo ============================================================
    echo 错误: 模型文件不存在
    echo ============================================================
    echo.
    echo 找不到训练好的模型文件
    echo 路径: multirotor\DQN_Weight\models\weight_predictor_simple.zip
    echo.
    echo 请先运行 训练权重DQN-模拟.bat 训练模型
    echo.
    pause
    exit /b 1
)

echo [OK] 找到模型文件
echo.

REM 运行测试脚本
echo 开始测试模型...
echo ============================================================
echo.
python %~dp0..\multirotor\DQN_Weight\test_trained_model.py

echo.
echo ============================================================
echo 测试完成
echo ============================================================
echo.

pause
