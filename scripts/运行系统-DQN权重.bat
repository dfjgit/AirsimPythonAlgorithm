@echo off
chcp 65001 >nul
echo ============================================================
echo 启动无人机系统 - 使用DQN权重预测
echo ============================================================
echo.

REM 检查模型文件
if not exist "%~dp0..\multirotor\DQN_Weight\models\weight_predictor_simple.zip" (
    echo ============================================================
    echo 警告: 未找到训练好的DQN模型
    echo ============================================================
    echo.
    echo 模型文件不存在: multirotor\DQN_Weight\models\weight_predictor_simple.zip
    echo.
    echo 请选择:
    echo   1. 先训练模型 - 运行 训练权重DQN-模拟.bat
    echo   2. 使用固定权重 - 运行 运行系统-固定权重.bat
    echo.
    pause
    exit /b 1
)

echo [OK] DQN模型文件存在
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

REM 显示配置信息
echo ============================================================
echo 运行配置
echo ============================================================
echo 模式: DQN权重预测
echo 模型: multirotor\DQN_Weight\models\weight_predictor_simple.zip
echo 奖励配置: multirotor\DQN_Weight\dqn_reward_config.json
echo ============================================================
echo.

REM 运行算法服务器（使用DQN）
echo 启动算法服务器...
python %~dp0..\multirotor\AlgorithmServer.py --use-learned-weights

echo.
echo ============================================================
echo 程序已退出
echo ============================================================
echo.

pause


