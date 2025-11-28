@echo off
chcp 65001 >nul
cls

echo ============================================================
echo 启动系统 - DDPG权重预测模式
echo ============================================================
echo.

REM 激活虚拟环境
echo [1/3] 激活Python虚拟环境...
if exist "%~dp0..\.venv\Scripts\activate.bat" (
    call "%~dp0..\.venv\Scripts\activate.bat"
    echo [OK] 虚拟环境已激活
) else (
    echo [!] 虚拟环境不存在，使用系统Python
)
echo.

REM 检查模型文件
echo [2/3] 检查模型文件...
if exist "%~dp0..\multirotor\DDPG_Weight\models\best_model.zip" (
    echo [OK] 模型文件已找到
) else (
    echo [!] 警告: 模型文件不存在
    echo [!] 请先运行选项 [4] 训练权重DDPG
    echo.
    pause
    exit /b 1
)
echo.

REM 显示配置信息
echo ============================================================
echo 运行配置
echo ============================================================
echo 模式: DDPG权重预测
echo 模型: multirotor\DDPG_Weight\models\best_model.zip
echo 奖励配置: multirotor\DDPG_Weight\dqn_reward_config.json
echo 无人机数量: 3
echo ============================================================
echo.

REM 运行算法服务器（使用DDPG权重）
echo [3/3] 启动算法服务器...
python %~dp0..\multirotor\AlgorithmServer.py --use-learned-weights --model-path DDPG_Weight/models/best_model --drones 3

echo.
echo ============================================================
echo 程序已退出
echo ============================================================
echo.
pause


