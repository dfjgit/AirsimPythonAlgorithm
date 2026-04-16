@echo off
chcp 65001 >nul
cls

echo ============================================================
echo 系统运行（DDPG 权重预测）
echo ============================================================
echo.

set "AIRSIM_DRONE_COUNT=3"
if not "%AIRSIM_QUICK_DRONES%"=="" set "AIRSIM_DRONE_COUNT=%AIRSIM_QUICK_DRONES%"

REM 激活虚拟环境
echo [1/3] 激活 Python 虚拟环境...
if exist "%~dp0..\myvenv\Scripts\activate.bat" (
    call "%~dp0..\myvenv\Scripts\activate.bat"
    echo [OK] 虚拟环境已激活
) else (
    echo [!] 虚拟环境不存在，将使用系统 Python
)
echo.

REM 检查模型文件
echo [2/3] 检查模型文件...
if exist "%~dp0..\multirotor\DDPG_Weight\models\best_model.zip" (
    echo [OK] 模型文件已找到
) else (
    echo [!] 警告: 模型文件不存在
    echo [!] 请先运行选项 [4] 启动 DDPG+APF 训练
    echo.
    pause
    exit /b 1
)
echo.

REM 显示配置信息
echo ============================================================
echo 运行配置
echo ============================================================
echo 模式: DDPG 权重预测
echo 模型: multirotor\DDPG_Weight\models\best_model.zip
echo 环境配置: multirotor\apf_algorithm_config.json
echo 无人机数量: %AIRSIM_DRONE_COUNT%
echo ============================================================
echo.

REM 运行算法服务器(使用DDPG权重)
echo [3/3] 启动算法服务器...
python %~dp0..\multirotor\AlgorithmServer.py --use-learned-weights --model-path DDPG_Weight/models/best_model --drones %AIRSIM_DRONE_COUNT%

echo.
echo ============================================================
echo 系统已退出
echo ============================================================
echo.
pause


