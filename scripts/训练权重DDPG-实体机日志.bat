@echo off
chcp 65001 >nul 2>&1
cls

echo ============================================================
echo DDPG权重训练（Crazyflie日志-离线）
echo ============================================================
echo.
echo 本脚本将使用Crazyflie日志数据进行离线训练
echo.
echo 重要提示:
echo   1. 需在JSON配置中提供 log_path（.json/.csv）
echo   2. 日志仅用于离线训练，不控制实体机
echo   3. 模型将保存到 multirotor\DDPG_Weight\models\
echo.
echo 训练参数将从 JSON 配置文件读取
echo 默认配置:
echo   [推荐] 统一配置: unified_train_config.json
echo   [兼容] 旧配置: crazyflie_logs_train_config.json
echo.
echo 示例:
echo   训练权重DDPG-实体机日志.bat "path\to\custom_config.json" ^
echo     --continue-model "multirotor\DDPG_Weight\models\weight_predictor_airsim" ^
echo     --total-timesteps 2000
echo.
echo ============================================================
echo.

set "CONFIG_PATH=%~dp0..\multirotor\DDPG_Weight\unified_train_config.json"
if not "%~1"=="" (
    set "CONFIG_PATH=%~1"
    shift
)

echo 使用配置: %CONFIG_PATH%
echo.

echo 按任意键开始训练...
pause >nul
echo.

REM 激活虚拟环境（如果存在）
echo [1/3] 激活Python虚拟环境...
if exist "%~dp0..\myvenv\Scripts\activate.bat" (
    call "%~dp0..\myvenv\Scripts\activate.bat"
    echo [OK] 虚拟环境已激活
) else (
    echo [!] 虚拟环境不存在，使用系统Python
)
echo.

REM 检查训练脚本
echo [2/3] 检查训练脚本...
if exist "%~dp0..\multirotor\DDPG_Weight\train_with_crazyflie_logs.py" (
    echo [OK] 训练脚本已找到
) else (
    echo [!] 错误: 训练脚本不存在
    pause
    exit /b 1
)
echo.

REM 切换到训练脚本目录并运行
echo [3/3] 开始训练...
echo.
cd /d "%~dp0..\multirotor\DDPG_Weight"
python train_with_crazyflie_logs.py --config "%CONFIG_PATH%" %*

echo.
echo ============================================================
echo 训练结束
echo ============================================================
echo.
echo 按任意键退出...
pause >nul
