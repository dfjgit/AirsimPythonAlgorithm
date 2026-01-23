@echo off
chcp 65001 >nul 2>&1
cls

echo ============================================================
echo DDPG权重APF训练 (虚实融合训练)
echo ============================================================
echo.
echo 本脚本将使用虚实融合模式训练DDPG模型
echo.
echo 重要提示:
echo   1. 请先启动Unity AirSim仿真场景
echo   2. 确保实体Crazyflie已连接并处于安全可控状态
echo   3. 虚实融合训练将自动设置指定无人机的isCrazyflieMirror=true
echo   4. 这些无人机将使用实体机的实时状态数据
echo   5. 其他无人机仍使用虚拟AirSim环境数据
echo   6. 训练完成后，模型将保存到 multirotor\DDPG_Weight\models\
echo.
echo 训练参数将从JSON配置文件读取
echo 默认配置:
echo   multirotor\DDPG_Weight\hybrid_train_config_template.json
echo.
echo 可将自定义配置路径作为第一个参数传入，
echo 或直接覆盖参数，例如:
echo   "path\to\custom_config.json"
echo   --mirror-drones UAV1 UAV2
echo   --total-timesteps 500 --step-duration 5
echo.
echo ============================================================
echo.

set "CONFIG_PATH=%~dp0..\multirotor\DDPG_Weight\hybrid_train_config_template.json"
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
if exist "%~dp0..\multirotor\DDPG_Weight\train_with_hybrid.py" (
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
python train_with_hybrid.py --config "%CONFIG_PATH%" %*

echo.
echo ============================================================
echo 训练结束
echo ============================================================
echo.
echo 按任意键退出...
pause >nul
