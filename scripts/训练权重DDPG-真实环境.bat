@echo off
chcp 65001 >nul 2>&1
cls

echo ============================================================
echo DDPG权重APF训练 (真实AirSim环境)
echo ============================================================
echo.
echo 本脚本将使用真实的Unity AirSim仿真环境训练DDPG模型
echo.
echo 重要提示:
   1. 请先启动Unity AirSim仿真场景
   2. 确保Unity场景中有4台无人机 (UAV1-UAV4) 和环境
   3. 训练时间约33分钟，请保持Unity运行
   4. 训练完成后，模型将保存到 multirotor\DDPG_Weight\models\
echo.
echo 配置文件选项:
echo   [推荐] 统一配置: unified_train_config.json
echo   [兼容] 旧配置: airsim_train_config_template.json
echo.
echo 你可以直接运行使用统一配置，或作为参数传入自定义配置:
echo   使用统一配置: 训练权重DDPG-真实环境.bat
echo   自定义配置: 训练权重DDPG-真实环境.bat "path\to\config.json"
echo   命令行覆盖: 训练权重DDPG-真实环境.bat --overwrite-model --model-name my_model
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
if exist "%~dp0..\multirotor\DDPG_Weight\train_with_airsim_improved.py" (
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
python train_with_airsim_improved.py --config "%CONFIG_PATH%" %*

echo.
echo ============================================================
echo 训练结束
echo ============================================================
echo.
echo 按任意键退出...
pause >nul

