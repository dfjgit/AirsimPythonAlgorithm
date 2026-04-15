@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
cls

echo ============================================================
echo DDPG权重APF训练 (真实AirSim环境)
echo ============================================================
echo.
echo 本脚本将在 Unity AirSim 仿真环境中训练 DDPG 权重模型
echo.
echo 重要提示:
echo   1. 请先启动 Unity AirSim 仿真场景
echo   2. 请确认 Unity 场景中的环境已就绪
echo   3. 训练时间约 33 分钟，请保持 Unity 持续运行
echo   4. 训练完成后，模型将保存到 multirotor\DDPG_Weight\models\
echo.
echo 配置文件选项:
echo   [推荐] 统一配置: unified_train_config.json
echo   [归档] 旧配置样例: configs\legacy\airsim_train_config_template.json
echo.
echo 你可以直接使用统一配置，或通过参数传入自定义配置:
echo   使用统一配置: 训练权重DDPG-真实环境.bat
echo   自定义配置: 训练权重DDPG-真实环境.bat "path\to\config.json"
echo   命令行覆盖: 训练权重DDPG-真实环境.bat --overwrite-model --model-name my_model
echo.
echo ============================================================
echo.

set "CONFIG_PATH=%~dp0..\multirotor\DDPG_Weight\configs\unified_train_config.json"
if not "%~1"=="" (
    set "FIRST_ARG=%~1"
    if /i not "!FIRST_ARG:~0,2!"=="--" (
        set "CONFIG_PATH=%~1"
        shift
    )
)

echo 使用配置: %CONFIG_PATH%
echo.
echo 按任意键开始训练...
pause >nul
echo.

echo [1/3] 激活 Python 虚拟环境...
if exist "%~dp0..\myvenv\Scripts\activate.bat" (
    call "%~dp0..\myvenv\Scripts\activate.bat"
    echo [OK] 虚拟环境已激活
) else (
    echo [!] 虚拟环境不存在，将使用系统 Python
)
echo.

echo [2/3] 检查训练脚本...
if exist "%~dp0..\multirotor\DDPG_Weight\train_with_airsim_improved.py" (
    echo [OK] 训练脚本已找到
) else (
    echo [错误] 未找到训练脚本
    pause
    exit /b 1
)
echo.

echo [3/3] 开始训练...
echo.
cd /d "%~dp0..\multirotor\DDPG_Weight"
python train_with_airsim_improved.py --config "%CONFIG_PATH%" %*
set "TRAIN_EXIT_CODE=%ERRORLEVEL%"

echo.
if %TRAIN_EXIT_CODE% neq 0 (
    echo ============================================================
    echo [X] 训练失败，错误码: %TRAIN_EXIT_CODE%
    echo ============================================================
    echo.
    echo 请检查:
    echo   1. 上方训练日志中的错误详情
    echo   2. external_vis.log 可视化日志
    echo   3. Unity / AirSim 连接状态
) else (
    echo ============================================================
    echo [OK] 训练完成
    echo ============================================================
)
echo.
echo 按任意键退出...
pause >nul
exit /b %TRAIN_EXIT_CODE%

