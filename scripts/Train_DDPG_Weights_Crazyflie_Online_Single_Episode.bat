@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
cls

set "ROOT_DIR=%~dp0.."
set "CONFIG_PATH=%ROOT_DIR%\multirotor\DDPG_Weight\configs\crazyflie_online_single_episode_config.json"
set "MODELS_DIR=%ROOT_DIR%\multirotor\DDPG_Weight\models"
set "HELPER_SCRIPT=%ROOT_DIR%\multirotor\DDPG_Weight\single_episode_launcher.py"
set "PYTHON_CMD=python"

echo ============================================================
echo DDPG权重APF训练 (实体无人机在线单轮次训练)
echo ============================================================
echo.
echo 本入口将执行：
echo   1. 自动选择默认模型
echo   2. 实体无人机在线飞行 1 个 episode
echo   3. 在 episode 结束后执行高权重更新
echo   4. 保存模型后自动退出
echo.

if exist "%ROOT_DIR%\myvenv\Scripts\python.exe" (
    set "PYTHON_CMD=%ROOT_DIR%\myvenv\Scripts\python.exe"
) else (
    if exist "%ROOT_DIR%\.venv\Scripts\python.exe" (
        set "PYTHON_CMD=%ROOT_DIR%\.venv\Scripts\python.exe"
    ) else (
        if exist "%ROOT_DIR%\..\..\.venv\Scripts\python.exe" (
            set "PYTHON_CMD=%ROOT_DIR%\..\..\.venv\Scripts\python.exe"
        ) else (
            echo [提示] 未检测到虚拟环境，将使用系统 Python
        )
    )
)

if not exist "%HELPER_SCRIPT%" (
    echo [错误] 未找到模型选择辅助脚本：
    echo         %HELPER_SCRIPT%
    echo.
    pause
    exit /b 1
)

if not exist "%CONFIG_PATH%" (
    echo [错误] 未找到单轮训练配置：
    echo         %CONFIG_PATH%
    echo.
    pause
    exit /b 1
)

for /f "usebackq tokens=1,* delims==" %%A in (`"%PYTHON_CMD%" "%HELPER_SCRIPT%" --models-dir "%MODELS_DIR%" --emit-env`) do (
    set "%%A=%%B"
)

echo ============================================================
if /i "!MODEL_STATUS!"=="online" (
    echo [默认模型] 本次将继续使用最新实体机模型
) else (
    if /i "!MODEL_STATUS!"=="airsim" (
        echo [默认模型] 本次将从仿真预训练模型开始
    ) else (
        echo [默认模型] 未找到可用模型，请先完成仿真训练
    )
)
if defined MODEL_PATH (
    echo            !MODEL_PATH!.zip
)
echo ============================================================
echo.

if defined MODEL_PATH (
    echo 直接回车：使用默认模型
    echo 输入路径：使用自定义模型地址（可带 .zip）
) else (
    echo 如需手动指定模型，请输入完整路径（可带 .zip）
    echo 直接回车：返回主菜单
)
echo.
set "MODEL_OVERRIDE="
set /p MODEL_OVERRIDE=模型路径:

set "SELECTED_MODEL="
if defined MODEL_OVERRIDE (
    for /f "usebackq delims=" %%I in (`"%PYTHON_CMD%" "%HELPER_SCRIPT%" --normalize-model-path "!MODEL_OVERRIDE!"`) do (
        set "SELECTED_MODEL=%%I"
    )
) else (
    if defined MODEL_PATH (
        set "SELECTED_MODEL=!MODEL_PATH!"
    )
)

if not defined SELECTED_MODEL (
    echo.
    echo [提示] 未指定模型，返回主菜单
    timeout /t 2 >nul
    exit /b 0
)

if not exist "!SELECTED_MODEL!.zip" (
    echo.
    echo [错误] 模型文件不存在：
    echo         !SELECTED_MODEL!.zip
    echo.
    pause
    exit /b 1
)

echo.
echo [最终模型] !SELECTED_MODEL!.zip
echo [配置文件] %CONFIG_PATH%
echo.
echo 按任意键开始单轮实体机在线训练...
pause >nul
echo.

cd /d "%ROOT_DIR%\multirotor\DDPG_Weight"
"%PYTHON_CMD%" train_with_crazyflie_online.py --config "%CONFIG_PATH%" --continue-model "!SELECTED_MODEL!"
set "TRAIN_EXIT_CODE=%ERRORLEVEL%"

echo.
if %TRAIN_EXIT_CODE% neq 0 (
    echo ============================================================
    echo [X] 单轮实体机训练失败，错误码：%TRAIN_EXIT_CODE%
    echo ============================================================
) else (
    echo ============================================================
    echo [OK] 单轮实体机训练完成
    echo ============================================================
)
echo.
echo 按任意键返回主菜单...
pause >nul
exit /b %TRAIN_EXIT_CODE%
