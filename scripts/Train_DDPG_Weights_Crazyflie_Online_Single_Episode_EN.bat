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
echo DDPG Weight APF Training (Single-Episode Crazyflie Online)
echo ============================================================
echo.
echo This shortcut will:
echo   1. Auto-select the default model
echo   2. Run one Crazyflie online training episode
echo   3. Apply the weighted real-flight update after that episode
echo   4. Save the model and exit automatically
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
            echo [Tip] No virtual environment detected, using system Python
        )
    )
)

if not exist "%HELPER_SCRIPT%" (
    echo [Error] Model selection helper script not found:
    echo         %HELPER_SCRIPT%
    echo.
    pause
    exit /b 1
)

if not exist "%CONFIG_PATH%" (
    echo [Error] Single-episode config not found:
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
    echo [Default model] The latest Crazyflie online model will be used
) else (
    if /i "!MODEL_STATUS!"=="airsim" (
        echo [Default model] Falling back to the AirSim pretrained model
    ) else (
        echo [Default model] No available model was found. Please finish AirSim pretraining first
    )
)
if defined MODEL_PATH (
    echo                !MODEL_PATH!.zip
)
echo ============================================================
echo.

if defined MODEL_PATH (
    echo Press Enter to use the default model
    echo Or type a custom model path ^(with or without .zip^)
) else (
    echo Type a custom model path ^(with or without .zip^)
    echo Or press Enter to return to the main menu
)
echo.
set "MODEL_OVERRIDE="
set /p MODEL_OVERRIDE=Model path: 

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
    echo [Tip] No model selected, returning to the main menu
    timeout /t 2 >nul
    exit /b 0
)

if not exist "!SELECTED_MODEL!.zip" (
    echo.
    echo [Error] Model file not found:
    echo         !SELECTED_MODEL!.zip
    echo.
    pause
    exit /b 1
)

echo.
echo [Selected model] !SELECTED_MODEL!.zip
echo [Config] %CONFIG_PATH%
echo.
echo Press any key to start the single-episode Crazyflie training run...
pause >nul
echo.

cd /d "%ROOT_DIR%\multirotor\DDPG_Weight"
"%PYTHON_CMD%" train_with_crazyflie_online.py --config "%CONFIG_PATH%" --continue-model "!SELECTED_MODEL!"
set "TRAIN_EXIT_CODE=%ERRORLEVEL%"

echo.
if %TRAIN_EXIT_CODE% neq 0 (
    echo ============================================================
    echo [X] Single-episode Crazyflie training failed with exit code %TRAIN_EXIT_CODE%
    echo ============================================================
) else (
    echo ============================================================
    echo [OK] Single-episode Crazyflie training completed successfully
    echo ============================================================
)
echo.
echo Press any key to return to the main menu...
pause >nul
exit /b %TRAIN_EXIT_CODE%
