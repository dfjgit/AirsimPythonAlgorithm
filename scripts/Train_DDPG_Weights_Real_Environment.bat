﻿@echo off
chcp 65001 >nul 2>&1
cls

echo ============================================================
echo DDPG Weight APF Training (Real AirSim Environment)
echo ============================================================
echo.
echo This script will train the DDPG model using the real Unity AirSim simulation environment
echo.
echo Important Notes:
   1. Please start the Unity AirSim simulation scene first
   2. Ensure there are 3 UAVs (UAV1-UAV3) and the environment in the Unity scene
   3. Training takes about 33 minutes, keep Unity running
   4. After training, the model will be saved to multirotor\DDPG_Weight\models\
echo.
echo Configuration Options:
echo   [Recommended] Unified config: unified_train_config.json
echo   [Compatible] Old config: airsim_train_config_template.json
echo.
echo You can run directly with unified config or pass custom config:
echo   With unified config: Train_DDPG_Weights_Real_Environment.bat
echo   Custom config: Train_DDPG_Weights_Real_Environment.bat "path\to\config.json"
echo   Override via CLI: Train_DDPG_Weights_Real_Environment.bat --overwrite-model --model-name my_model
echo.
echo ============================================================
echo.

set "CONFIG_PATH=%~dp0..\multirotor\DDPG_Weight\configs\unified_train_config.json"
if not "%~1"=="" (
    set "CONFIG_PATH=%~1"
    shift
)

echo Using config: %CONFIG_PATH%
echo.
echo Press any key to start training...
pause >nul
echo.

REM Activate virtual environment (if exists)
echo [1/3] Activating Python virtual environment...
if exist "%~dp0..\myvenv\Scripts\activate.bat" (
    call "%~dp0..\myvenv\Scripts\activate.bat"
    echo [OK] Virtual environment activated successfully
) else (
    echo [!] Virtual environment does not exist, using system Python
)
echo.

REM Check training script
echo [2/3] Checking training script...
if exist "%~dp0..\multirotor\DDPG_Weight\train_with_airsim_improved.py" (
    echo [OK] Training script found
) else (
    echo [!] Error: Training script does not exist
    pause
    exit /b 1
)
echo.

REM Switch to training script directory and run
echo [3/3] Starting training...
echo.
cd /d "%~dp0..\multirotor\DDPG_Weight"
python train_with_airsim_improved.py --config "%CONFIG_PATH%" %*
set TRAIN_EXIT_CODE=%errorlevel%

echo.
if %TRAIN_EXIT_CODE% neq 0 (
    echo ============================================================
    echo [X] Training FAILED with error code: %TRAIN_EXIT_CODE%
    echo ============================================================
    echo.
    echo Please check:
    echo   1. Training log above for error details
    echo   2. External visualization log: multirotor\DDPG_Weight\logs\ddpg_airsim\external_vis.log
    echo   3. AlgorithmServer log for Unity/AirSim connection issues
    echo.
    echo Common causes:
    echo   - Unity AirSim not running or disconnected
    echo   - Out of memory (reduce total_timesteps in config)
    echo   - Visualization process crashed
    echo.
) else (
    echo ============================================================
    echo [OK] Training completed successfully
    echo ============================================================
)
echo.
echo Press any key to exit...
pause >nul
exit /b %TRAIN_EXIT_CODE%