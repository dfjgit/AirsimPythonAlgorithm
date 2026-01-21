@echo off
chcp 65001 >nul 2>&1
cls

echo ============================================================
echo DDPG Weight Training (Crazyflie Real Drone - Online)
echo ============================================================
echo.
echo This script will train DDPG weights using real Crazyflie online data.
echo.
echo Important Notes:
echo   1. Ensure the Crazyflie is connected and safe to operate
echo   2. Ensure the AlgorithmServer and Crazyswarm are ready
echo   3. Training will update weights in real time
echo   4. Model will be saved to multirotor\DDPG_Weight\models\
echo.
echo Training parameters are loaded from a JSON config.
echo Default config:
echo   multirotor\DDPG_Weight\crazyflie_online_train_config.json
echo.
echo You can pass a custom config path as the first argument,
echo or override any parameter, for example:
echo   "path\to\custom_config.json"
echo   --continue-model "multirotor\DDPG_Weight\models\weight_predictor_airsim"
echo   --total-timesteps 500 --step-duration 5
echo.
echo ============================================================
echo.
set "CONFIG_PATH=%~dp0..\multirotor\DDPG_Weight\crazyflie_online_train_config.json"
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
if exist "%~dp0..\multirotor\DDPG_Weight\train_with_crazyflie_online.py" (
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
python train_with_crazyflie_online.py --config "%CONFIG_PATH%" %*

echo.
echo ============================================================
echo Training completed
echo ============================================================
echo.
echo Press any key to exit...
pause >nul
