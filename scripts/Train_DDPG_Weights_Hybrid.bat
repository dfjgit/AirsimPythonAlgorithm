@echo off
chcp 65001 >nul 2>&1
cls

echo ============================================================
echo DDPG Weight APF Training (Hybrid Training)
echo ============================================================
echo.
echo This script will train the DDPG model using hybrid mode
echo.
echo Important Notes:
echo   1. Please start the Unity AirSim simulation scene first
echo   2. Ensure the real Crazyflie is connected and in a safe controllable state
echo   3. Hybrid training will automatically set isCrazyflieMirror=true for specified drones
echo   4. These drones will use real-time state data from the physical drones
echo   5. Other drones will still use virtual AirSim environment data
echo   6. After training, the model will be saved to multirotor\DDPG_Weight\models\
echo.
echo Training parameters will be read from JSON configuration file
echo Default configuration:
echo   [Recommended] Unified config: unified_train_config.json
echo   [Compatible] Old config: hybrid_train_config_template.json
echo.
echo You can pass a custom configuration path as the first argument,
echo or override parameters directly, for example:
echo   "path\to\custom_config.json"
echo   --mirror-drones UAV1 UAV2
echo   --total-timesteps 500 --step-duration 5
echo.
echo ============================================================
echo.

set "CONFIG_PATH=%~dp0..\multirotor\DDPG_Weight\unified_train_config.json"
if not "%~1"=="" (
    set "CONFIG_PATH=%~1"
    shift
)

echo Using configuration: %CONFIG_PATH%
echo.
echo Press any key to start training...
pause >nul
echo.

REM Activate virtual environment (if exists)
echo [1/3] Activating Python virtual environment...
if exist "%~dp0..\myvenv\Scripts\activate.bat" (
    call "%~dp0..\myvenv\Scripts\activate.bat"
    echo [OK] Virtual environment activated
) else (
    echo [!] Virtual environment does not exist, using system Python
)
echo.

REM Check training script
echo [2/3] Checking training script...
if exist "%~dp0..\multirotor\DDPG_Weight\train_with_hybrid.py" (
    echo [OK] Training script found
) else (
    echo [!] Error: Training script does not exist
    pause
    exit /b 1
)
echo.

REM Change to training script directory and run
echo [3/3] Starting training...
echo.
cd /d "%~dp0..\multirotor\DDPG_Weight"
python train_with_hybrid.py --config "%CONFIG_PATH%" %*

echo.
echo ============================================================
echo Training completed
echo ============================================================
echo.
echo Press any key to exit...
pause >nul
