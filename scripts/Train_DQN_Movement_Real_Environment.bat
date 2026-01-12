@echo off
chcp 65001 >nul
cls

echo ============================================================
echo DQN Movement Control Training (Real AirSim Environment)
echo ============================================================
echo.
echo This script will train the DQN movement control model using the real Unity AirSim simulation environment
echo.
echo Important Notes:
echo   1. Please start the Unity AirSim simulation scene first
echo   2. Ensure the Unity environment is started and running
echo   3. Training takes a long time, keep Unity running
echo   4. After training, the model will be saved to multirotor\DQN_Movement\models\
echo.
echo ============================================================
echo.
echo Press any key to start training...
pause >nul
echo.

REM Activate virtual environment
echo [1/3] Activating virtual environment...
if exist "%~dp0..\myvenv\Scripts\activate.bat" (
    call "%~dp0..\myvenv\Scripts\activate.bat"
    if %ERRORLEVEL% NEQ 0 (
        echo [!] Failed to activate virtual environment, using system Python
    ) else (
        echo [OK] Virtual environment activated successfully
    )
) else (
    echo [!] Virtual environment does not exist, using system Python
)
echo.

REM Check training script
echo [2/3] Checking training script...
if exist "%~dp0..\multirotor\DQN_Movement\train_movement_with_airsim.py" (
    echo [OK] Training script found
) else (
    echo [!] Error: Training script does not exist
    pause
    exit /b 1
)
echo.

REM Run training
echo [3/3] Starting training (connecting to AirSim)...
echo.
echo ============================================================
echo Tip: Press Ctrl+C to interrupt training at any time
echo ============================================================
echo.
python %~dp0..\multirotor\DQN_Movement\train_movement_with_airsim.py

echo.
echo ============================================================
echo Training completed
echo ============================================================
echo.
echo Press any key to exit...
pause >nul