@echo off
chcp 65001 >nul
cls

echo ============================================================
echo Test DQN Movement Model
echo ============================================================
echo.
echo This script will test the trained DQN movement control model
echo.
echo Important Notes:
echo   1. Ensure the DQN model has been trained
echo   2. Model file location: multirotor\DQN_Movement\models\
echo   3. If testing in Unity environment, start Unity AirSim first
echo.
echo ============================================================
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

REM Check model file
echo [2/3] Checking DQN model...
if exist "%~dp0..\multirotor\DQN_Movement\models\movement_dqn_final.zip" (
    echo [OK] Trained DQN model found
) else (
    echo [!] Warning: Trained model not found, please train the model first
    echo     You can use main menu option [8] to train DQN movement
    echo.
    pause
    exit /b 1
)
echo.

REM Check test script
echo [3/3] Checking test script...
if exist "%~dp0..\multirotor\DQN_Movement\test_movement_dqn.py" (
    echo [OK] Test script found
) else (
    echo [!] Error: Test script does not exist
    pause
    exit /b 1
)
echo.

echo ============================================================
echo Starting DQN model testing...
echo ============================================================
echo.
echo Tip: Press Ctrl+C to interrupt testing at any time
echo.

python "%~dp0..\multirotor\DQN_Movement\test_movement_dqn.py"

echo.
echo ============================================================
echo Testing completed
echo ============================================================
echo.
echo Press any key to exit...
pause >nul
