@echo off
chcp 65001 >nul
cls

echo ============================================================
echo Start System - DDPG Weight Prediction Mode
echo ============================================================
echo.

REM Activate virtual environment
echo [1/3] Activating Python virtual environment...
if exist "%~dp0..\myvenv\Scripts\activate.bat" (
    call "%~dp0..\myvenv\Scripts\activate.bat"
    echo [OK] Virtual environment activated successfully
) else (
    echo [!] Virtual environment does not exist, using system Python
)
echo.

REM Check model file
echo [2/3] Checking model file...
if exist "%~dp0..\multirotor\DDPG_Weight\models\best_model.zip" (
    echo [OK] Model file found
) else (
    echo [!] Warning: Model file does not exist
    echo [!] Please run option [4] Train DDPG Weights first
    echo.
    pause
    exit /b 1
)
echo.

REM Display configuration information
echo ============================================================
echo Running Configuration
echo ============================================================
echo Mode: DDPG Weight Prediction
echo Model: multirotor\DDPG_Weight\models\best_model.zip
echo Reward Config: multirotor\DDPG_Weight\dqn_reward_config.json
echo Number of Drones: 3
echo ============================================================
echo.

REM Run algorithm server (using DDPG weights)
echo [3/3] Starting algorithm server...
python %~dp0..\multirotor\AlgorithmServer.py --use-learned-weights --model-path DDPG_Weight/models/best_model --drones 3

echo.
echo ============================================================
echo Program exited
echo ============================================================
echo.
pause