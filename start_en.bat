@echo off
chcp 65001 >nul
:menu
cls
echo ============================================================
echo    AirSim UAV Simulation System - Main Menu
echo ============================================================
echo.
echo === System Operation ===
echo   [1] Run System (Fixed Weights)[⭐Available!]
echo   [2] Run System (DDPG Weight Prediction) [⭐Available!]
echo   [3] Run System (DQN Model) [Under Development]
echo.
echo === DDPG Weight APF Training ===
echo   [4] Train DDPG Weights (Real AirSim Environment) [⭐Available!]
echo.
echo === DQN Movement Control Training ===
echo   [5] Train DQN Movement (Real AirSim Environment)[Has some issues]
echo.
echo === System Information ===
echo   [6] View System Information
echo   [0] Exit
echo.
echo ============================================================
echo.

set /p choice=Please enter an option (0-6): 

if "%choice%"=="1" goto run_normal
if "%choice%"=="2" goto run_dqn
if "%choice%"=="3" goto run_dqn_movement
if "%choice%"=="4" goto train_weight_airsim
if "%choice%"=="5" goto train_movement_airsim
if "%choice%"=="6" goto info
if "%choice%"=="0" goto end

echo.
echo Invalid option, please select again
timeout /t 2 >nul
goto menu

:run_normal
cls
echo ============================================================
echo Start System - Fixed Weights Mode
echo ============================================================
echo.
call scripts\Run_System_Fixed_Weights.bat
goto menu

:run_dqn
cls
echo ============================================================
echo Start System - DDPG Weight Prediction Mode
echo ============================================================
echo.
call scripts\Run_System_DDPG_Weights.bat
goto menu

:run_dqn_movement
cls
echo ============================================================
echo Run System - DQN Model
echo ============================================================
echo.
echo [Tip] This feature is under development, coming soon...
echo.
pause
goto menu

:train_weight_airsim
cls
echo ============================================================
echo DDPG Weight APF Training (Real AirSim Environment)
echo ============================================================
echo.
call scripts\Train_DDPG_Weights_Real_Environment.bat
goto menu

:train_movement_airsim
cls
echo ============================================================
echo DQN Movement Control Training (Real AirSim Environment)
echo ============================================================
echo.
call scripts\Train_DQN_Movement_Real_Environment.bat
goto menu

:info
cls
echo ============================================================
echo    System Information
echo ============================================================
echo.
echo Project Structure:
echo   - multirotor\AlgorithmServer.py    : Algorithm Server
echo   - multirotor\Algorithm\            : APF Algorithm Implementation
echo   - multirotor\DDPG_Weight\         : DDPG Weight APF Training
echo   - multirotor\DQN_Movement\         : DQN Movement Control Training
echo   - myvenv\                           : Python Virtual Environment
echo.
echo Configuration Files:
echo   - multirotor\scanner_config.json                : APF Algorithm Config
echo   - multirotor\DDPG_Weight\dqn_reward_config.json  : Weight DDPG Config
echo   - multirotor\DQN_Movement\movement_dqn_config.json : Movement DQN Config
echo.
echo Batch Files:
echo   - start.bat                         : Main Menu (Current)
echo   - scripts\Run_System_Fixed_Weights.bat      : Run System (Fixed Weights)
echo   - scripts\Run_System_DDPG_Weights.bat       : Run System (DDPG Weights)
echo   - scripts\Train_DDPG_Weights_Real_Environment.bat   : Train DDPG Weights (Real Env)
echo   - scripts\Train_DQN_Movement_Real_Environment.bat   : Train DQN Movement (Real Env)
echo.
echo Python Environment:
call .venv\Scripts\activate.bat 2>nul
if %ERRORLEVEL% EQU 0 (
    python --version 2>nul
    echo [OK] Virtual environment is ready
) else (
    echo [!] Virtual environment not created, please run run_two_drones.bat first
)
echo.
echo DDPG Models:
if exist "multirotor\DDPG_Weight\models\best_model.zip" (
    echo [OK] Weight APF model trained (best_model.zip)
) else (
    echo [!] Weight APF model not trained, please run option [4] to train
)
if exist "multirotor\DQN_Movement\models\movement_dqn_final.zip" (
    echo [OK] Movement control model trained
) else (
    echo [!] Movement control model not trained, please run option [5] to train
)
echo.
echo ============================================================
echo.
pause
goto menu

:end
cls
echo ============================================================
echo Thank you for using AirSim UAV Simulation System!
echo ============================================================
echo.
timeout /t 2 >nul
exit