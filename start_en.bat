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
echo   [5] Train DQN Movement (Real AirSim Environment)[⭐Available!]
echo   [H] Train Hierarchical DQN (Offline/Mock Mode)
echo   [F] Train Hierarchical DQN (AirSim Fusion Mode) [⭐New!]
echo   [D] Test DQN Movement Model [⭐Available!]
echo.
echo === Data Analysis ===
echo   [A] Data Visualization Analysis [⭐Available!]
echo   [B] DDPG vs DQN Algorithm Comparison [⭐Available!]
echo.
echo === Cleanup ===
echo   [C] Delete training outputs (models/logs/analysis)
echo.
echo === System Information ===
echo   [6] View System Information
echo   [0] Exit
echo.
echo ============================================================
echo.

set /p choice=Please enter an option (0-6,A-C): 

if /i "%choice%"=="1" goto run_normal
if /i "%choice%"=="2" goto run_dqn
if /i "%choice%"=="3" goto run_dqn_movement
if /i "%choice%"=="4" goto train_weight_airsim
if /i "%choice%"=="5" goto train_movement_airsim
if /i "%choice%"=="H" goto train_hierarchical_dqn
if /i "%choice%"=="F" goto train_hierarchical_airsim
if /i "%choice%"=="d" goto test_movement_dqn
if /i "%choice%"=="a" goto data_visualization
if /i "%choice%"=="b" goto compare_algorithms
if /i "%choice%"=="c" goto cleanup_menu
if /i "%choice%"=="6" goto info
if /i "%choice%"=="0" goto end

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

:train_hierarchical_dqn
cls
echo ============================================================
echo Hierarchical Reinforcement Learning (HRL) - Offline Mode
echo ============================================================
echo.
call scripts\Train_Hierarchical_DQN.bat
goto menu

:train_hierarchical_airsim
cls
echo ============================================================
echo Hierarchical Reinforcement Learning (HRL) - AirSim Fusion Mode
echo ============================================================
echo.
call scripts\Train_Hierarchical_With_AirSim.bat
goto menu

:test_movement_dqn
cls
echo ============================================================
echo Test DQN Movement Model
echo ============================================================
echo.
call scripts\Test_DQN_Movement.bat
goto menu

:data_visualization
cls
echo ============================================================
echo Data Visualization Analysis
echo ============================================================
echo.
call scripts\Data_Visualization_Analysis.bat
goto menu

:compare_algorithms
cls
echo ============================================================
echo DDPG vs DQN Algorithm Comparison Analysis
echo ============================================================
echo.
echo [Tip] This feature will compare the training effects of DDPG and DQN algorithms
echo.
if not exist "myvenv\Scripts\activate.bat" (
    echo [Error] Python virtual environment does not exist
    pause
    goto menu
)
call myvenv\Scripts\activate.bat
python "multirotor\Algorithm\visualize_training_data.py" --auto --compare-algorithms --out analysis_results
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [Success] Algorithm comparison analysis completed!
    echo [Results] Please check analysis_results\algorithm_comparison_ddpg_vs_dqn\ directory
) else (
    echo.
    echo [Failed] Algorithm comparison analysis failed, please check if there is enough training data
)
echo.
pause
goto menu

:cleanup_menu
cls
echo ============================================================
echo Delete Training Outputs (Models/Logs/Analysis)
echo ============================================================
echo.
echo [1] Delete DDPG Models (multirotor\DDPG_Weight\models)
echo [2] Delete DDPG Logs (multirotor\DDPG_Weight\logs)
echo [3] Delete DQN Models (multirotor\DQN_Movement\models)
echo [4] Delete DQN Logs (multirotor\DQN_Movement\logs)
echo [5] Delete Analysis Results (analysis_results)
echo.
echo [8] Delete ALL of the above
echo [9] Back to Main Menu
echo.
echo ============================================================
echo.
set /p cleanup_choice=Please enter an option (1-5, 8-9): 

if "%cleanup_choice%"=="1" goto cleanup_ddpg_models
if "%cleanup_choice%"=="2" goto cleanup_ddpg_logs
if "%cleanup_choice%"=="3" goto cleanup_dqn_models
if "%cleanup_choice%"=="4" goto cleanup_dqn_logs
if "%cleanup_choice%"=="5" goto cleanup_analysis_results
if "%cleanup_choice%"=="8" goto cleanup_all
if "%cleanup_choice%"=="9" goto menu

echo.
echo Invalid option, please select again
timeout /t 2 >nul
goto cleanup_menu

:confirm_delete
set "TARGET_DIR=%~1"
set "TARGET_DESC=%~2"
if "%TARGET_DIR%"=="" goto cleanup_menu
if "%TARGET_DESC%"=="" set "TARGET_DESC=%TARGET_DIR%"
cls
echo ============================================================
echo Preparing to delete: %TARGET_DESC%
echo Directory: %TARGET_DIR%
echo ============================================================
echo.
echo [WARNING] This action is irreversible!
echo.
echo Type YES to confirm deletion, any other input to cancel:
echo.
set /p confirm=Confirmation: 
if /i not "%confirm%"=="YES" (
    echo.
    echo Deletion cancelled.
    timeout /t 2 >nul
    goto cleanup_menu
)
if not exist "%TARGET_DIR%" (
    echo.
    echo [Tip] Directory does not exist, no need to delete.
    timeout /t 2 >nul
    goto cleanup_menu
)
rmdir /s /q "%TARGET_DIR%" 2>nul
if exist "%TARGET_DIR%" (
    echo.
    echo [Failed] Deletion failed, check if files are in use or for permissions.
) else (
    echo.
    echo [Success] Deleted successfully.
)
echo.
pause
goto cleanup_menu

:cleanup_ddpg_models
call :confirm_delete "multirotor\DDPG_Weight\models" "DDPG Models"

:cleanup_ddpg_logs
call :confirm_delete "multirotor\DDPG_Weight\logs" "DDPG Logs"

:cleanup_dqn_models
call :confirm_delete "multirotor\DQN_Movement\models" "DQN Models"

:cleanup_dqn_logs
call :confirm_delete "multirotor\DQN_Movement\logs" "DQN Logs"

:cleanup_analysis_results
call :confirm_delete "analysis_results" "Analysis Results"

:cleanup_all
cls
echo ============================================================
echo Preparing to delete ALL training outputs
echo ============================================================
echo.
echo Will delete:
echo   - multirotor\DDPG_Weight\models
echo   - multirotor\DDPG_Weight\logs
echo   - multirotor\DQN_Movement\models
echo   - multirotor\DQN_Movement\logs
echo   - analysis_results
echo.
echo Type YES to confirm deletion, any other input to cancel:
echo.
set /p confirm_all=Confirmation: 
if /i not "%confirm_all%"=="YES" (
    echo.
    echo Deletion cancelled.
    timeout /t 2 >nul
    goto cleanup_menu
)
for %%D in ("multirotor\DDPG_Weight\models" "multirotor\DDPG_Weight\logs" "multirotor\DQN_Movement\models" "multirotor\DQN_Movement\logs" "analysis_results") do (
    if exist "%%~D" (
        rmdir /s /q "%%~D" 2>nul
    )
)
echo.
echo Deletion executed (missing directories were skipped).
echo.
pause
goto cleanup_menu

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
echo   - multirotor\scanner_config.json                     : APF Algorithm Config
echo   - multirotor\DDPG_Weight\unified_train_config.json    : Unified Training Config (Recommended)
echo   - multirotor\DDPG_Weight\*_train_config*.json        : Old Training Configs (Compatible)
echo   - multirotor\DQN_Movement\movement_dqn_config.json    : Movement DQN Config
echo.
echo Batch Files:
echo   - start.bat                         : Main Menu (Current)
echo   - scripts\Run_System_Fixed_Weights.bat      : Run System (Fixed Weights)
echo   - scripts\Run_System_DDPG_Weights.bat       : Run System (DDPG Weights)
echo   - scripts\Train_DDPG_Weights_Real_Environment.bat   : Train DDPG Weights (Real Env)
echo   - scripts\Train_DQN_Movement_Real_Environment.bat   : Train DQN Movement (Real Env)
echo   - scripts\Train_Hierarchical_DQN.bat                : Train Hierarchical DQN (HL+LL)
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