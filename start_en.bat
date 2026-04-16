@echo off
chcp 65001 >nul
setlocal EnableExtensions
set "AIRSIM_UI_LANG=en"
if not defined AIRSIM_RUNTIME_LOG_MODE set "AIRSIM_RUNTIME_LOG_MODE=user"
set "START_PYTHON_EXE=python"
if exist "%~dp0myvenv\Scripts\python.exe" set "START_PYTHON_EXE=%~dp0myvenv\Scripts\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set "START_PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
:menu
cls
echo ============================================================
echo    AirSim UAV Simulation Platform - Console
echo ============================================================
echo.
echo === System Operations ===
echo   [1] Launch System (Fixed Weights)
echo   [2] Launch System (DDPG Weight Prediction)
echo   [3] Launch System (DQN Control, Reserved)
echo.
echo === DDPG+APF Training ===
echo   [4] Launch DDPG+APF Training (AirSim, New Model)
echo   [E] Run DDPG+APF Training (Single-Episode Crazyflie Online)
echo.
echo === DQN Control Training ===
echo   [5] Launch DQN Control Training (AirSim, New Model)
echo   [H] Launch Hierarchical DQN Training (Offline / Mock)
echo   [F] Launch Hierarchical DQN Training (AirSim Fusion)
echo   [D] Validate DQN Control Model
echo.
echo === Analysis ===
echo   [A] Generate Visualization Outputs
echo   [B] Generate DDPG vs DQN Comparison
echo.
echo === Experiment Workflows ===
echo   [M] Four-Group Unified Simulation Comparison
echo   [N] Virtual-Real Two-Stage Workflow
echo.
echo === Four-Group Experiments ===
echo   [G] Run Four-Group Unified Simulation Comparison / Benchmark
echo   [I] Generate Four-Group Main Analysis
echo   [J] Generate Family Comparison Analysis
echo   [K] Run Paper Multi-Seed Training (DDPG+APF)
echo   [L] Run Paper Multi-Seed Training (Pure DQN)
echo.
echo === Maintenance ===
echo   [C] Clean Up Training and Analysis Outputs
echo.
echo === Platform Information ===
echo   [6] View Platform Information
if /i "%AIRSIM_RUNTIME_LOG_MODE%"=="detail" (
echo   Current Runtime Log Mode: Detail Mode
) else (
echo   Current Runtime Log Mode: User Mode
)
echo   [T] Toggle Runtime Log Mode (Current Session)
echo   [0] Exit
echo.
echo ============================================================
echo.

set /p choice=Select an option (0-6,A-N,C,D,E,F,G,H,I,J,K,L,M,T):

if /i "%choice%"=="1" goto run_normal
if /i "%choice%"=="2" goto run_dqn
if /i "%choice%"=="3" goto run_dqn_movement
if /i "%choice%"=="4" goto train_weight_airsim
if /i "%choice%"=="E" goto train_weight_crazyflie_online_single
if /i "%choice%"=="5" goto train_movement_airsim
if /i "%choice%"=="H" goto train_hierarchical_dqn
if /i "%choice%"=="F" goto train_hierarchical_airsim
if /i "%choice%"=="d" goto test_movement_dqn
if /i "%choice%"=="a" goto data_visualization
if /i "%choice%"=="b" goto compare_algorithms
if /i "%choice%"=="g" goto four_group_benchmark
if /i "%choice%"=="i" goto analyze_four_group_benchmark
if /i "%choice%"=="j" goto analyze_family_comparisons
if /i "%choice%"=="k" goto train_paper_ddpg_seeds
if /i "%choice%"=="l" goto train_paper_dqn_seeds
if /i "%choice%"=="m" goto comparison_workflow
if /i "%choice%"=="n" goto virtual_real_two_stage_workflow
if /i "%choice%"=="t" goto toggle_runtime_log_mode
if /i "%choice%"=="c" goto cleanup_menu
if /i "%choice%"=="6" goto info
if /i "%choice%"=="0" goto end

echo.
echo Invalid selection. Please try again.
timeout /t 2 >nul
goto menu

:toggle_runtime_log_mode
if /i "%AIRSIM_RUNTIME_LOG_MODE%"=="detail" (
    set "AIRSIM_RUNTIME_LOG_MODE=user"
    echo.
    echo Switched to User Mode.
) else (
    set "AIRSIM_RUNTIME_LOG_MODE=detail"
    echo.
    echo Switched to Detail Mode.
)
echo.
call :wait_for_continue
if /i "%AIRSIM_TEST_EXIT_AFTER_TOGGLE%"=="1" goto end
goto menu

:wait_for_continue
if /i "%AIRSIM_TEST_NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0

:clear_quick_overrides
for %%V in (
    AIRSIM_QUICK_DRONES
    AIRSIM_QUICK_DDPG_TIMESTEPS
    AIRSIM_QUICK_DQN_TIMESTEPS
    AIRSIM_QUICK_HRL_TIMESTEPS
    AIRSIM_QUICK_APF_BASELINE_EPISODES
    AIRSIM_QUICK_BENCHMARK_EPISODES
    AIRSIM_QUICK_VISUALIZATION
    AIRSIM_QUICK_SEEDS
) do set "%%V="
exit /b 0

:collect_quick_config
set "QC_PROFILE=%~1"
if "%QC_PROFILE%"=="" exit /b 0
if /i "%AIRSIM_TEST_SKIP_QUICK_CONFIG%"=="1" exit /b 0
call :clear_quick_overrides
set "QC_FILE=%TEMP%\airsim_quick_config_%RANDOM%_%RANDOM%.env"
if exist "%QC_FILE%" del /f /q "%QC_FILE%" >nul 2>nul
"%START_PYTHON_EXE%" "%~dp0scripts\start_quick_config_helper.py" --schema "%~dp0scripts\start_quick_config_schema.json" --profile "%QC_PROFILE%" --output "%QC_FILE%" --lang en
set "QC_EXIT=%ERRORLEVEL%"
if exist "%QC_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%QC_FILE%") do set "%%A=%%B"
    del /f /q "%QC_FILE%" >nul 2>nul
)
exit /b %QC_EXIT%

:query_latest_workflow
set "WORKFLOW_QUERY_TYPE=%~1"
set "WORKFLOW_QUERY_RESULT="
set "WORKFLOW_QUERY_STATUS="
set "WORKFLOW_QUERY_PHASE="
set "WORKFLOW_QUERY_FILE=%TEMP%\airsim_workflow_query_%RANDOM%_%RANDOM%.txt"
if "%WORKFLOW_QUERY_TYPE%"=="" exit /b 0
if exist "%WORKFLOW_QUERY_FILE%" del /f /q "%WORKFLOW_QUERY_FILE%" >nul 2>nul
if defined AIRSIM_WORKFLOW_WORKSPACE_ROOT (
    <nul "%START_PYTHON_EXE%" "%~dp0multirotor\Algorithm\paper_workflow_orchestrator.py" --workflow %WORKFLOW_QUERY_TYPE% --query-latest-resumable --workspace-root "%AIRSIM_WORKFLOW_WORKSPACE_ROOT%" > "%WORKFLOW_QUERY_FILE%"
) else (
    <nul "%START_PYTHON_EXE%" "%~dp0multirotor\Algorithm\paper_workflow_orchestrator.py" --workflow %WORKFLOW_QUERY_TYPE% --query-latest-resumable > "%WORKFLOW_QUERY_FILE%"
)
if exist "%WORKFLOW_QUERY_FILE%" (
    for /f "usebackq tokens=1,2,* delims=|" %%A in ("%WORKFLOW_QUERY_FILE%") do (
        set "WORKFLOW_QUERY_RESULT=%%A"
        set "WORKFLOW_QUERY_STATUS=%%B"
        set "WORKFLOW_QUERY_PHASE=%%C"
    )
    del /f /q "%WORKFLOW_QUERY_FILE%" >nul 2>nul
)
exit /b 0

:run_paper_workflow
set "WORKFLOW_RUN_TYPE=%~1"
set "WORKFLOW_RUN_MODE=%~2"
if "%WORKFLOW_RUN_TYPE%"=="" exit /b 1
if defined AIRSIM_TEST_PAPER_WORKFLOW_CAPTURE_FILE (
    >> "%AIRSIM_TEST_PAPER_WORKFLOW_CAPTURE_FILE%" echo %WORKFLOW_RUN_TYPE%^|%WORKFLOW_RUN_MODE%
    exit /b 0
)
if /i "%WORKFLOW_RUN_MODE%"=="resume" (
    if defined AIRSIM_WORKFLOW_WORKSPACE_ROOT (
        call scripts\Run_Paper_Workflow.bat --workflow %WORKFLOW_RUN_TYPE% --resume-latest --workspace-root "%AIRSIM_WORKFLOW_WORKSPACE_ROOT%"
    ) else (
        call scripts\Run_Paper_Workflow.bat --workflow %WORKFLOW_RUN_TYPE% --resume-latest
    )
) else (
    if defined AIRSIM_WORKFLOW_WORKSPACE_ROOT (
        call scripts\Run_Paper_Workflow.bat --workflow %WORKFLOW_RUN_TYPE% --workspace-root "%AIRSIM_WORKFLOW_WORKSPACE_ROOT%"
    ) else (
        call scripts\Run_Paper_Workflow.bat --workflow %WORKFLOW_RUN_TYPE%
    )
)
exit /b %ERRORLEVEL%

:prompt_workflow_action
set /p workflow_action=Select an action ^(C/N/Q^):
exit /b 0

:select_workflow_action
if defined AIRSIM_TEST_WORKFLOW_ACTION (
    set "workflow_action=%AIRSIM_TEST_WORKFLOW_ACTION%"
) else (
    call :prompt_workflow_action
)
exit /b 0

:resolve_workflow_mode_en
set "WORKFLOW_MODE=new"
set "workflow_action="
set "workflow_confirm="
call :query_latest_workflow "%~1"
if not defined WORKFLOW_QUERY_RESULT goto resolve_workflow_mode_en_confirm
echo Unfinished workflow detected:
echo   Path: %WORKFLOW_QUERY_RESULT%
if defined WORKFLOW_QUERY_STATUS echo   Status: %WORKFLOW_QUERY_STATUS%
if defined WORKFLOW_QUERY_PHASE echo   Current phase: %WORKFLOW_QUERY_PHASE%
echo.
echo Available actions:
echo   [C] Resume the latest workflow
echo   [N] Start a new workflow from scratch
echo   [Q] Return to the main menu
call :select_workflow_action
if /i "%workflow_action%"=="Q" exit /b 1
if /i "%workflow_action%"=="C" set "WORKFLOW_MODE=resume"
if /i "%workflow_action%"=="C" exit /b 0
if /i "%workflow_action%"=="N" set "WORKFLOW_MODE=new"
if /i "%workflow_action%"=="N" exit /b 0
echo.
echo Invalid selection. Please try again.
timeout /t 2 >nul
exit /b 2

:resolve_workflow_mode_en_confirm
set /p workflow_confirm=Type Y to continue, or any other key to return to the main menu:
if /i not "%workflow_confirm%"=="Y" exit /b 1
exit /b 0

:run_normal
cls
echo ============================================================
echo System Operations (Fixed Weights)
echo ============================================================
echo.
call :collect_quick_config "run_fixed"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Run_System_Fixed_Weights.bat
goto menu

:run_dqn
cls
echo ============================================================
echo System Operations (DDPG Weight Prediction)
echo ============================================================
echo.
call :collect_quick_config "run_ddpg"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Run_System_DDPG_Weights.bat
goto menu

:run_dqn_movement
cls
echo ============================================================
echo System Operations (DQN Control)
echo ============================================================
echo.
echo [Tip] This entry is not yet available. Please use the training or evaluation options instead.
echo.
pause
goto menu

:train_weight_airsim
cls
echo ============================================================
echo DDPG+APF Training (AirSim, New Model)
echo ============================================================
echo.
call :collect_quick_config "ddpg_train"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Train_DDPG_Weights_Real_Environment.bat
goto menu

:train_weight_crazyflie_online_single
cls
echo ============================================================
echo DDPG+APF Training (Single-Episode Crazyflie Online)
echo ============================================================
echo.
call :collect_quick_config "ddpg_single_episode"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Train_DDPG_Weights_Crazyflie_Online_Single_Episode_EN.bat
goto menu

:train_movement_airsim
cls
echo ============================================================
echo DQN Control Training (AirSim, New Model)
echo ============================================================
echo.
call :collect_quick_config "dqn_train"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Train_DQN_Movement_Real_Environment.bat
goto menu

:train_hierarchical_dqn
cls
echo ============================================================
echo Hierarchical DQN Training (Offline / Mock)
echo ============================================================
echo.
call :collect_quick_config "hrl_train"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Train_Hierarchical_DQN.bat
goto menu

:train_hierarchical_airsim
cls
echo ============================================================
echo Hierarchical DQN Training (AirSim Fusion)
echo ============================================================
echo.
call :collect_quick_config "hrl_train"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Train_Hierarchical_With_AirSim.bat
goto menu

:test_movement_dqn
cls
echo ============================================================
echo DQN Control Model Validation
echo ============================================================
echo.
call scripts\Test_DQN_Movement.bat
goto menu

:data_visualization
cls
echo ============================================================
echo Visualization Output Generation
echo ============================================================
echo.
call scripts\Data_Visualization_Analysis.bat
goto menu

:compare_algorithms
cls
echo ============================================================
echo DDPG vs DQN Comparison
echo ============================================================
echo.
echo [Tip] This task generates a comparison report from the available training outputs.
echo.
if not exist "myvenv\Scripts\activate.bat" (
    echo [Error] Python virtual environment was not found.
    pause
    goto menu
)
call myvenv\Scripts\activate.bat
python "multirotor\Algorithm\visualize_training_data.py" --auto --compare-algorithms --out analysis_results
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [Success] Comparison analysis completed.
    echo [Output] See analysis_results\algorithm_comparison_ddpg_vs_dqn\
) else (
    echo.
    echo [Failed] Comparison analysis did not complete. Please verify the available training data.
)
echo.
pause
goto menu

:comparison_workflow
cls
echo ============================================================
echo Four-Group Unified Simulation Comparison
echo ============================================================
echo.
echo This workflow runs the following stages in order:
echo   [1] APF baseline multi-episode simulation (fixed APF / random APF)
echo   [2] DDPG+APF stage01 training
echo   [3] Pure DQN stage01 training
echo   [4] Final four-group unified simulation benchmark
echo   [5] Comparison analysis and stage02 recommendation
echo.
echo Notes:
echo   - fixed APF and random APF do not enter a training stage.
echo   - They first run multi-episode baseline simulation before the final unified comparison.
echo   - Configurable items:
echo     * Drone count (default 3)
echo     * APF baseline simulation episodes
echo     * Benchmark episodes per seed
echo     * DDPG training steps
echo     * DQN training steps
echo.
call :resolve_workflow_mode_en "comparison"
if errorlevel 2 goto comparison_workflow
if errorlevel 1 goto menu
echo.
call :collect_quick_config "comparison_workflow"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call :run_paper_workflow "comparison" "%WORKFLOW_MODE%"
set "WORKFLOW_EXIT=%ERRORLEVEL%"
if not "%WORKFLOW_EXIT%"=="0" (
    echo.
    echo [Error] Four-Group Unified Simulation Comparison failed. Exit code: %WORKFLOW_EXIT%
    echo [Tip] Please review the first error shown above before trying again.
    echo.
    pause
)
if /i "%AIRSIM_TEST_EXIT_AFTER_WORKFLOW%"=="1" goto end
goto menu

:virtual_real_two_stage_workflow
cls
echo ============================================================
echo Virtual-Real Two-Stage Workflow
echo ============================================================
echo.
echo Quick-configurable items:
echo   * Drone count (default 3)
echo   * DDPG training steps (simulation-time estimate shown)
echo.
call :resolve_workflow_mode_en "virtual_real_two_stage"
if errorlevel 2 goto virtual_real_two_stage_workflow
if errorlevel 1 goto menu
call :collect_quick_config "two_stage_workflow"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call :run_paper_workflow "virtual_real_two_stage" "%WORKFLOW_MODE%"
set "WORKFLOW_EXIT=%ERRORLEVEL%"
if not "%WORKFLOW_EXIT%"=="0" (
    echo.
    echo [Error] Virtual-Real Two-Stage Workflow failed. Exit code: %WORKFLOW_EXIT%
    echo [Tip] Please review the first error shown above before trying again.
    echo.
    pause
)
if /i "%AIRSIM_TEST_EXIT_AFTER_WORKFLOW%"=="1" goto end
goto menu

:four_group_benchmark
cls
echo ============================================================
echo Four-Group Unified Simulation Comparison / Benchmark
echo ============================================================
echo.
call :collect_quick_config "four_group_benchmark"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Run_Four_Group_Benchmark.bat
goto menu

:analyze_four_group_benchmark
cls
echo ============================================================
echo Four-Group Main Analysis
echo ============================================================
echo.
call scripts\Analyze_Four_Group_Benchmark.bat
goto menu

:analyze_family_comparisons
cls
echo ============================================================
echo Family Comparison Analysis
echo ============================================================
echo.
call scripts\Analyze_Family_Comparisons.bat
goto menu

:train_paper_ddpg_seeds
cls
echo ============================================================
echo Paper Multi-Seed Training (DDPG+APF)
echo ============================================================
echo.
call :collect_quick_config "paper_ddpg_seeds"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
powershell -ExecutionPolicy Bypass -File "scripts\Run_Paper_Training_Seeds.ps1" -Algorithm ddpg_apf
goto menu

:train_paper_dqn_seeds
cls
echo ============================================================
echo Paper Multi-Seed Training (Pure DQN)
echo ============================================================
echo.
call :collect_quick_config "paper_dqn_seeds"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
powershell -ExecutionPolicy Bypass -File "scripts\Run_Paper_Training_Seeds.ps1" -Algorithm pure_dqn
goto menu

:cleanup_menu
cls
echo ============================================================
echo Maintenance - Output Cleanup
echo ============================================================
echo.
echo [1] Clean Up DDPG Model Files
echo [2] Clean Up DDPG Training Logs
echo [3] Clean Up DQN Model Files
echo [4] Clean Up DQN Training Logs
echo [5] Clean Up Analysis Outputs
echo.
echo [8] Clean Up All Outputs Above
echo [9] Back to Main Menu
echo.
echo ============================================================
echo.
set /p cleanup_choice=Select a maintenance option (1-5, 8-9):

if "%cleanup_choice%"=="1" goto cleanup_ddpg_models
if "%cleanup_choice%"=="2" goto cleanup_ddpg_logs
if "%cleanup_choice%"=="3" goto cleanup_dqn_models
if "%cleanup_choice%"=="4" goto cleanup_dqn_logs
if "%cleanup_choice%"=="5" goto cleanup_analysis_results
if "%cleanup_choice%"=="8" goto cleanup_all
if "%cleanup_choice%"=="9" goto menu

echo.
echo Invalid selection. Please try again.
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
cls
echo ============================================================
echo Preparing to delete: DDPG Training Logs
echo ============================================================
echo.
echo Will delete:
echo   - multirotor\DDPG_Weight\logs (visualization logs)
echo   - multirotor\DDPG_Weight\airsim_training_logs (AirSim training data)
echo   - multirotor\DDPG_Weight\crazyflie_logs (Crazyflie training data)
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
echo.
echo Deleting DDPG logs...
rmdir /s /q "multirotor\DDPG_Weight\logs" 2>nul
rmdir /s /q "multirotor\DDPG_Weight\airsim_training_logs" 2>nul
rmdir /s /q "multirotor\DDPG_Weight\crazyflie_logs" 2>nul
echo.
echo [Success] DDPG logs deleted.
echo.
pause
goto cleanup_menu

:cleanup_dqn_models
call :confirm_delete "multirotor\DQN_Movement\models" "DQN Models"

:cleanup_dqn_logs
cls
echo ============================================================
echo Preparing to delete: DQN Training Logs
echo ============================================================
echo.
echo Will delete:
echo   - multirotor\DQN_Movement\logs (main logs)
echo   - multirotor\DQN_Movement\logs\dqn_scan_data (scan training data)
echo   - multirotor\DQN_Movement\scripts\logs (HRL training data)
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
echo.
echo Deleting DQN logs...
rmdir /s /q "multirotor\DQN_Movement\logs" 2>nul
rmdir /s /q "multirotor\DQN_Movement\scripts\logs" 2>nul
echo.
echo [Success] DQN logs deleted.
echo.
pause
goto cleanup_menu

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
echo   - multirotor\DDPG_Weight\airsim_training_logs
echo   - multirotor\DDPG_Weight\crazyflie_logs
echo   - multirotor\DQN_Movement\models
echo   - multirotor\DQN_Movement\logs
echo   - multirotor\DQN_Movement\scripts\logs
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
for %%D in ("multirotor\DDPG_Weight\models" "multirotor\DDPG_Weight\logs" "multirotor\DDPG_Weight\airsim_training_logs" "multirotor\DDPG_Weight\crazyflie_logs" "multirotor\DQN_Movement\models" "multirotor\DQN_Movement\logs" "multirotor\DQN_Movement\scripts\logs" "analysis_results") do (
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
echo    Platform Information
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
echo   - multirotor\apf_algorithm_config.json               : APF Algorithm Config
echo   - multirotor\DDPG_Weight\unified_train_config.json    : Unified Training Config (Recommended)
echo   - multirotor\DDPG_Weight\configs\legacy\*.json       : Legacy Training Config Examples
echo   - multirotor\DQN_Movement\configs\movement_dqn_config.json : Movement DQN Config
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
echo Thank you for using AirSim UAV Simulation Platform!
echo ============================================================
echo.
timeout /t 2 >nul
endlocal
exit /b 0
