@echo off
chcp 65001 >nul
cls
echo ============================================================
echo    Training Data Visualization Analysis Tool
echo ============================================================
echo.
echo This tool analyzes training logs and generates visualization charts.
echo Charts will be saved to logs^/analyze_log^/ folder.
echo.
echo ============================================================
echo.

:main_menu
echo.
echo Select analysis mode:
echo.
echo   [1] Analyze Training Data by Algorithm
echo   [0] Return to main menu
echo.
set /p choice=Enter option (0-1):

if "%choice%"=="1" goto algorithm_menu
if "%choice%"=="0" goto end

echo.
echo Invalid option, please try again
timeout /t 2 >nul
goto main_menu

:algorithm_menu
cls
echo ============================================================
echo    Select Algorithm to Analyze
echo ============================================================
echo.
echo   [1] DDPG Weight (AirSim Environment)
echo   [2] DDPG Weight (Crazyflie Environment)
echo   [3] DQN Movement Control
echo   [4] Hierarchical DQN (HRL)
echo.
echo   [0] Return to previous menu
echo.
echo ============================================================
echo.
set /p algo_choice=Enter option (0-4):

if "%algo_choice%"=="1" goto analyze_ddpg_airsim
if "%algo_choice%"=="2" goto analyze_ddpg_crazyflie
if "%algo_choice%"=="3" goto analyze_dqn_movement
if "%algo_choice%"=="4" goto analyze_hierarchical_dqn
if "%algo_choice%"=="0" goto main_menu

echo.
echo Invalid option, please try again
timeout /t 2 >nul
goto algorithm_menu

REM ==================== DDPG Weight (AirSim) ====================
:analyze_ddpg_airsim
cls
echo ============================================================
echo    DDPG Weight (AirSim) - Training Data Analysis
echo ============================================================
echo.

set "LOG_DIR=multirotor\DDPG_Weight\airsim_training_logs"
set "ANALYZE_DIR=multirotor\DDPG_Weight\airsim_training_logs\analyze_log"
set "ALGO_NAME=DDPG_Weight_AirSim"

call :run_analysis "%LOG_DIR%" "%ANALYZE_DIR%" "%ALGO_NAME%"
goto algorithm_menu

REM ==================== DDPG Weight (Crazyflie) ====================
:analyze_ddpg_crazyflie
cls
echo ============================================================
echo    DDPG Weight (Crazyflie) - Training Data Analysis
echo ============================================================
echo.

set "LOG_DIR=multirotor\DDPG_Weight\crazyflie_logs"
set "ANALYZE_DIR=multirotor\DDPG_Weight\crazyflie_logs\analyze_log"
set "ALGO_NAME=DDPG_Weight_Crazyflie"

call :run_analysis "%LOG_DIR%" "%ANALYZE_DIR%" "%ALGO_NAME%"
goto algorithm_menu

REM ==================== DQN Movement ====================
:analyze_dqn_movement
cls
echo ============================================================
echo    DQN Movement Control - Training Data Analysis
echo ============================================================
echo.

set "LOG_DIR=multirotor\DQN_Movement\logs\dqn_scan_data"
set "ANALYZE_DIR=multirotor\DQN_Movement\logs\dqn_scan_data\analyze_log"
set "ALGO_NAME=DQN_Movement"

call :run_analysis "%LOG_DIR%" "%ANALYZE_DIR%" "%ALGO_NAME%"
goto algorithm_menu

REM ==================== Hierarchical DQN ====================
:analyze_hierarchical_dqn
cls
echo ============================================================
echo    Hierarchical DQN (HRL) - Training Data Analysis
echo ============================================================
echo.

set "LOG_DIR=multirotor\DQN_Movement\scripts\logs\hrl_dqn_airsim"
set "ANALYZE_DIR=multirotor\DQN_Movement\scripts\logs\hrl_dqn_airsim\analyze_log"
set "ALGO_NAME=Hierarchical_DQN"

call :run_analysis "%LOG_DIR%" "%ANALYZE_DIR%" "%ALGO_NAME%"
goto algorithm_menu

REM ==================== Analysis Subroutine ====================
:run_analysis
set "TARGET_LOG_DIR=%~1"
set "TARGET_ANALYZE_DIR=%~2"
set "TARGET_ALGO_NAME=%~3"

cd /d "%~dp0.."

REM Check if log directory exists
if not exist "%TARGET_LOG_DIR%" (
    echo.
    echo [Error] Log directory not found: %TARGET_LOG_DIR%
    echo Please train the model first to generate logs.
    echo.
    pause
    exit /b
)

REM Create analyze_log directory if not exists
if not exist "%TARGET_ANALYZE_DIR%" (
    echo [Info] Creating analyze_log directory...
    mkdir "%TARGET_ANALYZE_DIR%"
)

REM Find latest log file
echo [Info] Scanning for latest training log...
set "LATEST_LOG="

REM Check for CSV files
for /f "delims=" %%F in ('dir /b /o-d "%TARGET_LOG_DIR%\*.csv" 2^>nul') do (
    set "LATEST_LOG=%TARGET_LOG_DIR%\%%F"
    goto :found_csv
)

REM Check for JSON files
for /f "delims=" %%F in ('dir /b /o-d "%TARGET_LOG_DIR%\*.json" 2^>nul') do (
    set "LATEST_LOG=%TARGET_LOG_DIR%\%%F"
    goto :found_json
)

REM Check for event files (TensorBoard)
for /f "delims=" %%F in ('dir /b /o-d "%TARGET_LOG_DIR%\events.*" 2^>nul') do (
    set "LATEST_LOG=%TARGET_LOG_DIR%\%%F"
    goto :found_events
)

echo.
echo [Error] No training log files found in: %TARGET_LOG_DIR%
echo Expected files: *.csv, *.json, or events.*
echo.
pause
exit /b

:found_csv
:found_json
:found_events
echo [Found] Latest log: %LATEST_LOG%
echo.
echo [Info] Starting analysis...
echo.

call myvenv\Scripts\activate.bat

REM Determine file type and run appropriate analysis
echo "%LATEST_LOG%" | findstr /i ".csv" >nul
if %ERRORLEVEL% EQU 0 (
    python multirotor\Algorithm\visualize_training_data.py --csv "%LATEST_LOG%" --out "%TARGET_ANALYZE_DIR%" --show
) else (
    echo "%LATEST_LOG%" | findstr /i ".json" >nul
    if %ERRORLEVEL% EQU 0 (
        python multirotor\Algorithm\visualize_training_data.py --json "%LATEST_LOG%" --out "%TARGET_ANALYZE_DIR%" --show
    ) else (
        python multirotor\Algorithm\visualize_training_data.py --dir "%TARGET_LOG_DIR%" --out "%TARGET_ANALYZE_DIR%" --show
    )
)

echo.
if %ERRORLEVEL% EQU 0 (
    echo ============================================================
    echo [Success] Analysis completed!
    echo.
    echo Results saved in: %TARGET_ANALYZE_DIR%
    echo ============================================================
    echo.
    echo [Reminder] Please review the generated charts:
    echo   1. Transfer comparison charts to a safe location if needed
    echo   2. Consider backing up or deleting raw log files to save space
    echo.
    echo Source log: %LATEST_LOG%
    echo.
    set /p open_dir=Open result directory? (y/n):
    if /i "%open_dir%"=="y" (
        start "" "%TARGET_ANALYZE_DIR%"
    )
) else (
    echo ============================================================
    echo [Failed] Analysis failed. Please check:
    echo   - Log file format is valid
    echo   - Python environment is activated
    echo   - Required packages are installed
    echo ============================================================
)

echo.
pause
exit /b

:end
echo.
echo Returning to main menu...
timeout /t 1 >nul
exit /b
