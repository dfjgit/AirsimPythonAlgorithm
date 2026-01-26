@echo off
chcp 65001 >nul
cls
echo ============================================================
echo    Training Data Visualization Analysis Tool
echo ============================================================
echo.
echo Features:
echo   - Auto discover and analyze all training data
echo   - Support Crazyflie drone training logs
echo   - Support DataCollector scan data
echo   - Generate comprehensive analysis reports and charts
echo.
echo ============================================================
echo.

:menu
echo.
echo Please select analysis mode:
echo.
echo   [1] Auto analyze all data (Recommended)
echo   [2] Analyze Crazyflie training logs
echo   [3] Analyze scan data
echo   [4] Analyze specific file
echo   [0] Return to main menu
echo.
set /p choice=Enter option (0-4): 

if "%choice%"=="1" goto auto_analyze
if "%choice%"=="2" goto analyze_crazyflie
if "%choice%"=="3" goto analyze_scan
if "%choice%"=="4" goto analyze_file
if "%choice%"=="0" goto end

echo.
echo Invalid option, please try again
timeout /t 2 >nul
goto menu

:auto_analyze
cls
echo ============================================================
echo Auto Analyze All Data
echo ============================================================
echo.
echo Scanning data directories...
echo.

cd /d "%~dp0.."
call myvenv\Scripts\activate.bat

python multirotor\Algorithm\visualize_training_data.py --auto --out analysis_results

echo.
echo ============================================================
echo Analysis completed!
echo Results saved in: %cd%\analysis_results
echo ============================================================
echo.

REM Ask if user wants to open result directory
set /p open_dir=Open result directory? (y/n): 
if /i "%open_dir%"=="y" (
    start "" "%cd%\analysis_results"
)

pause
goto menu

:analyze_crazyflie
cls
echo ============================================================
echo Analyze Crazyflie Training Logs
echo ============================================================
echo.
echo Analyzing Crazyflie training data...
echo Directory: multirotor\DDPG_Weight\crazyflie_logs
echo.

cd /d "%~dp0.."
call myvenv\Scripts\activate.bat

python multirotor\Algorithm\visualize_training_data.py --dir multirotor\DDPG_Weight\crazyflie_logs --out analysis_results

echo.
echo ============================================================
echo Analysis completed!
echo Results saved in: %cd%\analysis_results
echo ============================================================
echo.

REM Ask if user wants to open result directory
set /p open_dir=Open result directory? (y/n): 
if /i "%open_dir%"=="y" (
    start "" "%cd%\analysis_results"
)

pause
goto menu

:analyze_scan
cls
echo ============================================================
echo Analyze Scan Data
echo ============================================================
echo.
echo Analyzing scan data...
echo Directory: multirotor\DDPG_Weight\airsim_training_logs
echo.

cd /d "%~dp0.."
call myvenv\Scripts\activate.bat

python multirotor\Algorithm\visualize_scan_csv.py --csv-dir multirotor\DDPG_Weight\airsim_training_logs --out analysis_results

echo.
echo ============================================================
echo Analysis completed!
echo Results saved in: %cd%\analysis_results
echo ============================================================
echo.

REM Ask if user wants to open result directory
set /p open_dir=Open result directory? (y/n): 
if /i "%open_dir%"=="y" (
    start "" "%cd%\analysis_results"
)

pause
goto menu

:analyze_file
cls
echo ============================================================
echo Analyze Specific File
echo ============================================================
echo.
echo Please drag and drop file to this window, or enter file path:
echo Supported formats: .json, .csv
echo.
set /p file_path=File path: 

if "%file_path%"=="" (
    echo.
    echo No file path entered
    timeout /t 2 >nul
    goto menu
)

REM Remove quotes
set file_path=%file_path:"=%

REM Check if file exists
if not exist "%file_path%" (
    echo.
    echo File not found: %file_path%
    pause
    goto menu
)

echo.
echo Analyzing file...
echo.

cd /d "%~dp0.."
call myvenv\Scripts\activate.bat

REM Determine file type
echo %file_path% | findstr /i ".json" >nul
if %ERRORLEVEL% EQU 0 (
    python multirotor\Algorithm\visualize_training_data.py --json "%file_path%" --out analysis_results
) else (
    python multirotor\Algorithm\visualize_training_data.py --csv "%file_path%" --out analysis_results
)

echo.
echo ============================================================
echo Analysis completed!
echo Results saved in: %cd%\analysis_results
echo ============================================================
echo.

REM Ask if user wants to open result directory
set /p open_dir=Open result directory? (y/n): 
if /i "%open_dir%"=="y" (
    start "" "%cd%\analysis_results"
)

pause
goto menu

:end
echo.
echo Returning to main menu...
timeout /t 1 >nul
exit /b
