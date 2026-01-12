@echo off
chcp 65001 >nul 2>&1
cls

echo ============================================================
echo Start System - Fixed Weights Mode
echo ============================================================
echo.

REM Activate virtual environment
echo [1/2] Activating Python virtual environment...
if exist "%~dp0..\myvenv\Scripts\activate.bat" (
    call "%~dp0..\myvenv\Scripts\activate.bat"
    echo [OK] Virtual environment activated successfully
) else (
    echo [!] Virtual environment does not exist, using system Python
)
echo.

REM Display configuration information
echo ============================================================
echo Running Configuration
echo ============================================================
echo Mode: Fixed Weights
echo Number of Drones: 3
echo ============================================================
echo.

REM Key modification: First switch to the directory where AlgorithmServer.py is located
echo [2/2] Switching to script directory and starting algorithm server....
echo Debug: Batch file path (%%~dp0): %~dp0
echo Debug: Target directory: "%~dp0..\multirotor"

cd /d "%~dp0..\multirotor"
if %errorlevel% neq 0 (
    echo [Error] Failed to switch to multirotor directory!
    pause
    exit /b 1
)
echo Debug: Current directory after switching: %cd%
 
REM Run directly with script name (now in multirotor directory)
python AlgorithmServer.py --drones 3

echo.
echo ============================================================
echo Program exited
echo ============================================================
echo.
pause