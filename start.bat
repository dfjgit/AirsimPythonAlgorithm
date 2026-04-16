@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start_Batch_Bootstrap.ps1" -RepoRoot "%~dp0." -MainBatch "start_main.bat"
exit /b %ERRORLEVEL%
