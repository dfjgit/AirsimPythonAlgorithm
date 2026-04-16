@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start_Menu_ZH.ps1" -RepoRoot "%~dp0."
exit /b %ERRORLEVEL%
