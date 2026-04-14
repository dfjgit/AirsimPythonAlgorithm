@echo off
setlocal

set "ROOT=%~dp0.."
set "PYTHON_EXE=python"
if exist "%ROOT%\myvenv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%\myvenv\Scripts\python.exe"

"%PYTHON_EXE%" "%ROOT%\multirotor\Algorithm\paper_workflow_orchestrator.py" %*
exit /b %ERRORLEVEL%
