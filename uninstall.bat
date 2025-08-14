@echo off
for /f "delims=" %%i in ('pip freeze') do (
    pip uninstall -y %%i
)
