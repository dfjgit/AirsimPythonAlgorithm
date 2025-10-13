@echo off
chcp 65001 >nul
echo ============================================================
echo 启动双无人机系统
echo ============================================================
echo.

REM 激活虚拟环境
echo 正在激活Python虚拟环境...
call .venv\Scripts\activate.bat

REM 运行算法服务器
echo 启动算法服务器...
python multirotor\AlgorithmServer.py

pause

