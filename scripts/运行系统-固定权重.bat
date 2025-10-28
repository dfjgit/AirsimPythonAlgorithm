@echo off
chcp 65001 >nul
echo 正在激活Python虚拟环境...
call %~dp0..\.venv\Scripts\activate.bat

echo 启动算法服务器...
python %~dp0..\multirotor\AlgorithmServer.py  --drones 4
pause