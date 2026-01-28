@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ================================
echo 测试多无人机 DQN 环境
echo ================================
python test_multi_drone_env.py
pause
