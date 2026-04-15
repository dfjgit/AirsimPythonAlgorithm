@echo off
setlocal

:: 设置编码为 UTF-8
chcp 65001 > nul

echo ============================================================
echo   分层强化学习 (HRL) 训练启动器
echo   高层: DQN 协同规划器 (每 5 秒决策一次)
echo   底层: DQN+APF 局部控制器 (每 0.5 秒决策一次)
echo ============================================================

:: 检查虚拟环境
if not exist "myvenv\Scripts\python.exe" (
    echo [错误] 未发现虚拟环境，请先运行 setup.bat
    pause
    exit /b 1
)

:: 启动训练
echo [状态] 正在启动分层训练脚本...
.\myvenv\Scripts\python.exe multirotor\DQN_Movement\scripts\train_hierarchical_dqn.py

if %ERRORLEVEL% neq 0 (
    echo [错误] 训练意外中断
) else (
    echo [成功] 训练完成
)

pause
