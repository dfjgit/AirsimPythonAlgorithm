@echo off
setlocal

:: 设置编码为 UTF-8
chcp 65001 > nul

echo ============================================================
echo   分层强化学习 (HRL) 融合训练启动器 [无可视化]
echo   环境: AirSim + Unity
echo   高层: DQN 协同规划器 (5s 周期)
echo   底层: DQN+APF 控制器 (0.5s 周期)
echo ============================================================

:: 检查虚拟环境
if not exist "myvenv\Scripts\python.exe" (
    echo [错误] 未发现虚拟环境，请先运行 setup.bat
    pause
    exit /b 1
)

:: 启动训练（禁用可视化）
echo [状态] 正在启动分层融合训练脚本（无可视化模式）...
.\myvenv\Scripts\python.exe multirotor\DQN_Movement\scripts\train_hierarchical_with_airsim.py --no-visualization

if %ERRORLEVEL% neq 0 (
    echo [错误] 训练意外中断
) else (
    echo [成功] 训练完成
)

pause
