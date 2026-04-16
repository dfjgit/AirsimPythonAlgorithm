@echo off
setlocal

:: 设置编码为 UTF-8
chcp 65001 > nul

if /i "%AIRSIM_UI_LANG%"=="en" (
    echo ============================================================
    echo   Hierarchical DQN AirSim Fusion Training Launcher
    echo   Environment: AirSim + Unity
    echo   High Level: DQN Coordination Planner ^(5s cycle^)
    echo   Low Level : DQN+APF Controller ^(0.5s cycle^)
    echo ============================================================
) else (
    echo ============================================================
    echo   分层强化学习 (HRL) 融合训练启动器
    echo   环境: AirSim + Unity
    echo   高层: DQN 协同规划器 (5s 周期)
    echo   底层: DQN+APF 控制器 (0.5s 周期)
    echo ============================================================
)

:: 检查虚拟环境
if not exist "myvenv\Scripts\python.exe" (
    if /i "%AIRSIM_UI_LANG%"=="en" (
        echo [Error] Virtual environment not found. Please run setup.bat first.
    ) else (
        echo [错误] 未发现虚拟环境，请先运行 setup.bat
    )
    pause
    exit /b 1
)

:: 启动训练
if /i "%AIRSIM_UI_LANG%"=="en" (
    echo [Status] Launching hierarchical AirSim fusion training script...
    echo [Tip] If the visualization window freezes, press Ctrl+C and retry with --no-visualization.
) else (
    echo [状态] 正在启动分层融合训练脚本...
    echo [提示] 如果可视化窗口卡死，可以按 Ctrl+C 中断，然后使用 --no-visualization 参数
)
.\myvenv\Scripts\python.exe multirotor\DQN_Movement\scripts\train_hierarchical_with_airsim.py

if %ERRORLEVEL% neq 0 (
    if /i "%AIRSIM_UI_LANG%"=="en" (
        echo [Error] Training terminated unexpectedly.
    ) else (
        echo [错误] 训练意外中断
    )
) else (
    if /i "%AIRSIM_UI_LANG%"=="en" (
        echo [Success] Training completed.
    ) else (
        echo [成功] 训练完成
    )
)

pause
