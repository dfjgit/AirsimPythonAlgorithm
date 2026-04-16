@echo off
chcp 65001 >nul
setlocal EnableExtensions
set "AIRSIM_UI_LANG=zh"
if not defined AIRSIM_RUNTIME_LOG_MODE set "AIRSIM_RUNTIME_LOG_MODE=user"
set "START_PYTHON_EXE=python"
if exist "%~dp0myvenv\Scripts\python.exe" set "START_PYTHON_EXE=%~dp0myvenv\Scripts\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set "START_PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

:menu
cls
echo ============================================================
echo    AirSim 无人机仿真平台 - 控制台
echo ============================================================
echo.
echo === 系统运行 ===
echo   [1] 启动系统（固定权重）
echo   [2] 启动系统（DDPG 权重预测）
echo   [3] 启动系统（DQN 控制，预留）
echo.
echo === DDPG+APF 训练 ===
echo   [4] 启动 DDPG+APF 训练（AirSim，新模型）
echo   [5] 继续 DDPG+APF 训练（AirSim）
echo   [6] 执行 DDPG+APF 训练（实体日志离线）
echo   [E] 执行 DDPG+APF 训练（实体无人机单轮在线）
echo.
echo === DQN 控制训练 ===
echo   [7] 启动 DQN 控制训练（AirSim，新模型）
echo   [8] 继续 DQN 控制训练（AirSim）
echo   [R] 重新执行当前 stage02 训练
echo   [H] 启动分层 DQN 训练（离线 / Mock）
echo   [F] 启动分层 DQN 训练（AirSim 融合）
echo   [D] 验证 DQN 控制模型
echo.
echo === 结果分析 ===
echo   [A] 生成可视化分析结果
echo   [B] 生成 DDPG 与 DQN 对比分析
echo.
echo === 实验工作流 ===
echo   [M] 四组统一仿真对比阶段
echo   [N] 虚实两阶段实验工作流 (Virtual-Real Two-Stage Workflow)
echo.
echo === 四组论文实验 ===
echo   [G] 执行四组仿真评测（冻结策略）/ Benchmark
echo   [I] 生成四组主结果分析
echo   [J] 生成 Family 维度对比分析
echo   [K] 执行论文多 Seed 训练（DDPG+APF）
echo   [L] 执行论文多 Seed 训练（Pure DQN）
echo.
echo === 系统维护 ===
echo   [C] 清理训练与分析产出
echo.
echo === 平台信息 ===
echo   [9] 查看平台信息
if /i "%AIRSIM_RUNTIME_LOG_MODE%"=="detail" (
echo   当前运行时日志模式: 详细模式
) else (
echo   当前运行时日志模式: 用户模式
)
echo   [T] 切换运行时日志模式（当前会话）
echo   [0] 退出系统
echo.
echo ============================================================
echo.

set /p choice=请选择功能选项 (0-9,A-N,R,T,E):

if /i "%choice%"=="1" goto run_normal
if /i "%choice%"=="2" goto run_ddpg
if /i "%choice%"=="3" goto run_dqn_movement
if /i "%choice%"=="4" goto train_weight_airsim_fresh
if /i "%choice%"=="5" goto train_weight_airsim_resume
if /i "%choice%"=="6" goto train_weight_crazyflie_logs
if /i "%choice%"=="E" goto train_weight_crazyflie_online_single
if /i "%choice%"=="7" goto train_movement_airsim_fresh
if /i "%choice%"=="8" goto train_movement_airsim_resume
if /i "%choice%"=="R" goto train_movement_airsim_rerun_stage02
if /i "%choice%"=="H" goto train_hierarchical_dqn
if /i "%choice%"=="F" goto train_hierarchical_airsim
if /i "%choice%"=="D" goto test_movement_dqn
if /i "%choice%"=="A" goto data_visualization
if /i "%choice%"=="B" goto compare_algorithms
if /i "%choice%"=="G" goto four_group_benchmark
if /i "%choice%"=="I" goto analyze_four_group_benchmark
if /i "%choice%"=="J" goto analyze_family_comparisons
if /i "%choice%"=="K" goto train_paper_ddpg_seeds
if /i "%choice%"=="L" goto train_paper_dqn_seeds
if /i "%choice%"=="M" goto comparison_workflow
if /i "%choice%"=="N" goto virtual_real_two_stage_workflow
if /i "%choice%"=="T" goto toggle_runtime_log_mode
if /i "%choice%"=="C" goto cleanup_menu
if /i "%choice%"=="9" goto info
if /i "%choice%"=="0" goto end

echo.
echo 当前输入无效，请重新选择。
timeout /t 2 >nul
goto menu

:toggle_runtime_log_mode
if /i "%AIRSIM_RUNTIME_LOG_MODE%"=="detail" (
    set "AIRSIM_RUNTIME_LOG_MODE=user"
    echo.
    echo 已切换到用户模式。
) else (
    set "AIRSIM_RUNTIME_LOG_MODE=detail"
    echo.
    echo 已切换到详细模式。
)
echo.
call :wait_for_continue
if /i "%AIRSIM_TEST_EXIT_AFTER_TOGGLE%"=="1" goto end
goto menu

:wait_for_continue
if /i "%AIRSIM_TEST_NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0

:clear_quick_overrides
for %%V in (
    AIRSIM_QUICK_DRONES
    AIRSIM_QUICK_DDPG_TIMESTEPS
    AIRSIM_QUICK_DQN_TIMESTEPS
    AIRSIM_QUICK_HRL_TIMESTEPS
    AIRSIM_QUICK_APF_BASELINE_EPISODES
    AIRSIM_QUICK_BENCHMARK_EPISODES
    AIRSIM_QUICK_VISUALIZATION
    AIRSIM_QUICK_SEEDS
) do set "%%V="
exit /b 0

:collect_quick_config
set "QC_PROFILE=%~1"
if "%QC_PROFILE%"=="" exit /b 0
if /i "%AIRSIM_TEST_SKIP_QUICK_CONFIG%"=="1" exit /b 0
call :clear_quick_overrides
set "QC_FILE=%TEMP%\airsim_quick_config_%RANDOM%_%RANDOM%.env"
if exist "%QC_FILE%" del /f /q "%QC_FILE%" >nul 2>nul
"%START_PYTHON_EXE%" "%~dp0scripts\start_quick_config_helper.py" --schema "%~dp0scripts\start_quick_config_schema.json" --profile "%QC_PROFILE%" --output "%QC_FILE%" --lang zh
set "QC_EXIT=%ERRORLEVEL%"
if exist "%QC_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%QC_FILE%") do set "%%A=%%B"
    del /f /q "%QC_FILE%" >nul 2>nul
)
exit /b %QC_EXIT%

:query_latest_workflow
set "WORKFLOW_QUERY_TYPE=%~1"
set "WORKFLOW_QUERY_RESULT="
set "WORKFLOW_QUERY_STATUS="
set "WORKFLOW_QUERY_PHASE="
set "WORKFLOW_QUERY_FILE=%TEMP%\airsim_workflow_query_%RANDOM%_%RANDOM%.txt"
if "%WORKFLOW_QUERY_TYPE%"=="" exit /b 0
if exist "%WORKFLOW_QUERY_FILE%" del /f /q "%WORKFLOW_QUERY_FILE%" >nul 2>nul
if defined AIRSIM_WORKFLOW_WORKSPACE_ROOT (
    <nul "%START_PYTHON_EXE%" "%~dp0multirotor\Algorithm\paper_workflow_orchestrator.py" --workflow %WORKFLOW_QUERY_TYPE% --query-latest-resumable --workspace-root "%AIRSIM_WORKFLOW_WORKSPACE_ROOT%" > "%WORKFLOW_QUERY_FILE%"
) else (
    <nul "%START_PYTHON_EXE%" "%~dp0multirotor\Algorithm\paper_workflow_orchestrator.py" --workflow %WORKFLOW_QUERY_TYPE% --query-latest-resumable > "%WORKFLOW_QUERY_FILE%"
)
if exist "%WORKFLOW_QUERY_FILE%" (
    for /f "usebackq tokens=1,2,* delims=|" %%A in ("%WORKFLOW_QUERY_FILE%") do (
        set "WORKFLOW_QUERY_RESULT=%%A"
        set "WORKFLOW_QUERY_STATUS=%%B"
        set "WORKFLOW_QUERY_PHASE=%%C"
    )
    del /f /q "%WORKFLOW_QUERY_FILE%" >nul 2>nul
)
exit /b 0

:run_paper_workflow
set "WORKFLOW_RUN_TYPE=%~1"
set "WORKFLOW_RUN_MODE=%~2"
if "%WORKFLOW_RUN_TYPE%"=="" exit /b 1
if defined AIRSIM_TEST_PAPER_WORKFLOW_CAPTURE_FILE (
    >> "%AIRSIM_TEST_PAPER_WORKFLOW_CAPTURE_FILE%" echo %WORKFLOW_RUN_TYPE%^|%WORKFLOW_RUN_MODE%
    exit /b 0
)
if /i "%WORKFLOW_RUN_MODE%"=="resume" (
    if defined AIRSIM_WORKFLOW_WORKSPACE_ROOT (
        call scripts\Run_Paper_Workflow.bat --workflow %WORKFLOW_RUN_TYPE% --resume-latest --workspace-root "%AIRSIM_WORKFLOW_WORKSPACE_ROOT%"
    ) else (
        call scripts\Run_Paper_Workflow.bat --workflow %WORKFLOW_RUN_TYPE% --resume-latest
    )
) else (
    if defined AIRSIM_WORKFLOW_WORKSPACE_ROOT (
        call scripts\Run_Paper_Workflow.bat --workflow %WORKFLOW_RUN_TYPE% --workspace-root "%AIRSIM_WORKFLOW_WORKSPACE_ROOT%"
    ) else (
        call scripts\Run_Paper_Workflow.bat --workflow %WORKFLOW_RUN_TYPE%
    )
)
exit /b %ERRORLEVEL%

:prompt_workflow_action
set /p workflow_action=请选择操作 ^(C/N/Q^)：
exit /b 0

:select_workflow_action
if defined AIRSIM_TEST_WORKFLOW_ACTION (
    set "workflow_action=%AIRSIM_TEST_WORKFLOW_ACTION%"
) else (
    call :prompt_workflow_action
)
exit /b 0

:show_resume_options_zh
echo 检测到未完成的 workflow：
echo   路径: %WORKFLOW_QUERY_RESULT%
echo   状态: %WORKFLOW_QUERY_STATUS%
echo   当前阶段: %WORKFLOW_QUERY_PHASE%
echo.
echo 可选操作：
echo   [C] 继续当前实验
echo   [N] 新建实验并从头执行
echo   [Q] 返回主菜单
exit /b 0

:confirm_new_workflow_zh
set /p workflow_confirm=请输入 Y 继续执行，输入其它任意键返回主菜单：
exit /b 0

:show_invalid_workflow_action_zh
echo.
echo 当前输入无效，请重新选择。
timeout /t 2 >nul
exit /b 0

:show_comparison_workflow_failure_zh
echo.
echo [错误] 四组统一仿真对比阶段执行失败，错误码：%WORKFLOW_EXIT%
echo [提示] 请检查上方输出中的首个报错信息后再重试。
echo.
pause
exit /b 0

:show_two_stage_workflow_failure_zh
echo.
echo [错误] 虚实两阶段实验工作流执行失败，错误码：%WORKFLOW_EXIT%
echo [提示] 请检查上方输出中的首个报错信息后再重试。
echo.
pause
exit /b 0

:resolve_workflow_mode_zh
set "WORKFLOW_MODE=new"
set "workflow_action="
set "workflow_confirm="
call :query_latest_workflow "%~1"
if not defined WORKFLOW_QUERY_RESULT goto resolve_workflow_mode_zh_confirm
call :show_resume_options_zh
call :select_workflow_action
if /i "%workflow_action%"=="Q" exit /b 1
if /i "%workflow_action%"=="C" set "WORKFLOW_MODE=resume"
if /i "%workflow_action%"=="C" exit /b 0
if /i "%workflow_action%"=="N" set "WORKFLOW_MODE=new"
if /i "%workflow_action%"=="N" exit /b 0
call :show_invalid_workflow_action_zh
exit /b 2

:resolve_workflow_mode_zh_confirm
call :confirm_new_workflow_zh
if /i not "%workflow_confirm%"=="Y" exit /b 1
exit /b 0

:run_normal
cls
echo ============================================================
echo 系统运行（固定权重）
echo ============================================================
echo.
call :collect_quick_config "run_fixed"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Run_System_Fixed_Weights.bat
goto menu

:run_ddpg
cls
echo ============================================================
echo 系统运行（DDPG 权重预测）
echo ============================================================
echo.
call :collect_quick_config "run_ddpg"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Run_System_DDPG_Weights.bat
goto menu

:run_dqn_movement
cls
echo ============================================================
echo 系统运行（DQN 控制）
echo ============================================================
echo.
echo [提示] 该入口暂未开放，建议使用训练入口或评测入口。
echo.
pause
goto menu

:train_weight_airsim_fresh
cls
echo ============================================================
echo DDPG+APF 训练（AirSim，新模型）
echo ============================================================
echo.
call :collect_quick_config "ddpg_train"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Train_DDPG_Weights_Real_Environment.bat
goto menu

:train_weight_airsim_resume
cls
echo ============================================================
echo DDPG+APF 训练（AirSim，继续训练）
echo ============================================================
echo.
call :collect_quick_config "ddpg_train"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
set "DDPG_CONTINUE_MODEL=%~dp0multirotor\DDPG_Weight\models\weight_predictor_airsim"
if not exist "%DDPG_CONTINUE_MODEL%.zip" (
    echo [错误] 未检测到可继续训练的 DDPG 模型：
    echo        %DDPG_CONTINUE_MODEL%.zip
    echo.
    pause
    goto menu
)
call scripts\Train_DDPG_Weights_Real_Environment.bat --continue-model "%DDPG_CONTINUE_MODEL%"
goto menu

:train_weight_crazyflie_logs
cls
echo ============================================================
echo DDPG+APF 训练（实体日志离线）
echo ============================================================
echo.
call :collect_quick_config "ddpg_logs_train"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Train_DDPG_Weights_Crazyflie_Logs.bat
goto menu

:train_weight_crazyflie_online_single
cls
echo ============================================================
echo DDPG+APF 训练（实体无人机单轮在线）
echo ============================================================
echo.
call :collect_quick_config "ddpg_single_episode"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Train_DDPG_Weights_Crazyflie_Online_Single_Episode.bat
goto menu

:train_movement_airsim_fresh
cls
echo ============================================================
echo DQN 控制训练（AirSim，新模型）
echo ============================================================
echo.
call :collect_quick_config "dqn_train"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
set "USE_PRETRAINED=0"
call scripts\Train_DQN_Movement_Real_Environment.bat
set "USE_PRETRAINED="
goto menu

:train_movement_airsim_resume
cls
echo ============================================================
echo DQN 控制训练（AirSim，继续训练）
echo ============================================================
echo.
call :collect_quick_config "dqn_resume_train"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
if not exist "%~dp0multirotor\DQN_Movement\models\movement_dqn_airsim_final.zip" if not exist "%~dp0multirotor\DQN_Movement\models\movement_dqn_final.zip" if not exist "%~dp0multirotor\DQN_Movement\scripts\models\movement_dqn_airsim_final.zip" if not exist "%~dp0multirotor\DQN_Movement\scripts\models\movement_dqn_final.zip" (
    echo [错误] 未检测到可继续训练的 DQN 模型。
    echo.
    pause
    goto menu
)
set "USE_PRETRAINED=1"
call scripts\Train_DQN_Movement_Real_Environment.bat
set "USE_PRETRAINED="
goto menu

:train_movement_airsim_rerun_stage02
cls
echo ============================================================
echo DQN 控制训练（重新执行当前 stage02）
echo ============================================================
echo.
call :collect_quick_config "dqn_resume_train"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
if not exist "%~dp0multirotor\DQN_Movement\models\movement_dqn_airsim_final.zip" if not exist "%~dp0multirotor\DQN_Movement\models\movement_dqn_final.zip" if not exist "%~dp0multirotor\DQN_Movement\scripts\models\movement_dqn_airsim_final.zip" if not exist "%~dp0multirotor\DQN_Movement\scripts\models\movement_dqn_final.zip" (
    echo [错误] 未检测到可继续训练的 DQN 模型。
    echo.
    pause
    goto menu
)
set "USE_PRETRAINED=1"
set "TRAIN_STAGE_NAME=stage02_finetune"
set "TRAIN_STAGE_INDEX=2"
call scripts\Train_DQN_Movement_Real_Environment.bat
set "USE_PRETRAINED="
set "TRAIN_STAGE_NAME="
set "TRAIN_STAGE_INDEX="
goto menu

:train_hierarchical_dqn
cls
echo ============================================================
echo 分层 DQN 训练（离线 / Mock）
echo ============================================================
echo.
call :collect_quick_config "hrl_train"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Train_Hierarchical_DQN.bat
goto menu

:train_hierarchical_airsim
cls
echo ============================================================
echo 分层 DQN 训练（AirSim 融合）
echo ============================================================
echo.
call :collect_quick_config "hrl_train"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Train_Hierarchical_With_AirSim.bat
goto menu

:test_movement_dqn
cls
echo ============================================================
echo DQN 控制模型验证
echo ============================================================
echo.
call scripts\Test_DQN_Movement.bat
goto menu

:data_visualization
cls
echo ============================================================
echo 可视化分析结果生成
echo ============================================================
echo.
call scripts\Data_Visualization_Analysis.bat
goto menu

:compare_algorithms
cls
echo ============================================================
echo DDPG 与 DQN 对比分析
echo ============================================================
echo.
if not exist "myvenv\Scripts\activate.bat" (
    echo [错误] Python 虚拟环境不存在
    pause
    goto menu
)
call myvenv\Scripts\activate.bat
echo.
python "multirotor\Algorithm\visualize_training_data.py" --compare-algorithms --out analysis_results
echo.
pause
goto menu

:comparison_workflow
cls
echo ============================================================
echo 四组统一仿真对比阶段
echo ============================================================
echo.
echo 本流程将依次执行以下阶段：
echo   [1] APF 基线多轮仿真阶段（fixed APF / random APF）
echo   [2] DDPG+APF stage01 训练
echo   [3] Pure DQN stage01 训练
echo   [4] 在 Unity/AirSim 中执行四组仿真评测（冻结策略：fixed APF / random APF / DDPG+APF / Pure DQN）
echo   [5] 训练结果对比分析与 stage02 建议
echo.
echo 说明：
echo   - fixed APF 与 random APF 不参加训练阶段，但会先进行多轮仿真，再进入最终统一对比
echo   - 可配置项（直接回车使用默认值）：
echo     * 无人机数量（默认 3）
echo     * APF 基线多轮仿真轮次
echo     * 四组 benchmark 每 seed 评测轮次
echo     * DDPG 训练步数
echo     * DQN 训练步数
echo   - 执行过程中，训练脚本仍会保留各自的启动确认提示
echo.
call :resolve_workflow_mode_zh "comparison"
if errorlevel 2 goto comparison_workflow
if errorlevel 1 goto menu
echo.
call :collect_quick_config "comparison_workflow"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call :run_paper_workflow "comparison" "%WORKFLOW_MODE%"
set "WORKFLOW_EXIT=%ERRORLEVEL%"
if not "%WORKFLOW_EXIT%"=="0" (
    call :show_comparison_workflow_failure_zh
)
if /i "%AIRSIM_TEST_EXIT_AFTER_WORKFLOW%"=="1" goto end
goto menu

:virtual_real_two_stage_workflow
cls
echo ============================================================
echo 虚实两阶段实验工作流 (Virtual-Real Two-Stage Workflow)
echo ============================================================
echo.
echo 本流程支持快速配置以下参数：
echo   * 无人机数量（默认 3）
echo   * DDPG 训练步数（将显示仿真时间预估）
echo.
call :resolve_workflow_mode_zh "virtual_real_two_stage"
if errorlevel 2 goto virtual_real_two_stage_workflow
if errorlevel 1 goto menu
call :collect_quick_config "two_stage_workflow"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call :run_paper_workflow "virtual_real_two_stage" "%WORKFLOW_MODE%"
set "WORKFLOW_EXIT=%ERRORLEVEL%"
if not "%WORKFLOW_EXIT%"=="0" (
    call :show_two_stage_workflow_failure_zh
)
if /i "%AIRSIM_TEST_EXIT_AFTER_WORKFLOW%"=="1" goto end
goto menu

:four_group_benchmark
cls
echo ============================================================
echo 四组仿真评测（冻结策略）/ Benchmark
echo ============================================================
echo.
call :collect_quick_config "four_group_benchmark"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
call scripts\Run_Four_Group_Benchmark.bat
goto menu

:analyze_four_group_benchmark
cls
echo ============================================================
echo 四组主结果分析
echo ============================================================
echo.
call scripts\Analyze_Four_Group_Benchmark.bat
goto menu

:analyze_family_comparisons
cls
echo ============================================================
echo Family 维度对比分析
echo ============================================================
echo.
call scripts\Analyze_Family_Comparisons.bat
goto menu

:train_paper_ddpg_seeds
cls
echo ============================================================
echo 论文多 Seed 训练（DDPG+APF）
echo ============================================================
echo.
call :collect_quick_config "paper_ddpg_seeds"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
powershell -ExecutionPolicy Bypass -File "scripts\Run_Paper_Training_Seeds.ps1" -Algorithm ddpg_apf
goto menu

:train_paper_dqn_seeds
cls
echo ============================================================
echo 论文多 Seed 训练（Pure DQN）
echo ============================================================
echo.
call :collect_quick_config "paper_dqn_seeds"
if errorlevel 2 goto menu
if errorlevel 1 goto menu
powershell -ExecutionPolicy Bypass -File "scripts\Run_Paper_Training_Seeds.ps1" -Algorithm pure_dqn
goto menu

:cleanup_menu
cls
echo ============================================================
echo           系统维护与清理
echo ============================================================
echo.
echo --- [DDPG+APF 训练产出] ---
echo   [1] 清理 DDPG 模型文件 (multirotor\DDPG_Weight\models)
echo   [2] 清理 DDPG 训练日志 (logs + airsim_training_logs + crazyflie_logs)
echo.
echo --- [DQN 控制训练产出] ---
echo   [3] 清理 DQN 模型文件 (multirotor\DQN_Movement\models)
echo   [4] 清理 DQN 训练日志 (logs + scripts\logs)
echo.
echo --- [分析结果与全局清理] ---
echo   [5] 清理分析结果 (analysis_results)
echo   [8] 清理全部模型与日志（慎用）
echo   [9] 返回主菜单
echo.
echo ============================================================
echo.
set /p cleanup_choice=请选择维护选项 (1-9):

if "%cleanup_choice%"=="1" goto cleanup_ddpg_models
if "%cleanup_choice%"=="2" goto cleanup_ddpg_logs
if "%cleanup_choice%"=="3" goto cleanup_dqn_models
if "%cleanup_choice%"=="4" goto cleanup_dqn_logs
if "%cleanup_choice%"=="5" goto cleanup_analysis_results
if "%cleanup_choice%"=="8" goto cleanup_all
if "%cleanup_choice%"=="9" goto menu

echo.
echo 当前输入无效，请重新选择。
timeout /t 2 >nul
goto cleanup_menu

:confirm_delete
set "TARGET_DIR=%~1"
set "TARGET_DESC=%~2"
if "%TARGET_DIR%"=="" goto cleanup_menu
cls
echo ============================================================
echo 删除确认: %TARGET_DESC%
echo ------------------------------------------------------------
echo 路径: %TARGET_DIR%
echo ============================================================
echo.
echo [警告] 该操作不可撤销！
echo.
set /p confirm=请输入 YES 确认删除，输入其他内容取消:
if /i not "%confirm%"=="YES" (
    echo.
    echo 已取消操作。
    timeout /t 2 >nul
    goto cleanup_menu
)

if not exist "%TARGET_DIR%" (
    echo.
    echo [提示] 目标目录不存在，无需清理。
    timeout /t 2 >nul
    goto cleanup_menu
)

rmdir /s /q "%TARGET_DIR%" 2>nul
if exist "%TARGET_DIR%" (
    echo.
    echo [错误] 清理失败，请检查文件是否被占用。
) else (
    echo.
    echo [成功] %TARGET_DESC% 清理完成。
)
pause
goto cleanup_menu

:cleanup_ddpg_models
call :confirm_delete "multirotor\DDPG_Weight\models" "DDPG 权重模型"
goto cleanup_menu

:cleanup_ddpg_logs
cls
echo ============================================================
echo 清理 DDPG 训练日志
echo ============================================================
echo.
echo 将清理以下目录:
echo   - multirotor\DDPG_Weight\logs
echo   - multirotor\DDPG_Weight\airsim_training_logs
echo   - multirotor\DDPG_Weight\crazyflie_logs
echo.
echo [警告] 该操作不可撤销！
echo.
set /p confirm=请输入 YES 确认删除，输入其他内容取消:
if /i not "%confirm%"=="YES" (
    echo.
    echo 已取消操作。
    timeout /t 2 >nul
    goto cleanup_menu
)
if exist "multirotor\DDPG_Weight\logs" rmdir /s /q "multirotor\DDPG_Weight\logs" 2>nul
if exist "multirotor\DDPG_Weight\airsim_training_logs" rmdir /s /q "multirotor\DDPG_Weight\airsim_training_logs" 2>nul
if exist "multirotor\DDPG_Weight\crazyflie_logs" rmdir /s /q "multirotor\DDPG_Weight\crazyflie_logs" 2>nul
echo.
echo [成功] DDPG 日志清理完成。
pause
goto cleanup_menu

:cleanup_dqn_models
echo 正在清理 DQN 模型文件...
if exist "multirotor\DQN_Movement\models\movement_dqn_airsim_final.zip" del /f /q "multirotor\DQN_Movement\models\movement_dqn_airsim_final.zip"
if exist "multirotor\DQN_Movement\scripts\models\movement_dqn_airsim_final.zip" del /f /q "multirotor\DQN_Movement\scripts\models\movement_dqn_airsim_final.zip"
echo [成功] DQN 模型文件清理完成。
pause
goto cleanup_menu

:cleanup_dqn_logs
echo 正在清理 DQN 训练日志...
if exist "multirotor\DQN_Movement\logs" rmdir /s /q "multirotor\DQN_Movement\logs"
if exist "multirotor\DQN_Movement\scripts\logs" rmdir /s /q "multirotor\DQN_Movement\scripts\logs"
echo [成功] DQN 日志清理完成。
pause
goto cleanup_menu

:cleanup_analysis_results
call :confirm_delete "analysis_results" "分析结果"
goto cleanup_menu

:cleanup_all
cls
echo ============================================================
echo           高风险操作：清理全部产出
echo ============================================================
echo.
echo 将清理所有 DDPG / DQN / HRL 的模型、日志及分析结果。
echo.
set /p final_confirm=请输入 DELETE_ALL 确认执行:
if /i not "%final_confirm%"=="DELETE_ALL" (
    echo.
    echo 操作已取消。
    timeout /t 2 >nul
    goto cleanup_menu
)

echo 正在执行全面清理...
for %%D in (
    "multirotor\DDPG_Weight\models"
    "multirotor\DDPG_Weight\logs"
    "multirotor\DDPG_Weight\airsim_training_logs"
    "multirotor\DDPG_Weight\crazyflie_logs"
    "multirotor\DQN_Movement\models"
    "multirotor\DQN_Movement\scripts\models"
    "multirotor\DQN_Movement\logs"
    "multirotor\DQN_Movement\scripts\logs"
    "analysis_results"
) do (
    if exist "%%~D" rmdir /s /q "%%~D"
)
echo.
echo [成功] 全部训练与分析产出已清理完成。
pause
goto cleanup_menu

:info
cls
echo ============================================================
echo    平台信息
echo ============================================================
echo.
echo 核心目录:
echo   - multirotor\AlgorithmServer.py
echo   - multirotor\Algorithm\
echo   - multirotor\DDPG_Weight\
echo   - multirotor\DQN_Movement\
echo   - multirotor\benchmark_registry.json
echo   - docs\FOUR_GROUP_BENCHMARK_WORKFLOW_ZH.md
echo.
echo 常用入口:
echo   - start.bat
echo   - scripts\Run_Four_Group_Benchmark.bat
echo   - scripts\Analyze_Four_Group_Benchmark.bat
echo   - scripts\Analyze_Family_Comparisons.bat
echo   - scripts\Run_Paper_Training_Seeds.ps1
echo   - docs\START_QUICK_CONFIG_ZH.md
echo.
echo 运行环境:
call myvenv\Scripts\activate.bat 2>nul
if %ERRORLEVEL% EQU 0 (
    python --version 2>nul
    echo [OK] 虚拟环境已就绪
) else (
    echo [提示] 未检测到虚拟环境，请检查 myvenv
)
echo.
pause
goto menu

:end
cls
echo ============================================================
echo 感谢使用 AirSim 无人机仿真平台！
echo ============================================================
echo.
timeout /t 2 >nul
endlocal
exit /b 0
