@echo off
chcp 65001 >nul
setlocal EnableExtensions

:menu
cls
echo ============================================================
echo    AirSim 无人机仿真系统 - 主菜单
echo ============================================================
echo.
echo === 系统运行 ===
echo   [1] 运行系统 (固定权重) [可以使用]
echo   [2] 运行系统 (DDPG权重预测) [可以使用]
echo   [3] 运行系统 (DQN模型) [待开发]
echo.
echo === DDPG权重APF训练 ===
echo   [4] 训练权重DDPG (虚拟训练 - 新建模型) [可以使用]
echo   [5] 训练权重DDPG (虚拟训练 - 继续训练) [可以使用]
echo   [6] 训练权重DDPG (使用日志离线训练) [可以使用]
echo.
echo === DQN移动控制训练 ===
echo   [7] 训练移动DQN (真实AirSim环境 - 新建模型) [可以使用]
echo   [8] 训练移动DQN (真实AirSim环境 - 继续训练) [可以使用]
echo   [H] 训练分层DQN (离线/Mock模式)
echo   [F] 训练分层DQN (AirSim融合模式) [新功能]
echo   [D] 测试移动DQN模型 [可以使用]
echo.
echo === 数据分析 ===
echo   [A] 数据可视化分析 [可以使用]
echo   [B] DDPG vs DQN 算法对比 [可以使用]
echo.
echo === 数据清理 ===
echo   [C] 删除训练产出(模型/日志/分析结果)
echo.
echo === 系统信息 ===
echo   [9] 查看系统信息
echo   [0] 退出
echo.
echo ============================================================
echo.

set /p choice=请输入选项 (0-9,A-H): 

if /i "%choice%"=="1" goto run_normal
if /i "%choice%"=="2" goto run_ddpg
if /i "%choice%"=="3" goto run_dqn_movement
if /i "%choice%"=="4" goto train_weight_airsim_fresh
if /i "%choice%"=="5" goto train_weight_airsim_resume
if /i "%choice%"=="6" goto train_weight_crazyflie_logs
if /i "%choice%"=="7" goto train_movement_airsim_fresh
if /i "%choice%"=="8" goto train_movement_airsim_resume
if /i "%choice%"=="H" goto train_hierarchical_dqn
if /i "%choice%"=="F" goto train_hierarchical_airsim
if /i "%choice%"=="D" goto test_movement_dqn
if /i "%choice%"=="A" goto data_visualization
if /i "%choice%"=="B" goto compare_algorithms
if /i "%choice%"=="C" goto cleanup_menu
if /i "%choice%"=="9" goto info
if /i "%choice%"=="0" goto end

echo.
echo 无效的选项，请重新选择
timeout /t 2 >nul
goto menu

:run_normal
cls
echo ============================================================
echo 启动系统 - 固定权重模式
echo ============================================================
echo.
call scripts\Run_System_Fixed_Weights.bat
goto menu

:run_ddpg
cls
echo ============================================================
echo 启动系统 - DDPG权重预测模式
echo ============================================================
echo.
call scripts\Run_System_DDPG_Weights.bat
goto menu

:run_dqn_movement
cls
echo ============================================================
echo 运行系统 - DQN模型
echo ============================================================
echo.
echo [提示] 此功能正在开发中，敬请期待...
echo.
pause
goto menu

:train_weight_airsim_fresh
cls
echo ============================================================
echo DDPG权重APF训练 (虚拟训练 - 新建模型)
echo ============================================================
echo.
call scripts\Train_DDPG_Weights_Real_Environment.bat
goto menu

:train_weight_airsim_resume
cls
echo ============================================================
echo DDPG权重APF训练 (虚拟训练 - 继续训练)
echo ============================================================
echo.
set "DDPG_CONTINUE_MODEL=%~dp0multirotor\DDPG_Weight\models\weight_predictor_airsim"
if not exist "%DDPG_CONTINUE_MODEL%.zip" (
    echo [错误] 未找到可继续训练的 DDPG 模型：
    echo         %DDPG_CONTINUE_MODEL%.zip
    echo.
    pause
    goto menu
)
call scripts\Train_DDPG_Weights_Real_Environment.bat --continue-model "%DDPG_CONTINUE_MODEL%"
goto menu

:train_weight_crazyflie_logs
cls
echo ============================================================
echo DDPG权重APF训练 (使用日志离线训练)
echo ============================================================
echo.
call scripts\Train_DDPG_Weights_Crazyflie_Logs.bat
goto menu

:train_movement_airsim_fresh
cls
echo ============================================================
echo DQN移动控制训练 (真实AirSim环境 - 新建模型)
echo ============================================================
echo.
set "USE_PRETRAINED=0"
call scripts\Train_DQN_Movement_Real_Environment.bat
set "USE_PRETRAINED="
goto menu

:train_movement_airsim_resume
cls
echo ============================================================
echo DQN移动控制训练 (真实AirSim环境 - 继续训练)
echo ============================================================
echo.
if not exist "%~dp0multirotor\DQN_Movement\models\movement_dqn_airsim_final.zip" if not exist "%~dp0multirotor\DQN_Movement\models\movement_dqn_final.zip" if not exist "%~dp0multirotor\DQN_Movement\scripts\models\movement_dqn_airsim_final.zip" if not exist "%~dp0multirotor\DQN_Movement\scripts\models\movement_dqn_final.zip" (
    echo [错误] 未找到可继续训练的 DQN 模型。
    echo.
    pause
    goto menu
)
set "USE_PRETRAINED=1"
call scripts\Train_DQN_Movement_Real_Environment.bat
set "USE_PRETRAINED="
goto menu

:train_hierarchical_dqn
cls
echo ============================================================
echo 分层强化学习 (HRL) 训练 - 离线模式
echo ============================================================
echo.
call scripts\Train_Hierarchical_DQN.bat
goto menu

:train_hierarchical_airsim
cls
echo ============================================================
echo 分层强化学习 (HRL) 训练 - AirSim融合模式
echo ============================================================
echo.
call scripts\Train_Hierarchical_With_AirSim.bat
goto menu

:test_movement_dqn
cls
echo ============================================================
echo 测试移动DQN模型
echo ============================================================
echo.
call scripts\Test_DQN_Movement.bat
goto menu

:data_visualization
cls
echo ============================================================
echo 数据可视化分析
echo ============================================================
echo.
call scripts\Data_Visualization_Analysis.bat
goto menu

:compare_algorithms
cls
echo ============================================================
echo DDPG vs DQN 算法对比分析
echo ============================================================
echo.
echo [提示] 此功能将对比 DDPG 和 DQN 两种算法的训练效果
echo [提示] 包含：单独分析 + 基础对比 + 全方位对比
echo.
if not exist "myvenv\Scripts\activate.bat" (
    echo [错误] Python虚拟环境不存在
    pause
    goto menu
)
call myvenv\Scripts\activate.bat
echo.
echo [执行] 正在进行 DDPG 和 DQN 的完整对比分析...
echo.
python "multirotor\Algorithm\visualize_training_data.py" --compare-algorithms --out analysis_results
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo [成功] 算法对比分析完成！
    echo ============================================================
    echo.
    echo 输出目录结构：
    echo   analysis_results\
    echo   └── algorithm_comparison\           [统一对比 - 4图+报告]
    echo.
) else (
    echo.
    echo [失败] 算法对比分析失败，请检查是否有足够的训练数据
)
echo.
pause
goto menu

:cleanup_menu
cls
echo ============================================================
echo           数据清理与维护菜单
echo ============================================================
echo.
echo --- [DDPG 权重训练 (APF)] ---
echo   [1] 删除 DDPG 模型 (multirotor\DDPG_Weight\models)
echo   [2] 删除 DDPG 日志 (logs + airsim_training_logs + crazyflie_logs)
echo.
echo --- [DQN 基础移动控制] ---
echo   [3] 删除 DQN 移动模型 (multirotor\DQN_Movement\models)
echo   [4] 删除 DQN 移动日志 (logs + scripts\logs)
echo.
echo --- [分析结果与全局] ---
echo   [5] 删除 算法对比分析结果 (analysis_results)
echo   [8] 删除 所有模型和日志 (慎用！)
echo   [9] 返回主菜单
echo.
echo ============================================================
echo.
set /p cleanup_choice=请输入选项 (1-9): 

if "%cleanup_choice%"=="1" goto cleanup_ddpg_models
if "%cleanup_choice%"=="2" goto cleanup_ddpg_logs
if "%cleanup_choice%"=="3" goto cleanup_dqn_models
if "%cleanup_choice%"=="4" goto cleanup_dqn_logs
if "%cleanup_choice%"=="5" goto cleanup_analysis_results
if "%cleanup_choice%"=="8" goto cleanup_all
if "%cleanup_choice%"=="9" goto menu

echo.
echo 无效的选项，请重新选择
timeout /t 2 >nul
goto cleanup_menu

:confirm_delete
set "TARGET_DIR=%~1"
set "TARGET_DESC=%~2"
if "%TARGET_DIR%"=="" goto cleanup_menu
cls
echo ============================================================
echo 确认删除: %TARGET_DESC%
echo ------------------------------------------------------------
echo 路径: %TARGET_DIR%
echo ============================================================
echo.
echo [警告] 该操作不可恢复！
echo.
set /p confirm=请输入 YES 确认删除，输入其它取消: 
if /i not "%confirm%"=="YES" (
    echo.
    echo 已取消操作。
    timeout /t 2 >nul
    goto cleanup_menu
)

if not exist "%TARGET_DIR%" (
    echo.
    echo [提示] 目录不存在，无需清理。
    timeout /t 2 >nul
    goto cleanup_menu
)

rmdir /s /q "%TARGET_DIR%" 2>nul
if exist "%TARGET_DIR%" (
    echo.
    echo [错误] 删除失败，请检查文件是否被占用。
) else (
    echo.
    echo [成功] %TARGET_DESC% 已清理。
)
pause
goto cleanup_menu

:cleanup_ddpg_models
call :confirm_delete "multirotor\DDPG_Weight\models" "DDPG 权重模型"
goto cleanup_menu

:cleanup_ddpg_logs
cls
echo ============================================================
echo 删除 DDPG 训练日志
echo ============================================================
echo.
echo 将删除:
echo   - multirotor\DDPG_Weight\logs (可视化日志)
echo   - multirotor\DDPG_Weight\airsim_training_logs (AirSim训练数据)
echo   - multirotor\DDPG_Weight\crazyflie_logs (Crazyflie训练数据)
echo.
echo [警告] 该操作不可恢复！
echo.
set /p confirm=请输入 YES 确认删除，输入其它取消:
if /i not "%confirm%"=="YES" (
    echo.
    echo 已取消操作。
    timeout /t 2 >nul
    goto cleanup_menu
)
echo.
echo 正在删除 DDPG 日志...
if exist "multirotor\DDPG_Weight\logs" rmdir /s /q "multirotor\DDPG_Weight\logs" 2>nul
if exist "multirotor\DDPG_Weight\airsim_training_logs" rmdir /s /q "multirotor\DDPG_Weight\airsim_training_logs" 2>nul
if exist "multirotor\DDPG_Weight\crazyflie_logs" rmdir /s /q "multirotor\DDPG_Weight\crazyflie_logs" 2>nul
echo.
echo [成功] DDPG 日志已删除。
echo.
pause
goto cleanup_menu

:cleanup_dqn_models
echo 正在清理 DQN 基础移动模型...
if exist "multirotor\DQN_Movement\models\movement_dqn_airsim_final.zip" del /f /q "multirotor\DQN_Movement\models\movement_dqn_airsim_final.zip"
if exist "multirotor\DQN_Movement\scripts\models\movement_dqn_airsim_final.zip" del /f /q "multirotor\DQN_Movement\scripts\models\movement_dqn_airsim_final.zip"
echo [成功] DQN 移动模型已清理。
pause
goto cleanup_menu

:cleanup_dqn_logs
echo 正在清理 DQN 移动日志...
if exist "multirotor\DQN_Movement\logs" rmdir /s /q "multirotor\DQN_Movement\logs"
if exist "multirotor\DQN_Movement\scripts\logs" rmdir /s /q "multirotor\DQN_Movement\scripts\logs"
echo [成功] DQN 移动日志已清理。
pause
goto cleanup_menu

:cleanup_analysis_results
call :confirm_delete "analysis_results" "可视化分析结果"
goto cleanup_menu

:cleanup_all
cls
echo ============================================================
echo           危险操作：清理所有产出
echo ============================================================
echo.
echo 将删除所有 DDPG/DQN/HRL 的模型、日志以及分析结果。
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
echo [成功] 所有训练产出已清理完毕。
pause
goto cleanup_menu

:info
cls
echo ============================================================
echo    系统信息
echo ============================================================
echo.
echo 项目结构:
echo   - multirotor\AlgorithmServer.py              : 算法服务器
echo   - multirotor\Algorithm\                      : APF算法实现
echo   - multirotor\DDPG_Weight\                    : DDPG权重APF训练
echo   - multirotor\DQN_Movement\                   : DQN移动控制训练
echo   - myvenv\                                    : Python虚拟环境
echo.
echo 当前默认续训模型:
echo   - DDPG: multirotor\DDPG_Weight\models\weight_predictor_airsim.zip
echo   - DQN : multirotor\DQN_Movement\models\movement_dqn_airsim_final.zip
echo.
echo 批处理文件:
echo   - start.bat                                  : 主菜单(当前)
echo   - scripts\运行系统-固定权重.bat             : 运行系统(固定权重)
echo   - scripts\运行系统-DDPG权重.bat             : 运行系统(DDPG权重)
echo   - scripts\训练权重DDPG-真实环境.bat         : 训练权重DDPG(虚拟训练)
echo   - scripts\训练权重DDPG-实体机日志.bat       : 训练权重DDPG(实体日志)
echo   - scripts\训练移动DQN-真实环境.bat          : 训练移动DQN(真实环境)
echo   - scripts\训练分层DQN.bat                   : 训练分层DQN(高层+底层)
echo.
echo Python环境:
call myvenv\Scripts\activate.bat 2>nul
if %ERRORLEVEL% EQU 0 (
    python --version 2>nul
    echo [OK] 虚拟环境已就绪
) else (
    echo [提示] 虚拟环境未激活，请按项目说明检查 myvenv
)
echo.
echo DDPG模型:
if exist "multirotor\DDPG_Weight\models\weight_predictor_airsim.zip" (
    echo [OK] DDPG 主模型已存在
) else (
    echo [提示] DDPG 主模型不存在，可使用选项 [4] 从头训练
)
if exist "multirotor\DQN_Movement\models\movement_dqn_airsim_final.zip" (
    echo [OK] DQN 主模型已存在
) else (
    echo [提示] DQN 主模型不存在，可使用选项 [7] 从头训练
)
echo.
echo ============================================================
echo.
pause
goto menu

:end
cls
echo ============================================================
echo 感谢使用 AirSim 无人机仿真系统！
echo ============================================================
echo.
timeout /t 2 >nul
endlocal
exit /b 0
