@echo off
chcp 65001 >nul
:menu
cls
echo ============================================================
echo    AirSim 无人机仿真系统 - 主菜单
echo ============================================================
echo.
echo === 系统运行 ===
echo   [1] 运行系统 (固定权重)[⭐可以使用！]
echo   [2] 运行系统 (DDPG权重预测) [⭐可以使用！]
echo   [3] 运行系统 (DQN模型) [待开发]
echo.
echo === DDPG权重APF训练 ===
echo   [4] 训练权重DDPG (虚拟训练 - AirSim环境) [⭐可以使用！]
echo   [5] 训练权重DDPG (实体训练 - Crazyflie在线) [⭐可以使用！]
echo   [6] 训练权重DDPG (实体训练 - Crazyflie日志) [⭐可以使用！]
echo   [7] 训练权重DDPG (虚实融合训练) [⭐可以使用！]
echo.
echo === DQN移动控制训练 ===
echo   [8] 训练移动DQN (真实AirSim环境)[⭐可以使用！]
echo   [D] 测试移动DQN模型 [⭐可以使用！]
echo.
echo === 数据分析 ===
echo   [A] 数据可视化分析 [⭐可以使用！]
echo   [B] DDPG vs DQN 算法对比 [⭐可以使用！]
echo.
echo === 系统信息 ===
echo   [9] 查看系统信息
echo   [0] 退出
echo.
echo ============================================================
echo.

set /p choice=请输入选项 (0-9,A-D): 

if /i "%choice%"=="1" goto run_normal
if /i "%choice%"=="2" goto run_dqn
if /i "%choice%"=="3" goto run_dqn_movement
if /i "%choice%"=="4" goto train_weight_airsim
if /i "%choice%"=="5" goto train_weight_crazyflie_online
if /i "%choice%"=="6" goto train_weight_crazyflie_logs
if /i "%choice%"=="7" goto train_weight_hybrid
if /i "%choice%"=="8" goto train_movement_airsim
if /i "%choice%"=="d" goto test_movement_dqn
if /i "%choice%"=="a" goto data_visualization
if /i "%choice%"=="b" goto compare_algorithms
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

:run_dqn
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

:train_weight_airsim
cls
echo ============================================================
echo DDPG权重APF训练 (虚拟训练 - AirSim环境)
echo ============================================================
echo.
call scripts\Train_DDPG_Weights_Real_Environment.bat
goto menu

:train_weight_crazyflie_online
cls
echo ============================================================
echo DDPG权重APF训练 (实体训练 - Crazyflie在线)
echo ============================================================
echo.
call scripts\Train_DDPG_Weights_Crazyflie_Online.bat
goto menu

:train_weight_crazyflie_logs
cls
echo ============================================================
echo DDPG权重APF训练 (实体训练 - Crazyflie日志)
echo ============================================================
echo.
call scripts\Train_DDPG_Weights_Crazyflie_Logs.bat
goto menu

:train_weight_hybrid
cls
echo ============================================================
echo DDPG权重APF训练 (虚实融合训练)
echo ============================================================
echo.
call scripts\Train_DDPG_Weights_Hybrid.bat
goto menu

:train_movement_airsim
cls
echo ============================================================
echo DQN移动控制训练 (真实AirSim环境)
echo ============================================================
echo.
call scripts\Train_DQN_Movement_Real_Environment.bat
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
python "multirotor\Algorithm\visualize_training_data.py" --auto --compare-algorithms --compare-algorithms-full --out analysis_results
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo [成功] 算法对比分析完成！
    echo ============================================================
    echo.
    echo 输出目录结构：
    echo   analysis_results\
    echo   ├── DDPG_scan_data_XXXXXX\          [DDPG单独分析 - 11张图]
    echo   ├── DQN_scan_data_XXXXXX\           [DQN单独分析 - 11张图]
    echo   ├── dqn_movement_XXXXXX\            [DQN训练分析 - 4张图]
    echo   ├── algorithm_comparison_ddpg_vs_dqn\     [基础对比 - 4图+报告]
    echo   └── algorithm_comparison_ddpg_vs_dqn_full\ [全方位对比 - 6图+报告]
    echo.
) else (
    echo.
    echo [失败] 算法对比分析失败，请检查是否有足够的训练数据
)
echo.
pause
goto menu

:info
cls
echo ============================================================
echo    系统信息
echo ============================================================
echo.
echo 项目结构:
echo   - multirotor\AlgorithmServer.py    : 算法服务器
echo   - multirotor\Algorithm\            : APF算法实现
echo   - multirotor\DDPG_Weight\         : DDPG权重APF训练
echo   - multirotor\DQN_Movement\         : DQN移动控制训练
echo   - myvenv\                           : Python虚拟环境
echo.
echo 配置文件:
echo   - multirotor\scanner_config.json                     : APF算法配置
echo   - multirotor\DDPG_Weight\unified_train_config.json    : 统一训练配置 (推荐)
echo   - multirotor\DDPG_Weight\*_train_config*.json        : 旧式训练配置 (兼容)
echo   - multirotor\DQN_Movement\movement_dqn_config.json    : 移动DQN配置
echo.
echo 批处理文件:
echo   - start.bat                         : 主菜单(当前)
echo   - scripts\运行系统-固定权重.bat      : 运行系统(固定权重)
echo   - scripts\运行系统-DDPG权重.bat       : 运行系统(DDPG权重)
echo   - scripts\训练权重DDPG-真实环境.bat   : 训练权重DDPG(虚拟训练)
echo   - scripts\训练权重DDPG-实体机在线.bat : 训练权重DDPG(实体在线)
echo   - scripts\训练权重DDPG-实体机日志.bat : 训练权重DDPG(实体日志)
echo   - scripts\训练权重DDPG-虚实融合.bat   : 训练权重DDPG(虚实融合)
echo   - scripts\训练移动DQN-真实环境.bat   : 训练移动DQN(真实环境)
echo.
echo Python环境:
call .venv\Scripts\activate.bat 2>nul
if %ERRORLEVEL% EQU 0 (
    python --version 2>nul
    echo [OK] 虚拟环境已就绪
) else (
    echo [!] 虚拟环境未创建，请先运行 run_two_drones.bat
)
echo.
echo DDPG模型:
if exist "multirotor\DDPG_Weight\models\best_model.zip" (
    echo [OK] 权重APF模型已训练 (best_model.zip)
) else (
    echo [!] 权重APF模型未训练，请运行选项 [4-7] 训练
)
if exist "multirotor\DQN_Movement\models\movement_dqn_final.zip" (
    echo [OK] 移动控制模型已训练
) else (
    echo [!] 移动控制模型未训练，请运行选项 [8] 训练
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
exit

