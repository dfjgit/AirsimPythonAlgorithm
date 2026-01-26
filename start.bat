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
echo   [8] 训练移动DQN (真实AirSim环境)[还有些问题]
echo.
echo === 系统信息 ===
echo   [9] 查看系统信息
echo   [0] 退出
echo.
echo ============================================================
echo.

set /p choice=请输入选项 (0-9): 

if "%choice%"=="1" goto run_normal
if "%choice%"=="2" goto run_dqn
if "%choice%"=="3" goto run_dqn_movement
if "%choice%"=="4" goto train_weight_airsim
if "%choice%"=="5" goto train_weight_crazyflie_online
if "%choice%"=="6" goto train_weight_crazyflie_logs
if "%choice%"=="7" goto train_weight_hybrid
if "%choice%"=="8" goto train_movement_airsim
if "%choice%"=="9" goto info
if "%choice%"=="0" goto end

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
call scripts\运行系统-固定权重.bat
goto menu

:run_dqn
cls
echo ============================================================
echo 启动系统 - DDPG权重预测模式
echo ============================================================
echo.
call scripts\运行系统-DDPG权重.bat
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
call scripts\训练权重DDPG-真实环境.bat
goto menu

:train_weight_crazyflie_online
cls
echo ============================================================
echo DDPG权重APF训练 (实体训练 - Crazyflie在线)
echo ============================================================
echo.
call scripts\训练权重DDPG-实体机在线.bat
goto menu

:train_weight_crazyflie_logs
cls
echo ============================================================
echo DDPG权重APF训练 (实体训练 - Crazyflie日志)
echo ============================================================
echo.
call scripts\训练权重DDPG-实体机日志.bat
goto menu

:train_weight_hybrid
cls
echo ============================================================
echo DDPG权重APF训练 (虚实融合训练)
echo ============================================================
echo.
call scripts\训练权重DDPG-虚实融合.bat
goto menu

:train_movement_airsim
cls
echo ============================================================
echo DQN移动控制训练 (真实AirSim环境)
echo ============================================================
echo.
call scripts\训练移动DQN-真实环境.bat
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

