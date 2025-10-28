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
echo   [2] 运行系统 (DQN权重预测) [⭐可以使用！]
echo.
echo === DQN权重APF训练 ===
echo   [3] 训练权重DQN (模拟数据) [暂时别用这个，用4]
echo   [4] 训练权重DQN (真实AirSim环境) [⭐可以使用！]
echo   [5] 测试权重DQN模型 [别用这个，还没测试]
echo.
echo === DQN移动控制训练 ===
echo   [6] 训练移动DQN (模拟数据) [还有些问题]
echo   [7] 训练移动DQN (真实AirSim环境)[还有些问题]
echo   [8] 测试移动DQN模型[还有些问题]
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
if "%choice%"=="3" goto train_weight
if "%choice%"=="4" goto train_weight_airsim
if "%choice%"=="5" goto test_weight
if "%choice%"=="6" goto train_movement
if "%choice%"=="7" goto train_movement_airsim
if "%choice%"=="8" goto test_movement
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
echo 启动系统 - DQN权重预测模式
echo ============================================================
echo.
call scripts\运行系统-DQN权重.bat
goto menu

:train_weight
cls
echo ============================================================
echo DQN权重APF训练 (模拟数据)
echo ============================================================
echo.
call scripts\训练权重DQN-模拟.bat
goto menu

:train_weight_airsim
cls
echo ============================================================
echo DQN权重APF训练 (真实AirSim环境)
echo ============================================================
echo.
call scripts\训练权重DQN-真实环境.bat
goto menu

:test_weight
cls
echo ============================================================
echo 测试DQN权重APF模型
echo ============================================================
echo.
call scripts\测试权重DQN模型.bat
goto menu

:train_movement
cls
echo ============================================================
echo DQN移动控制训练 (模拟数据)
echo ============================================================
echo.
call scripts\训练移动DQN-模拟.bat
goto menu

:train_movement_airsim
cls
echo ============================================================
echo DQN移动控制训练 (真实AirSim环境)
echo ============================================================
echo.
call scripts\训练移动DQN-真实环境.bat
goto menu

:test_movement
cls
echo ============================================================
echo 测试DQN移动控制模型
echo ============================================================
echo.
call scripts\测试移动DQN模型.bat
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
echo   - multirotor\DQN_Weight\           : DQN权重APF训练
echo   - multirotor\DQN_Movement\         : DQN移动控制训练
echo   - .venv\                           : Python虚拟环境
echo.
echo 配置文件:
echo   - multirotor\scanner_config.json                : APF算法配置
echo   - multirotor\DQN_Weight\dqn_reward_config.json  : 权重DQN配置
echo   - multirotor\DQN_Movement\movement_dqn_config.json : 移动DQN配置
echo.
echo 批处理文件:
echo   - start.bat                         : 主菜单(当前)
echo   - scripts\运行系统-固定权重.bat      : 运行系统(固定权重)
echo   - scripts\运行系统-DQN权重.bat       : 运行系统(DQN权重)
echo   - scripts\训练权重DQN-模拟.bat       : 训练权重DQN(模拟)
echo   - scripts\训练权重DQN-真实环境.bat   : 训练权重DQN(真实环境)
echo   - scripts\测试权重DQN模型.bat        : 测试权重DQN
echo   - scripts\训练移动DQN-模拟.bat       : 训练移动DQN(模拟)
echo   - scripts\训练移动DQN-真实环境.bat   : 训练移动DQN(真实环境)
echo   - scripts\测试移动DQN模型.bat        : 测试移动DQN
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
echo DQN模型:
if exist "multirotor\DQN_Weight\models\weight_predictor_simple.zip" (
    echo [OK] 权重APF模型已训练
) else (
    echo [!] 权重APF模型未训练，请运行选项 [3] 训练
)
if exist "multirotor\DQN_Movement\models\movement_dqn_final.zip" (
    echo [OK] 移动控制模型已训练
) else (
    echo [!] 移动控制模型未训练，请运行选项 [6] 训练
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

