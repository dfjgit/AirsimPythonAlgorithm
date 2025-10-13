@echo off
chcp 65001 >nul
:menu
cls
echo ============================================================
echo    AirSim 无人机仿真系统 - 主菜单
echo ============================================================
echo.
echo 请选择操作:
echo.
echo   [1] 运行系统 (固定权重)
echo   [2] 运行系统 (DQN权重预测)
echo.
echo   [3] 训练DQN模型 (模拟数据)
echo   [4] 训练DQN模型 (真实AirSim环境)
echo   [5] 测试DQN模型
echo.
echo   [6] 查看系统信息
echo   [0] 退出
echo.
echo ============================================================
echo.

set /p choice=请输入选项 (0-6): 

if "%choice%"=="1" goto run_normal
if "%choice%"=="2" goto run_dqn
if "%choice%"=="3" goto train
if "%choice%"=="4" goto train_airsim
if "%choice%"=="5" goto test
if "%choice%"=="6" goto info
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
call run_two_drones.bat
goto menu

:run_dqn
cls
echo ============================================================
echo 启动系统 - DQN权重预测模式
echo ============================================================
echo.
call run_with_dqn.bat
goto menu

:train
cls
echo ============================================================
echo DQN模型训练 (模拟数据)
echo ============================================================
echo.
call train_dqn.bat
goto menu

:train_airsim
cls
echo ============================================================
echo DQN模型训练 (真实AirSim环境)
echo ============================================================
echo.
call train_with_airsim.bat
goto menu

:test
cls
echo ============================================================
echo 测试DQN模型
echo ============================================================
echo.
call test_dqn_model.bat
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
echo   - multirotor\DQN\                  : DQN训练和配置
echo   - .venv\                           : Python虚拟环境
echo.
echo 配置文件:
echo   - multirotor\scanner_config.json         : APF算法配置
echo   - multirotor\DQN\dqn_reward_config.json  : DQN奖励配置
echo.
echo 批处理文件:
echo   - run_two_drones.bat    : 运行系统(固定权重)
echo   - run_with_dqn.bat      : 运行系统(DQN权重)
echo   - train_dqn.bat         : 训练DQN(模拟数据)
echo   - train_with_airsim.bat : 训练DQN(真实环境)
echo   - test_dqn_model.bat    : 测试DQN模型
echo   - start.bat             : 主菜单(当前)
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
if exist "multirotor\DQN\models\weight_predictor_simple.zip" (
    echo [OK] 模型已训练
) else (
    echo [!] 模型未训练，请运行选项 [3] 训练模型
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

