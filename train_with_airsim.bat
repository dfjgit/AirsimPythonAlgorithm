@echo off
chcp 65001 >nul
echo ============================================================
echo DQN训练 - 使用真实AirSim环境
echo ============================================================
echo.
echo 本脚本将使用真实的Unity AirSim仿真环境训练DQN模型
echo.
echo 重要提示:
echo   1. 请先启动Unity AirSim仿真场景
echo   2. 确保Unity场景中有无人机和环境
echo   3. 训练过程会实时使用仿真数据
echo   4. 训练时间较长，请保持Unity运行
echo.
echo ============================================================
echo.

REM 检查Unity是否运行的提示
echo [检查] 请确认Unity AirSim仿真是否已运行
echo.
echo 如果Unity未运行:
echo   1. 打开Unity Hub
echo   2. 打开AirSim项目 (Airsim2022)
echo   3. 点击Play按钮启动场景
echo   4. 等待场景完全加载
echo.
set /p confirm=Unity已运行？(Y/N): 
if /i not "%confirm%"=="Y" (
    echo.
    echo 请先启动Unity，然后重新运行本脚本
    pause
    exit /b 1
)

echo.
echo [OK] 继续训练准备...
echo.

REM 激活虚拟环境
echo [1/4] 激活Python虚拟环境...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 虚拟环境激活失败
    echo 请先运行 run_two_drones.bat 确保虚拟环境已创建
    pause
    exit /b 1
)
echo [OK] 虚拟环境已激活
echo.

REM 检查依赖
echo [2/4] 检查训练依赖...
python -c "import torch; import stable_baselines3; print('[OK] 依赖检查通过')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 警告: 缺少必要的依赖库
    echo 正在安装依赖...
    pip install torch stable-baselines3 -i https://pypi.tuna.tsinghua.edu.cn/simple
)
echo.

REM 切换到DQN目录
echo [3/4] 切换到DQN目录...
cd multirotor\DQN
echo [OK] 当前目录: %CD%
echo.

REM 开始训练
echo [4/4] 开始DQN训练 (使用真实AirSim环境)
echo ============================================================
echo.
echo 训练配置:
echo   - 环境: 真实Unity AirSim仿真
echo   - 无人机: 1台 (UAV1)
echo   - 训练步数: 100000 步
echo   - 可视化: 启用
echo   - 模型保存: 自动保存最佳模型和检查点
echo.
echo 训练控制:
echo   - 按 Ctrl+C 可以随时停止训练
echo   - 训练会自动保存进度
echo   - 可以从检查点继续训练
echo.
echo ============================================================
echo.

python train_with_airsim.py

REM 检查训练结果
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo [SUCCESS] 训练成功完成！
    echo ============================================================
    echo.
    echo 模型文件:
    echo   - 最终模型: models\weight_predictor_airsim.zip
    echo   - 最佳模型: models\weight_predictor_airsim_best.zip
    echo   - 检查点: models\checkpoint_*.zip
    echo.
    echo 下一步操作:
    echo   1. 测试模型:
    echo      python test_trained_model.py
    echo.
    echo   2. 复制模型为默认模型:
    echo      copy models\weight_predictor_airsim_best.zip models\weight_predictor_simple.zip
    echo.
    echo   3. 使用训练好的模型:
    echo      cd ..\..
    echo      run_with_dqn.bat
    echo.
) else (
    echo.
    echo ============================================================
    echo [ERROR] 训练失败或被中断
    echo ============================================================
    echo.
    echo 可能的原因:
    echo   - Unity AirSim未运行或崩溃
    echo   - 用户手动中断 (Ctrl+C)
    echo   - 网络连接问题
    echo   - 内存不足
    echo.
    echo 如果是手动中断，模型已保存到检查点
    echo 可以查看 models\ 目录中的检查点文件
    echo.
)

echo 按任意键退出...
pause >nul

REM 返回项目根目录
cd ..\..

