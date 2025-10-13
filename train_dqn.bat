@echo off
chcp 65001 >nul
echo ============================================================
echo DQN模型训练 - APF权重学习
echo ============================================================
echo.
echo 本脚本将训练DQN模型用于预测APF权重系数
echo 训练过程可能需要较长时间，请耐心等待
echo.
echo 按任意键开始训练，或按 Ctrl+C 取消...
pause >nul
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
echo [4/4] 开始DQN训练...
echo ============================================================
echo.
python train_simple.py

REM 检查训练结果
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo [SUCCESS] 训练成功完成！
    echo ============================================================
    echo.
    echo 模型已保存到: multirotor\DQN\models\weight_predictor_simple.zip
    echo.
    echo 下一步操作:
    echo   1. 测试模型: python test_trained_model.py
    echo   2. 使用模型: 在启动时添加 --use-learned-weights 参数
    echo   3. 示例: python ..\AlgorithmServer.py --use-learned-weights
    echo.
) else (
    echo.
    echo ============================================================
    echo [ERROR] 训练失败或被中断
    echo ============================================================
    echo.
    echo 可能的原因:
    echo   - 用户手动中断 (Ctrl+C)
    echo   - 依赖库版本不兼容
    echo   - 内存不足
    echo.
    echo 请检查上方的错误信息
    echo.
)

echo 按任意键退出...
pause >nul

REM 返回项目根目录
cd ..\..

