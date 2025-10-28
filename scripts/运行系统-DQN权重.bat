@echo off
chcp 65001 >nul
echo ============================================================
echo 启动无人机系统 - 使用DQN权重预测
echo ============================================================
echo.

echo [OK] 虚拟环境已激活
echo.

REM 显示配置信息
echo ============================================================
echo 运行配置
echo ============================================================
echo 模式: DQN权重预测
echo 模型: multirotor\DQN_Weight\models\best_model.zip
echo 奖励配置: multirotor\DQN_Weight\dqn_reward_config.json
echo ============================================================
echo.

REM 运行算法服务器（使用DQN）
echo 启动算法服务器...
python %~dp0..\multirotor\AlgorithmServer.py --use-learned-weights --model-path DQN_Weight/models/best_model --drones 4

echo.
echo ============================================================
echo 程序已退出
echo ============================================================
echo.

pause


