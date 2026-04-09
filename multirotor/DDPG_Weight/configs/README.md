# DDPG Weight Configs

当前推荐只看这几个入口：

- `unified_train_config.json`: DDPG 训练统一入口，`train_with_airsim_improved.py`、`train_with_crazyflie_online.py`、`train_with_crazyflie_logs.py`、`train_with_hybrid.py` 都支持从这里读取模式化配置。
- `crazyflie_reward_config.json`: DDPG 权重环境默认奖励配置。
- `legacy/`: 已被统一配置取代的旧训练配置样例，仅用于兼容历史说明或复现实验记录。
- `samples/`: 不会被默认入口读取的辅助样例文件。

说明：

- `unified_train_config.json` 是当前主配置。
- `legacy/*.json` 不再作为批处理脚本默认入口。
- DQN 配置现在统一放在 `multirotor/DQN_Movement/configs/`，不再在此目录单独维护一份奖励配置。
- `samples/last_weights_template.json` 仅作为手工构造初始权重文件时的格式参考。
