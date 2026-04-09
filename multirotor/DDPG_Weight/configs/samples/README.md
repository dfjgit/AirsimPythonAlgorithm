# DDPG Weight Config Samples

这里放的是不会被训练入口默认读取的样例文件。

- `last_weights_template.json`: 按无人机分组的权重模板样例，可用来手工准备初始权重文件。

说明：

- 当前主训练入口默认读取的是 `unified_train_config.json`。
- 如果需要手工构造初始权重 JSON，可以参考这里的样例结构。
