# DataStruct

这个目录里保留两类内容：

- `DataPacks.cs`: Unity/Python 通信包的结构定义。
- `samples/`: `config_data`、`grid_data`、`runtime_data` 的示例 JSON 载荷，仅用于调试和理解通信格式。

说明：

- `samples/*.json` 不是系统启动时读取的配置文件。
- Python 运行时实际使用的是 `multirotor/apf_algorithm_config.json`、`multirotor/drones_config.json` 以及训练目录下的配置。
