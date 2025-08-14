# 安装流程

1. 安装python，运行python-3.13.2-amd64.exe
2. 创建虚拟环境

- 创建虚拟环境：

```bash
 python -m venv myenv
 ```

- 激活虚拟环境：

```bash
.\myenv\Scripts\activate
```

3. 安装环境运行 `./install.bat`

依赖清单：

```bash
    backports.functools-lru-cache==2.0.0  
    backports.ssl_match_hostname==3.7.0.1  
    msgpack-python==0.5.6  
    msgpack-rpc-python==0.4.1  
    numpy==2.2.3  
    opencv-python==4.11.0.86  
    tornado==4.5.3  
```

4.运行Airsim客户端控制无人机

```bash
python .\multirotor\DroneLine.py
```
