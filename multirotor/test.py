import setup_path
import airsim
import numpy as np
import os
import json
import socket
import threading
import base64
import io
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict


try:
    # 显式指定IP和端口
    client = airsim.MultirotorClient()
    client.confirmConnection()  # 尝试连接
    print("连接AirSim成功！")
except Exception as e:
    print(f"连接失败: {e}")