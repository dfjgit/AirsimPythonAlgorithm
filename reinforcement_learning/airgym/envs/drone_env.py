# 导入必要的库和模块\import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimDroneEnv(AirSimEnv):
    """基于AirSim的无人机强化学习环境类，继承自AirSimEnv基类"""

    def __init__(self, ip_address, step_length, image_shape):
        """初始化无人机环境

        参数:
            ip_address: AirSim服务器的IP地址
            step_length: 每一步移动的步长
            image_shape: 图像观察的形状尺寸
        """
        # 调用父类构造函数初始化图像观察空间
        super().__init__(image_shape)
        # 设置步长和图像形状
        self.step_length = step_length
        self.image_shape = image_shape

        # 初始化环境状态字典
        self.state = {
            "position": np.zeros(3),  # 无人机当前位置
            "collision": False,  # 碰撞状态标记
            "prev_position": np.zeros(3),  # 无人机上一时刻位置
        }

        # 创建AirSim多旋翼无人机客户端连接
        self.drone = airsim.MultirotorClient(ip=ip_address)
        # 定义离散动作空间，共7个动作
        self.action_space = spaces.Discrete(7)
        # 设置无人机初始飞行状态
        self._setup_flight()

        # 设置深度图像请求参数
        # 3表示相机ID，DepthPerspective表示深度透视图像，True表示启用压缩
        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

    def __del__(self):
        """析构函数，在对象销毁时重置无人机状态"""
        self.drone.reset()

    def _setup_flight(self):
        """设置无人机初始飞行状态和位置"""
        # 重置无人机状态
        self.drone.reset()
        # 启用API控制
        self.drone.enableApiControl(True)
        # 解锁无人机电机
        self.drone.armDisarm(True)

        # 设置初始位置
        self.drone.moveToPositionAsync(-0.55265, -31.9786, -19.0225, 10).join()
        # 设置初始速度
        self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()

    def transform_obs(self, responses):
        """将AirSim返回的原始图像数据转换为强化学习可用的观察格式

        参数:
            responses: AirSim图像请求的响应数据

        返回:
            处理后的图像数据，形状为[84, 84, 1]
        """
        # 从响应中提取浮点型图像数据
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        # 归一化图像数据到0-255范围
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        # 将一维数组重塑为二维图像
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        # 导入PIL库用于图像处理
        from PIL import Image

        # 将numpy数组转换为PIL图像，调整大小为84x84，并转换为灰度图
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        # 返回处理后的图像，添加通道维度
        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        """获取当前环境的观察状态

        返回:
            处理后的图像观察数据
        """
        # 请求获取深度图像
        responses = self.drone.simGetImages([self.image_request])
        # 处理图像数据
        image = self.transform_obs(responses)
        # 获取无人机当前状态
        self.drone_state = self.drone.getMultirotorState()

        # 更新位置信息
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        # 检查碰撞状态
        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        # 返回处理后的图像作为主要观察数据
        return image

    def _do_action(self, action):
        """执行强化学习代理选择的动作

        参数:
            action: 代理选择的离散动作索引
        """
        # 将离散动作转换为三维空间偏移量
        quad_offset = self.interpret_action(action)
        # 获取无人机当前速度
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        # 根据当前速度和动作偏移量，设置新的目标速度
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,  # 动作持续时间(秒)
        ).join()  # 等待动作完成

    def _compute_reward(self):
        """计算当前状态下的奖励值和完成状态

        返回:
            reward: 计算得到的奖励值
            done: 表示回合是否结束的布尔值
        """
        # 设置距离阈值和奖励系数
        thresh_dist = 7  # 距离阈值，超过此值将获得惩罚
        beta = 1  # 距离奖励的系数

        # 预定义的航线路径点
        pts = [
            np.array([-0.55265, -31.9786, -19.0225]),
            np.array([48.59735, -63.3286, -60.07256]),
            np.array([193.5974, -55.0786, -46.32256]),
            np.array([369.2474, 35.32137, -62.5725]),
            np.array([541.3474, 143.6714, -32.07256]),
        ]

        # 获取无人机当前位置坐标
        quad_pt = np.array(
            list(
                (
                    self.state["position"].x_val,
                    self.state["position"].y_val,
                    self.state["position"].z_val,
                )
            )
        )

        # 碰撞惩罚机制
        if self.state["collision"]:
            reward = -100  # 发生碰撞时给予大的负奖励
        else:
            # 计算无人机到航线的最短距离
            dist = 10000000  # 初始化为很大的值
            for i in range(0, len(pts) - 1):
                # 计算点到线段的距离
                dist = min(
                    dist,
                    np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1]))) / np.linalg.norm(
                        pts[i] - pts[i + 1]),
                )

            # 距离过远的惩罚
            if dist > thresh_dist:
                reward = -10  # 偏离航线过远时给予负奖励
            else:
                # 基于距离的奖励：距离越近奖励越高，使用指数函数转换
                reward_dist = math.exp(-beta * dist) - 0.5
                # 基于速度的奖励：鼓励无人机保持一定速度
                reward_speed = (
                        np.linalg.norm(
                            [
                                self.state["velocity"].x_val,
                                self.state["velocity"].y_val,
                                self.state["velocity"].z_val,
                            ]
                        )
                        - 0.5
                )
                # 总奖励为距离奖励和速度奖励之和
                reward = reward_dist + reward_speed

        # 判断回合是否结束
        done = 0
        if reward <= -10:  # 当奖励小于等于-10时，回合结束
            done = 1

        return reward, done

    def step(self, action):
        """强化学习环境的step函数，执行动作并返回下一状态

        参数:
            action: 代理选择的动作

        返回:
            obs: 新的观察状态
            reward: 执行动作后的奖励
            done: 是否结束当前回合
            self.state: 当前环境状态信息
        """
        # 执行选定的动作
        self._do_action(action)
        # 获取执行动作后的观察状态
        obs = self._get_obs()
        # 计算奖励和完成状态
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        """重置环境到初始状态

        返回:
            初始观察状态
        """
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        """将离散动作索引转换为三维空间中的移动偏移量

        参数:
            action: 离散动作索引(0-6)

        返回:
            quad_offset: 三维空间中的移动偏移量(x, y, z)
        """
        if action == 0:
            quad_offset = (self.step_length, 0, 0)  # 沿X轴正方向移动
        elif action == 1:
            quad_offset = (0, self.step_length, 0)  # 沿Y轴正方向移动
        elif action == 2:
            quad_offset = (0, 0, self.step_length)  # 沿Z轴正方向移动(上升)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)  # 沿X轴负方向移动
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)  # 沿Y轴负方向移动
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)  # 沿Z轴负方向移动(下降)
        else:
            quad_offset = (0, 0, 0)  # 保持当前位置不变

        return quad_offset