"""
DQN (Deep Q-Network) 学习算法实现
包含DQN神经网络、经验回放缓冲区和DQN智能体
"""
import os
# 强制禁用CUDA，避免在没有GPU的环境中卡住
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 设置PyTorch使用CPU并限制线程数
torch.set_num_threads(2)  # 限制线程数，避免占用所有CPU核心


class DQN(nn.Module):
    """DQN神经网络
    输入：状态向量
    输出：每个动作的Q值
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        初始化DQN网络
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super(DQN, self).__init__()
        
        # 定义网络结构：3层全连接网络
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """前向传播"""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """经验回放缓冲区
    存储和采样(s, a, r, s', done)经验元组
    """

    def __init__(self, capacity=10000):
        """
        初始化经验回放缓冲区
        参数:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """
        添加一条经验到缓冲区
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device=torch.device('cpu')):
        """
        从缓冲区随机采样一批数据
        参数:
            batch_size: 批次大小
            device: 目标设备 (CPU或CUDA)
        返回:
            states, actions, rewards, next_states, dones的批次数据
        """
        # 随机采样
        batch = random.sample(self.buffer, batch_size)
        
        # 解包并转换为张量，直接放到指定设备上
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """返回缓冲区当前的大小"""
        return len(self.buffer)


class DQNAgent:
    """DQN智能体
    实现DQN算法的核心逻辑，包括选择动作、学习等功能
    """

    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update=10, memory_capacity=10000):
        """
        初始化DQN智能体
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            lr: 学习率
            gamma: 折扣因子
            epsilon: 探索率初始值
            epsilon_min: 探索率最小值
            epsilon_decay: 探索率衰减因子
            batch_size: 批次大小
            target_update: 目标网络更新频率
            memory_capacity: 经验回放缓冲区容量
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma  # 折扣因子，用于计算未来奖励的现值
        self.epsilon = epsilon  # 探索率，用于ε-贪婪策略
        self.epsilon_min = epsilon_min  # 最小探索率
        self.epsilon_decay = epsilon_decay  # 探索率衰减率
        self.batch_size = batch_size  # 训练批次大小
        self.target_update = target_update  # 目标网络更新步数间隔

        # 设备管理：强制使用CPU，避免在没有GPU的环境中卡住
        self.device = torch.device('cpu')
        print(f"DQN使用设备: {self.device}")
        
        # 创建策略网络和目标网络
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        # 初始化目标网络的参数与策略网络相同
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络设置为评估模式

        # 创建优化器和损失函数
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()  # 使用均方误差损失函数

        # 创建经验回放缓冲区
        self.memory = ReplayBuffer(memory_capacity)  # 缓冲区容量可配置

        # 记录训练步数
        self.steps_done = 0

    def select_action(self, state):
        """选择动作（ε-贪婪策略）
        参数:
            state: 当前状态
        返回:
            选择的动作
        """
        # 随机生成一个数，如果小于epsilon则进行探索
        if random.random() < self.epsilon:
            # 探索：随机选择一个动作
            return random.randrange(self.action_dim)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 移到指定设备
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()  # 返回Q值最大的动作索引

    def learn(self):
        """从经验回放缓冲区中学习，更新策略网络"""
        # 如果缓冲区中的样本数量不足一个批次，则不进行学习
        if len(self.memory) < self.batch_size:
            return

        # 从缓冲区中采样一批数据，数据已在正确的设备上
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)

        # 计算当前状态下的Q值
        current_q = self.policy_net(states).gather(1, actions)

        # 计算下一状态的最大Q值（使用目标网络，并且不参与梯度计算）
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            # 计算目标Q值：r + γ * maxQ'(s',a') * (1-done)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算损失并更新网络参数
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        
        # 梯度裁剪，防止梯度爆炸（对CPU训练特别重要）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()  # 更新参数

        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 定期更新目标网络
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        """保存模型参数
        参数:
            path: 保存路径
        """
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        """加载模型参数
        参数:
            path: 加载路径
        """
        # 强制使用CPU加载，避免在没有GPU的环境中出错
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 示例使用代码
if __name__ == "__main__":
    # 创建一个简单的DQN智能体示例
    state_dim = 18  # 状态维度
    action_dim = 25  # 动作维度
    
    # 创建DQN智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update=10,
        memory_capacity=10000
    )
    
    print(f"DQN智能体创建成功")
    print(f"状态维度: {agent.state_dim}")
    print(f"动作维度: {agent.action_dim}")
    print(f"学习率: {agent.lr}")
    print(f"探索率: {agent.epsilon}")
    
    # 模拟训练过程
    print("\n开始模拟训练...")
    for episode in range(10):
        # 随机生成状态
        state = np.random.randn(state_dim).astype(np.float32)
        
        # 选择动作
        action = agent.select_action(state)
        
        # 模拟环境反馈
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = False
        
        # 存储经验
        agent.memory.push(state, action, reward, next_state, done)
        
        # 学习
        if len(agent.memory) >= agent.batch_size:
            agent.learn()
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}: 探索率 = {agent.epsilon:.4f}, 训练步数 = {agent.steps_done}")
    
    print("\n训练完成！")

