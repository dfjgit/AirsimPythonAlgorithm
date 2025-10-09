import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 设置随机种子，保证结果可复现
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class DQN(nn.Module):
    """DQN网络模型，用于估计Q值
    简单的前馈神经网络，接收状态作为输入，输出每个动作的Q值
    """

    def __init__(self, state_dim, action_dim):
        """
        初始化DQN网络
        参数:
            state_dim: 状态空间的维度
            action_dim: 动作空间的维度
        """
        super(DQN, self).__init__()
        # 定义网络层结构
        self.fc1 = nn.Linear(state_dim, 64)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(64, 64)  # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(64, action_dim)  # 隐藏层2到输出层
        # 使用ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        """前向传播，计算Q值
        参数:
            x: 状态张量
        返回:
            每个动作的Q值
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 输出层不需要激活函数
        return x


class ReplayBuffer:
    """经验回放缓冲区
    用于存储和采样训练数据，打破样本间的相关性
    """

    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        参数:
            capacity: 缓冲区的最大容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)  # 使用双端队列实现

    def push(self, state, action, reward, next_state, done):
        """将经验存入缓冲区
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 执行动作后的状态
            done: 是否结束
        """
        # 将数据转换为numpy数组，方便存储
        experience = (np.array(state), action, reward, np.array(next_state), done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """从缓冲区中随机采样一批数据
        参数:
            batch_size: 采样的批次大小
        返回:
            批量的状态、动作、奖励、下一状态和done标志
        """
        # 随机采样
        experiences = random.sample(self.buffer, batch_size)

        # 将数据分离并转换为PyTorch张量
        states = torch.FloatTensor([exp[0] for exp in experiences])
        actions = torch.LongTensor([exp[1] for exp in experiences]).unsqueeze(1)
        rewards = torch.FloatTensor([exp[2] for exp in experiences]).unsqueeze(1)
        next_states = torch.FloatTensor([exp[3] for exp in experiences])
        dones = torch.FloatTensor([exp[4] for exp in experiences]).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """返回缓冲区当前的大小"""
        return len(self.buffer)


class DQNAgent:
    """DQN智能体
    实现DQN算法的核心逻辑，包括选择动作、学习等功能
    """

    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update=10):
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

        # 创建策略网络和目标网络
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        # 初始化目标网络的参数与策略网络相同
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络设置为评估模式

        # 创建优化器和损失函数
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()  # 使用均方误差损失函数

        # 创建经验回放缓冲区
        self.memory = ReplayBuffer(10000)  # 缓冲区容量为10000

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
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 增加一个维度作为批次
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()  # 返回Q值最大的动作索引

    def learn(self):
        """从经验回放缓冲区中学习，更新策略网络"""
        # 如果缓冲区中的样本数量不足一个批次，则不进行学习
        if len(self.memory) < self.batch_size:
            return

        # 从缓冲区中采样一批数据
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

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
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 示例使用代码
if __name__ == "__main__":
    """这里是一个简单的示例，展示如何使用上面定义的DQN模型
    注意：这只是一个框架，实际使用时需要根据具体环境进行调整
    """
    # 假设环境的状态维度为4，动作维度为2
    state_dim = 4
    action_dim = 2

    # 创建DQN智能体
    agent = DQNAgent(state_dim, action_dim)

    # 模拟训练过程
    episodes = 1000
    for episode in range(episodes):
        # 重置环境，获取初始状态
        # state = env.reset()
        state = np.random.rand(state_dim)  # 这里用随机数模拟初始状态

        total_reward = 0
        done = False

        while not done:
            # 选择动作
            action = agent.select_action(state)

            # 在环境中执行动作，获取下一状态、奖励和done标志
            # next_state, reward, done, _ = env.step(action)
            # 这里用随机数模拟环境反馈
            next_state = np.random.rand(state_dim)
            reward = np.random.randn()
            done = random.random() < 0.1  # 10%的概率结束回合

            # 存储经验到回放缓冲区
            agent.memory.push(state, action, reward, next_state, done)

            # 更新当前状态
            state = next_state
            total_reward += reward

            # 学习
            agent.learn()

        # 每100个回合打印一次结果
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    # 保存模型
    agent.save_model("dqn_model.pth")
