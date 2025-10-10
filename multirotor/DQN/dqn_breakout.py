import pygame
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import time





class BreakoutEnv(gym.Env):
    """打砖块游戏的强化学习环境"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(BreakoutEnv, self).__init__()

        # 1. 游戏常量设置
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 8
        self.BALL_SIZE = 15
        self.BALL_SPEED_X = 5
        self.BALL_SPEED_Y = 5
        self.BRICK_WIDTH = 70
        self.BRICK_HEIGHT = 20
        self.BRICK_GAP = 5
        self.BRICK_ROWS = 5
        self.BRICK_COLS = 10
        self.BRICK_TOP_MARGIN = 50

        # 2. 观察空间：挡板位置、小球位置和速度、砖块状态
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.BRICK_ROWS * self.BRICK_COLS + 4,),  # 砖块状态(50) + 挡板位置(1) + 小球位置(2) + 小球速度方向(1)
            dtype=np.float32
        )

        # 3. 动作空间：0=左移, 1=不动, 2=右移
        self.action_space = spaces.Discrete(3)

        # 4. 渲染模式
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # 5. 初始化游戏状态
        self.reset()

    def reset(self, seed=None, options=None):
        """重置游戏状态"""
        super().reset(seed=seed)

        # 1. 初始化挡板位置（居中底部）
        self.paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) // 2
        self.paddle_y = self.SCREEN_HEIGHT - 40

        # 2. 初始化小球位置（初始位置在挡板中央上方）
        self.ball_x = self.paddle_x + self.PADDLE_WIDTH // 2 - self.BALL_SIZE // 2
        self.ball_y = self.paddle_y - self.BALL_SIZE
        # 初始x方向速度随机在-5到5之间
        self.ball_dx = np.random.uniform(-5, 5)
        self.ball_dy = -self.BALL_SPEED_Y  # y方向速度始终向上
        
        # 初始化prev_ball_dy属性，用于奖励函数中的小球反弹检测
        self.prev_ball_dy = self.ball_dy

        # 3. 初始化砖块矩阵
        self.bricks = []
        for row in range(self.BRICK_ROWS):
            brick_row = []
            for col in range(self.BRICK_COLS):
                brick_x = col * (self.BRICK_WIDTH + self.BRICK_GAP) + \
                          (self.SCREEN_WIDTH - self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_GAP)) // 2
                brick_y = row * (self.BRICK_HEIGHT + self.BRICK_GAP) + self.BRICK_TOP_MARGIN
                brick_row.append({"rect": pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH, self.BRICK_HEIGHT),
                                  "active": True})
            self.bricks.append(brick_row)

        # 4. 初始化得分和步数
        self.score = 0
        self.steps = 0
        self.max_steps = 5000  # 每局最大步数

        # 5. 初始化渲染
        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("强化学习打砖块")
            self.clock = pygame.time.Clock()

        return self._get_obs(), {}

    def _get_obs(self):
        """获取当前观察状态"""
        # 1. 砖块状态向量
        bricks_state = []
        for row in self.bricks:
            for brick in row:
                bricks_state.append(1.0 if brick["active"] else 0.0)

        # 2. 挡板位置归一化
        paddle_pos = [self.paddle_x / self.SCREEN_WIDTH]

        # 3. 小球位置归一化
        ball_pos = [self.ball_x / self.SCREEN_WIDTH, self.ball_y / self.SCREEN_HEIGHT]

        # 4. 小球速度方向（-1, 0, 1表示左、无、右）
        ball_dir = [1.0 if self.ball_dx > 0 else 0.0 if self.ball_dx == 0 else -1.0]

        # 5. 合并所有状态
        obs = np.array(bricks_state + paddle_pos + ball_pos + ball_dir, dtype=np.float32)
        return obs

    def _get_reward(self, bricks_hit):
        """计算奖励值 - 优化版本"""
        # 1. 基础生存奖励（随着时间增长，鼓励更长时间存活）
        # 初始值较小，防止消极策略，同时随步数缓慢增加
        reward = 0.05 + min(self.steps * 0.0001, 0.2)  # 最大0.25

        # 2. 击中砖块奖励 - 增加权重，使摧毁砖块更有吸引力
        if bricks_hit > 0:
            # 考虑到当前实现中每次最多击中一个砖块
            reward += 5.0  # 击中单个砖块的奖励
            # 额外奖励：随着剩余砖块减少，每个砖块的价值增加
            remaining_bricks = sum(1 for row in self.bricks for brick in row if brick['active'])
            reward += (50 - remaining_bricks) * 0.1  # 剩余砖块越少，奖励越多

        # 3. 成功接住小球奖励 - 直接奖励接住行为
        if self.prev_ball_dy > 0 and self.ball_dy < 0:  # 小球刚刚从向下变为向上
            # 检查是否是被挡板接住的（位置接近挡板）
            if abs(self.ball_y + self.BALL_SIZE - self.paddle_y) < self.BALL_SIZE * 2:
                reward += 8.0  # 接住小球的直接奖励

        # 4. 小球掉落惩罚 - 调整为更合适的值
        if self.ball_y >= self.SCREEN_HEIGHT:
            reward -= 80.0  # 减少惩罚强度，避免过度惩罚

        # 5. 胜利奖励 - 增加权重并添加额外奖励
        if self._check_win():
            reward += 1500.0  # 增加胜利奖励
            # 添加剩余步数奖励，鼓励更快完成
            reward += max(0, (self.max_steps - self.steps) * 0.1)  # 剩余每步额外加0.1

        # 6. 挡板与小球距离奖励 - 优化计算方式
        if self.ball_dy > 0 and self.ball_y > self.SCREEN_HEIGHT * 0.4:  # 小球向下移动且在屏幕下半部分
            # 计算挡板中心和小球中心的水平距离
            paddle_center_x = self.paddle_x + self.PADDLE_WIDTH / 2
            ball_center_x = self.ball_x + self.BALL_SIZE / 2
            distance = abs(paddle_center_x - ball_center_x)
            
            # 使用挡板宽度作为参考，而非屏幕宽度
            max_distance = self.SCREEN_WIDTH / 2  # 最大可能距离
            threshold_distance = self.PADDLE_WIDTH * 1.5  # 有效距离阈值
            
            # 非线性距离奖励：距离越近，奖励增长越快
            if distance < threshold_distance:
                # 使用二次函数增强近距离奖励
                normalized_distance = distance / threshold_distance
                distance_reward = 5.0 * (1 - normalized_distance ** 2)  # 最大5.0
                reward += distance_reward

        # 保存当前小球方向供下次判断使用
        self.prev_ball_dy = self.ball_dy

        # 确保奖励在合理范围内
        reward = np.clip(reward, -100.0, 2000.0)
        
        return reward

    def _check_win(self):
        """检查是否所有砖块都被消除"""
        for row in self.bricks:
            for brick in row:
                if brick["active"]:
                    return False
        return True

    def step(self, action):
        """执行一步动作"""
        # 1. 根据动作移动挡板
        if action == 0 and self.paddle_x > 0:  # 左移
            self.paddle_x -= self.PADDLE_SPEED
        elif action == 2 and self.paddle_x < self.SCREEN_WIDTH - self.PADDLE_WIDTH:  # 右移
            self.paddle_x += self.PADDLE_SPEED
        # 动作1表示不动

        # 2. 移动小球
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # 3. 小球与边界碰撞检测
        if self.ball_x <= 0 or self.ball_x >= self.SCREEN_WIDTH - self.BALL_SIZE:
            self.ball_dx = -self.ball_dx

        if self.ball_y <= 0:
            self.ball_dy = -self.ball_dy

        # 4. 小球与挡板碰撞检测
        paddle_rect = pygame.Rect(self.paddle_x, self.paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE)

        if ball_rect.colliderect(paddle_rect) and self.ball_dy > 0:
            self.ball_dy = -self.ball_dy
            # 小球反弹方向根据击中挡板的位置略微调整
            hit_pos = (self.ball_x + self.BALL_SIZE // 2 - self.paddle_x) / self.PADDLE_WIDTH
            self.ball_dx = (hit_pos - 0.5) * 2 * self.BALL_SPEED_X

        # 5. 小球与砖块碰撞检测
        bricks_hit = 0
        for row in self.bricks:
            for brick in row:
                if brick["active"] and ball_rect.colliderect(brick["rect"]):
                    brick["active"] = False
                    self.ball_dy = -self.ball_dy
                    bricks_hit += 1
                    break
            if bricks_hit > 0:
                break

        # 6. 更新得分和步数
        self.score += bricks_hit
        self.steps += 1

        # 7. 计算奖励
        reward = self._get_reward(bricks_hit)

        # 8. 检查终止条件
        terminated = False
        truncated = False

        if self.ball_y >= self.SCREEN_HEIGHT:  # 小球落地
            terminated = True
        elif self._check_win():  # 所有砖块被消除
            terminated = True
        elif self.steps >= self.max_steps:  # 达到最大步数
            truncated = True

        # 9. 渲染（如果需要）
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {}

    def _render_frame(self):
        """渲染游戏画面"""
        if self.screen is None:
            return

        # 1. 绘制背景
        self.screen.fill((0, 0, 0))  # 黑色背景

        # 2. 绘制挡板
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.paddle_x, self.paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT))

        # 3. 绘制小球
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (self.ball_x + self.BALL_SIZE // 2, self.ball_y + self.BALL_SIZE // 2),
                           self.BALL_SIZE // 2)

        # 4. 绘制砖块
        for row_idx, row in enumerate(self.bricks):
            for brick in row:
                if brick["active"]:
                    # 不同行砖块用不同颜色
                    color = (0, 255 - row_idx * 40, 255)
                    pygame.draw.rect(self.screen, color, brick["rect"])

        # 5. 显示得分
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        # 6. 更新屏幕
        pygame.display.flip()

        # 7. 控制帧率
        if self.clock is not None:
            self.clock.tick(self.metadata["render_fps"])

        # 8. 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def render(self):
        """渲染接口"""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        """关闭游戏"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


class SimpleProgressBarCallback(BaseCallback):
    """自定义的简单进度条回调类，不依赖外部库"""
    def __init__(self, total_timesteps, update_freq=1000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.start_time = None
        self.last_update_time = None
    
    def _on_training_start(self):
        self.start_time = time.time()
        self.last_update_time = self.start_time
        print("开始训练...")
    
    def _on_step(self):
        # 每update_freq步更新一次进度条
        if self.num_timesteps % self.update_freq == 0 or self.num_timesteps == self.total_timesteps:
            self._update_progress()
        return True
    
    def _update_progress(self):
        # 计算进度百分比
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        percentage = int(progress * 100)
        
        # 计算已用时间和预计剩余时间
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 只在有进度时计算预计剩余时间
        if self.num_timesteps > 0:
            steps_per_second = self.num_timesteps / elapsed_time
            remaining_steps = self.total_timesteps - self.num_timesteps
            remaining_time = remaining_steps / steps_per_second if steps_per_second > 0 else 0
        else:
            remaining_time = 0
        
        # 格式化时间显示
        elapsed_str = self._format_time(elapsed_time)
        remaining_str = self._format_time(remaining_time)
        
        # 绘制进度条
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # 清除当前行并显示新进度
        sys.stdout.write('\r')
        sys.stdout.write(f'训练进度: |{bar}| {percentage}% [{self.num_timesteps}/{self.total_timesteps}] 已用时间: {elapsed_str} 预计剩余: {remaining_str}')
        sys.stdout.flush()
        
    def _format_time(self, seconds):
        """将秒数格式化为时:分:秒"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h{minutes}m{seconds}s"
        elif minutes > 0:
            return f"{minutes}m{seconds}s"
        else:
            return f"{seconds}s"
    
    def _on_training_end(self):
        # 确保显示100%进度
        self._update_progress()
        print("\n训练完成!")

def train_agent(resume_training=False, total_timesteps=1000000):
    """训练DQN智能体"""
    # 1. 创建环境
    env = BreakoutEnv()
    env = Monitor(env)

    if resume_training:
        # 2. 加载已有模型
        try:
            print("加载已有模型，继续训练...")
            model = DQN.load("dqn_breakout_model", env=env)
            print(f"模型已加载，将继续训练 {total_timesteps} 步")
        except FileNotFoundError:
            print("未找到模型文件，将创建新模型开始训练")
            # 如果找不到模型文件，创建新模型
            model = DQN(
                "MlpPolicy",              # 策略网络类型：多层感知器策略
                env,                       # 环境实例
                verbose=0,                 # 日志级别：1表示显示训练过程中的信息
                buffer_size=100000,        # 经验回放缓冲区大小：存储100000条经验
                learning_rate=5e-4,        # 学习率：控制参数更新步长
                batch_size=64,             # 批次大小：每次训练使用64条经验
                gamma=0.99,                # 折扣因子：控制未来奖励的现值，接近1表示重视未来奖励
                exploration_fraction=0.1,  # 探索率衰减周期：总训练步数的10%用于衰减探索率
                exploration_initial_eps=1.0, # 初始探索率：1.0表示完全随机选择动作
                exploration_final_eps=0.05,  # 最终探索率：0.05表示训练结束后仍有5%的概率随机探索
                target_update_interval=1000, # 目标网络更新间隔：每1000步更新一次目标网络
                train_freq=4               # 训练频率：每4步训练一次网络
            )
    else:
        # 3. 创建新的DQN模型
        print("创建新模型，开始训练...")
        model = DQN(
            "MlpPolicy",              # 策略网络类型：多层感知器策略
            env,                       # 环境实例
            verbose=0,                 # 日志级别：1表示显示训练过程中的信息
            buffer_size=100000,        # 经验回放缓冲区大小：存储100000条经验
            learning_rate=5e-4,        # 学习率：控制参数更新步长
            batch_size=64,             # 批次大小：每次训练使用64条经验
            gamma=0.99,                # 折扣因子：控制未来奖励的现值，接近1表示重视未来奖励
            exploration_fraction=0.1,  # 探索率衰减周期：总训练步数的10%用于衰减探索率
            exploration_initial_eps=1.0, # 初始探索率：1.0表示完全随机选择动作
            exploration_final_eps=0.05,  # 最终探索率：0.05表示训练结束后仍有5%的概率随机探索
            target_update_interval=1000, # 目标网络更新间隔：每1000步更新一次目标网络
            train_freq=4               # 训练频率：每4步训练一次网络
        )

    # 4. 创建自定义进度条回调
    progress_bar = SimpleProgressBarCallback(total_timesteps)
    
    # 5. 训练模型（带进度条）
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10,
        callback=progress_bar
    )

    # 6. 保存模型
    model.save("dqn_breakout_model")
    print("模型已保存为dqn_breakout_model.zip")

    # 7. 关闭环境
    env.close()

    return model

def test_agent():
    """测试训练好的智能体"""
    # 1. 创建环境（带渲染）
    env = BreakoutEnv(render_mode="human")

    # 2. 加载模型
    try:
        model = DQN.load("dqn_breakout_model")
        print("已加载模型，开始测试...")

        # 3. 运行测试
        for episode in range(10):  # 运行10局测试
            obs, _ = env.reset()
            episode_reward = 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                # 使用确定性策略
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

            print(f"测试局 {episode + 1}: 得分 = {episode_reward}")
    except FileNotFoundError:
        print("未找到模型文件，请先训练模型")

    # 4. 关闭环境
    env.close()


def main():
    """主函数"""
    print("强化学习打砖块游戏")
    print("1. 从头开始训练模型")
    print("2. 继续训练已有模型")
    print("3. 测试模型")

    choice = input("请选择操作 (1/2/3): ")

    if choice == "1":
        # 从头开始训练
        total_steps = int(input("请输入训练总步数 (默认为1000000): ") or "1000000")
        train_agent(resume_training=False, total_timesteps=total_steps)
    elif choice == "2":
        # 继续训练已有模型
        total_steps = int(input("请输入要继续训练的步数 (默认为500000): ") or "500000")
        train_agent(resume_training=True, total_timesteps=total_steps)
    elif choice == "3":
        # 测试模型
        test_agent()
    else:
        print("无效选择")


if __name__ == "__main__":
    main()