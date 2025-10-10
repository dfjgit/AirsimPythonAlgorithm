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


# 创建早停回调类
class EarlyStoppingCallback(BaseCallback):
    def __init__(self, check_interval=100, window_size=1000, threshold=0.05, verbose=1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.check_interval = check_interval  # 检查间隔（局数）
        self.window_size = window_size  # 检查的窗口大小
        self.threshold = threshold  # 得分变化阈值
        self.scores = []  # 记录最近的得分
        self.episode_count = 0  # 记录局数

    def _on_step(self) -> bool:
        # 获取当前局的信息
        infos = self.locals.get('infos')
        if infos is not None and len(infos) > 0:
            # 检查是否有episode结束的信息
            if 'episode' in infos[0]:
                score = infos[0]['episode']['r']
                self.scores.append(score)
                self.episode_count += 1

                # 只保留最近window_size个得分
                if len(self.scores) > self.window_size:
                    self.scores.pop(0)

                # 检查是否需要停止训练
                if self.episode_count % self.check_interval == 0 and len(self.scores) == self.window_size:
                    # 计算得分的标准差
                    std_dev = np.std(self.scores)
                    # 计算得分的平均值
                    mean_score = np.mean(self.scores)

                    # 计算相对标准差（变异系数）
                    if mean_score > 0:
                        relative_std = std_dev / mean_score
                    else:
                        relative_std = std_dev  # 如果平均分为0，直接使用标准差

                    if self.verbose > 0:
                        print(f"\n早停检查 - 局数: {self.episode_count}")
                        print(f"最近{self.window_size}局平均得分: {mean_score:.2f}")
                        print(f"标准差: {std_dev:.4f}")
                        print(f"相对标准差: {relative_std:.4f}")
                        print(f"阈值: {self.threshold}")

                    # 如果相对标准差小于阈值，停止训练
                    if relative_std < self.threshold:
                        if self.verbose > 0:
                            print(
                                f"\n训练停止：最近{self.window_size}局得分几乎一样 (相对标准差 {relative_std:.4f} < 阈值 {self.threshold})")
                        # 保存当前模型
                        self.model.save("dqn_breakout_model_early_stopped")
                        if self.verbose > 0:
                            print("模型已保存为: dqn_breakout_model_early_stopped.zip")
                        return False
        return True


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
        self.ball_dx = self.BALL_SPEED_X  # x方向速度
        self.ball_dy = -self.BALL_SPEED_Y  # y方向速度（初始向上）

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
        """计算奖励值"""
        # 1. 基础移动奖励（保持小球在空中）
        reward = 0.01

        # 2. 击中砖块奖励
        if bricks_hit > 0:
            reward += bricks_hit * 1.0

        # 3. 小球掉落惩罚
        if self.ball_y >= self.SCREEN_HEIGHT:
            reward -= 10.0

        # 4. 胜利奖励
        if self._check_win():
            reward += 100.0

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
                "MlpPolicy",
                env,
                verbose=1,
                buffer_size=100000,
                learning_rate=5e-4,
                batch_size=64,
                gamma=0.99,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                target_update_interval=1000,
                train_freq=4
            )
    else:
        # 3. 创建新的DQN模型
        print("创建新模型，开始训练...")
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            buffer_size=100000,
            learning_rate=5e-4,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            target_update_interval=1000,
            train_freq=4
        )

    # 4. 创建早停回调实例
    early_stopping_callback = EarlyStoppingCallback(
        check_interval=100,  # 每100局检查一次
        window_size=1000,    # 检查最近1000局
        threshold=0.05,      # 相对标准差阈值为5%
        verbose=1            # 显示详细信息
    )

    # 5. 训练模型，传入早停回调
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10,
        callback=early_stopping_callback
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