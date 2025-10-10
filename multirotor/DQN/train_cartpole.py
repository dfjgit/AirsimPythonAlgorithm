import gymnasium as gym  # 将Gym替换为Gymnasium
from stable_baselines3 import PPO


def main():
    env = gym.make('CartPole-v1')  # 创建环境
    model = PPO("MlpPolicy", env, verbose=1)  # 创建模型
    model.learn(total_timesteps=20000)  # 训练模型
    model.save("ppo_cartpole")  # 保存模型
    test_model(model)  # 测试模型


def test_model(model):
    env = gym.make('CartPole-v1', render_mode='human')  # 可视化只能在初始化时指定
    obs, info = env.reset()  # Gymnasium的reset返回两个值
    terminated, truncated = False, False  # 使用terminated和truncated替代done1和done2
    total_reward = 0

    while not terminated and not truncated:  # 修正终止条件判断
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)  # step返回五个值
        total_reward += reward

    print(f'Total Reward: {total_reward}')
    env.close()


if __name__ == "__main__":
    main()