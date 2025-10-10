import gymnasium as gym
from stable_baselines3 import DQN


def main():
    # 使用 MountainCar-v0 或尝试更新的版本
    env = gym.make('MountainCar-v0')
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200000)
    model.save("dqn_mountaincar")
    test_model(model)


def test_model(model):
    # 尝试使用不同的环境版本和渲染模式
    try:
        # 首先尝试 MountainCar-v0
        env = gym.make('MountainCar-v0', render_mode='human')
    except:
        # 如果失败，尝试其他版本
        env = gym.make('MountainCarContinuous-v0', render_mode='human')

    obs, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0

    while not terminated and not truncated:
        env.render()  # 每一步都渲染
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f'Total Reward: {total_reward}')
    env.close()


if __name__ == "__main__":
    main()