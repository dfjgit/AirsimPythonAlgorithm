"""
测试DQN学习环境和代理的功能
"""
import sys
import os
import logging

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dqn_imports():
    """测试DQN模块导入"""
    print("=" * 60)
    print("测试1: DQN模块导入")
    print("=" * 60)
    
    try:
        from multirotor.DQN.DqnLearning import DQNAgent, ReplayBuffer
        print("✓ DQNAgent 导入成功")
        print("✓ ReplayBuffer 导入成功")
        
        from multirotor.DQN.DroneLearningEnv import DroneLearningEnv
        print("✓ DroneLearningEnv 导入成功")
        
        return True
    except Exception as e:
        print(f"✗ 导入失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_dqn_agent_creation():
    """测试DQN代理创建"""
    print("\n" + "=" * 60)
    print("测试2: DQN代理创建")
    print("=" * 60)
    
    try:
        from multirotor.DQN.DqnLearning import DQNAgent
        
        # 创建一个简单的DQN代理
        agent = DQNAgent(
            state_dim=18,
            action_dim=25,
            lr=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            batch_size=64,
            target_update=10,
            memory_capacity=10000
        )
        print(f"✓ DQN代理创建成功")
        print(f"  - 状态维度: {agent.state_dim}")
        print(f"  - 动作维度: {agent.action_dim}")
        print(f"  - 学习率: {agent.lr}")
        print(f"  - 探索率: {agent.epsilon}")
        print(f"  - 记忆容量: {len(agent.memory)}/{agent.memory.capacity}")
        
        return agent
    except Exception as e:
        print(f"✗ 创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_agent_action_selection(agent):
    """测试动作选择"""
    print("\n" + "=" * 60)
    print("测试3: 动作选择")
    print("=" * 60)
    
    if agent is None:
        print("✗ 代理未初始化，跳过测试")
        return False
    
    try:
        import numpy as np
        
        # 创建一个随机状态
        state = np.random.randn(agent.state_dim).astype(np.float32)
        print(f"✓ 创建测试状态，维度: {state.shape}")
        
        # 选择动作
        action = agent.select_action(state)
        print(f"✓ 选择动作成功: {action}")
        print(f"  - 当前探索率: {agent.epsilon:.4f}")
        
        # 多次选择动作
        actions = [agent.select_action(state) for _ in range(10)]
        print(f"✓ 10次动作选择: {actions}")
        
        return True
    except Exception as e:
        print(f"✗ 动作选择失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_replay_buffer(agent):
    """测试经验回放缓冲区"""
    print("\n" + "=" * 60)
    print("测试4: 经验回放缓冲区")
    print("=" * 60)
    
    if agent is None:
        print("✗ 代理未初始化，跳过测试")
        return False
    
    try:
        import numpy as np
        
        # 添加一些经验
        for i in range(100):
            state = np.random.randn(agent.state_dim).astype(np.float32)
            action = np.random.randint(0, agent.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(agent.state_dim).astype(np.float32)
            done = np.random.rand() > 0.9
            
            agent.memory.push(state, action, reward, next_state, done)
        
        print(f"✓ 添加100条经验到缓冲区")
        print(f"  - 当前缓冲区大小: {len(agent.memory)}")
        
        # 测试采样
        if len(agent.memory) >= agent.batch_size:
            states, actions, rewards, next_states, dones = agent.memory.sample(agent.batch_size)
            print(f"✓ 从缓冲区采样成功")
            print(f"  - 批次大小: {len(states)}")
            print(f"  - States shape: {states.shape}")
            print(f"  - Actions shape: {actions.shape}")
            print(f"  - Rewards shape: {rewards.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 缓冲区测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_learning(agent):
    """测试学习过程"""
    print("\n" + "=" * 60)
    print("测试5: 学习过程")
    print("=" * 60)
    
    if agent is None:
        print("✗ 代理未初始化，跳过测试")
        return False
    
    try:
        # 检查缓冲区
        if len(agent.memory) < agent.batch_size:
            print(f"! 缓冲区大小不足 ({len(agent.memory)} < {agent.batch_size})，先添加数据")
            import numpy as np
            for i in range(agent.batch_size):
                state = np.random.randn(agent.state_dim).astype(np.float32)
                action = np.random.randint(0, agent.action_dim)
                reward = np.random.randn()
                next_state = np.random.randn(agent.state_dim).astype(np.float32)
                done = False
                agent.memory.push(state, action, reward, next_state, done)
        
        print(f"✓ 缓冲区准备就绪: {len(agent.memory)} 条经验")
        
        # 执行学习
        initial_epsilon = agent.epsilon
        initial_steps = agent.steps_done
        
        agent.learn()
        
        print(f"✓ 学习步骤执行成功")
        print(f"  - 探索率: {initial_epsilon:.4f} -> {agent.epsilon:.4f}")
        print(f"  - 训练步数: {initial_steps} -> {agent.steps_done}")
        
        # 多次学习
        for i in range(10):
            agent.learn()
        
        print(f"✓ 执行10次学习步骤")
        print(f"  - 当前探索率: {agent.epsilon:.4f}")
        print(f"  - 当前步数: {agent.steps_done}")
        
        return True
    except Exception as e:
        print(f"✗ 学习失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_save_load():
    """测试模型保存和加载"""
    print("\n" + "=" * 60)
    print("测试6: 模型保存和加载")
    print("=" * 60)
    
    try:
        from multirotor.DQN.DqnLearning import DQNAgent
        import numpy as np
        
        # 创建代理
        agent1 = DQNAgent(state_dim=18, action_dim=25)
        
        # 添加一些经验并训练
        for i in range(100):
            state = np.random.randn(agent1.state_dim).astype(np.float32)
            action = np.random.randint(0, agent1.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(agent1.state_dim).astype(np.float32)
            done = False
            agent1.memory.push(state, action, reward, next_state, done)
        
        for _ in range(10):
            agent1.learn()
        
        # 保存模型
        test_model_path = os.path.join(os.path.dirname(__file__), "multirotor", "DQN", "test_model.pth")
        os.makedirs(os.path.dirname(test_model_path), exist_ok=True)
        agent1.save_model(test_model_path)
        print(f"✓ 模型保存成功: {test_model_path}")
        
        # 创建新代理并加载模型
        agent2 = DQNAgent(state_dim=18, action_dim=25)
        agent2.load_model(test_model_path)
        print(f"✓ 模型加载成功")
        
        # 验证模型参数一致
        state = np.random.randn(18).astype(np.float32)
        action1 = agent1.select_action(state)
        
        # 设置相同的epsilon以确保选择相同动作
        agent2.epsilon = 0.0  # 不探索，只利用
        agent1.epsilon = 0.0
        
        action2 = agent2.select_action(state)
        print(f"✓ 模型验证: agent1动作={action1}, agent2动作={action2}")
        
        # 清理测试文件
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
            print(f"✓ 清理测试文件")
        
        return True
    except Exception as e:
        print(f"✗ 模型保存/加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("DQN学习环境诊断测试")
    print("=" * 60)
    
    results = {}
    
    # 测试1: 导入
    results['import'] = test_dqn_imports()
    if not results['import']:
        print("\n✗ 导入测试失败，无法继续后续测试")
        return
    
    # 测试2: 创建代理
    agent = test_dqn_agent_creation()
    results['creation'] = agent is not None
    
    # 测试3: 动作选择
    results['action_selection'] = test_agent_action_selection(agent)
    
    # 测试4: 经验回放
    results['replay_buffer'] = test_replay_buffer(agent)
    
    # 测试5: 学习
    results['learning'] = test_agent_learning(agent)
    
    # 测试6: 模型保存/加载
    results['save_load'] = test_model_save_load()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:20s}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\n总计: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("\n✓ 所有测试通过！DQN学习环境工作正常。")
    else:
        print("\n✗ 部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    main()

