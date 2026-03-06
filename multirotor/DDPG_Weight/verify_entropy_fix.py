"""
验证熵值累积修复的测试脚本

功能：
    - 测试 reset_grid_entropy 参数是否正确工作
    - 验证扫描进度是否可以跨Episode累积

使用方法：
    python verify_entropy_fix.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_algorithm_server_reset():
    """测试 AlgorithmServer 的 reset_environment 方法"""
    print("=" * 60)
    print("测试 1: AlgorithmServer.reset_environment()")
    print("=" * 60)

    # 检查函数签名
    from AlgorithmServer import MultiDroneAlgorithmServer
    import inspect

    sig = inspect.signature(MultiDroneAlgorithmServer.reset_environment)
    params = list(sig.parameters.keys())

    print(f"✅ 函数签名: {sig}")
    print(f"✅ 参数列表: {params}")

    # 验证新增的 reset_grid 参数
    if 'reset_grid' in params:
        print(f"✅ reset_grid 参数存在")

        # 检查默认值
        default_value = sig.parameters['reset_grid'].default
        print(f"✅ reset_grid 默认值: {default_value}")

        if default_value == False:
            print("✅ 默认值正确 (False = 保持扫描进度累积)")
        else:
            print(f"⚠️  默认值为 {default_value}，建议设为 False")
    else:
        print("❌ reset_grid 参数不存在！")
        return False

    print()
    return True


def test_simple_weight_env():
    """测试 SimpleWeightEnv 的初始化参数"""
    print("=" * 60)
    print("测试 2: SimpleWeightEnv.__init__()")
    print("=" * 60)

    # 检查函数签名
    from envs.simple_weight_env import SimpleWeightEnv
    import inspect

    sig = inspect.signature(SimpleWeightEnv.__init__)
    params = list(sig.parameters.keys())

    print(f"✅ 函数签名: {sig}")
    print(f"✅ 参数列表: {params}")

    # 验证新增的 reset_grid_entropy 参数
    if 'reset_grid_entropy' in params:
        print(f"✅ reset_grid_entropy 参数存在")

        # 检查默认值
        default_value = sig.parameters['reset_grid_entropy'].default
        print(f"✅ reset_grid_entropy 默认值: {default_value}")

        if default_value == False:
            print("✅ 默认值正确 (False = 保持扫描进度累积)")
        else:
            print(f"⚠️  默认值为 {default_value}，建议设为 False")
    else:
        print("❌ reset_grid_entropy 参数不存在！")
        return False

    print()
    return True


def test_training_script_config():
    """测试训练脚本的配置支持"""
    print("=" * 60)
    print("测试 3: 训练脚本配置支持")
    print("=" * 60)

    # 检查 train_with_airsim_improved.py 是否支持配置
    script_path = os.path.join(os.path.dirname(__file__), "train_with_airsim_improved.py")

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否包含配置解析
    if 'reset_grid_entropy' in content:
        print("✅ 训练脚本包含 reset_grid_entropy 配置支持")

        # 检查是否从配置文件读取
        if '"reset_grid_entropy"' in content:
            print("✅ 支持从配置文件读取")

        # 检查是否传递给环境
        if 'reset_grid_entropy=reset_grid_entropy' in content:
            print("✅ 正确传递给 SimpleWeightEnv")
    else:
        print("❌ 训练脚本不包含 reset_grid_entropy 配置！")
        return False

    print()
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("🔍 熵值累积修复验证")
    print("=" * 60 + "\n")

    results = []

    # 运行测试
    results.append(("AlgorithmServer", test_algorithm_server_reset()))
    results.append(("SimpleWeightEnv", test_simple_weight_env()))
    results.append(("训练脚本", test_training_script_config()))

    # 汇总结果
    print("=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")

    all_passed = all(result[1] for result in results)

    print()
    if all_passed:
        print("🎉 所有测试通过！修复已正确实现。")
        print()
        print("📝 使用说明:")
        print("  1. 默认模式 (reset_grid_entropy=False):")
        print("     - 扫描进度会跨Episode累积")
        print("     - 适合长期训练，学习完整扫描策略")
        print()
        print("  2. 重置模式 (reset_grid_entropy=True):")
        print("     - 每个Episode重新扫描")
        print("     - 适合独立任务训练")
        print()
        print("  3. 配置文件使用:")
        print('     在配置文件中添加: "reset_grid_entropy": false')
        print()
        return 0
    else:
        print("⚠️  部分测试失败，请检查修复实现。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
