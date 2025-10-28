"""
测试模型加载功能
验证AlgorithmServer能够正确加载不同的DQN模型
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AlgorithmServer import MultiDroneAlgorithmServer

def test_model_loading():
    """测试不同模型路径的加载"""
    print("=" * 60)
    print("🧪 DQN模型加载测试")
    print("=" * 60)
    
    # 测试场景列表
    test_cases = [
        {
            "name": "不使用DQN（固定权重）",
            "use_learned_weights": False,
            "model_path": None
        },
        {
            "name": "自动选择模型",
            "use_learned_weights": True,
            "model_path": None
        },
        {
            "name": "使用best_model",
            "use_learned_weights": True,
            "model_path": "DQN_Weight/models/best_model"
        },
        {
            "name": "使用weight_predictor_airsim",
            "use_learned_weights": True,
            "model_path": "DQN_Weight/models/weight_predictor_airsim"
        },
        {
            "name": "使用weight_predictor_simple",
            "use_learned_weights": True,
            "model_path": "DQN_Weight/models/weight_predictor_simple"
        },
        {
            "name": "使用checkpoint_5000",
            "use_learned_weights": True,
            "model_path": "DQN_Weight/models/checkpoint_5000"
        },
        {
            "name": "不存在的模型（应该失败）",
            "use_learned_weights": True,
            "model_path": "DQN_Weight/models/non_existent_model"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"测试 {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'─' * 60}")
        
        try:
            # 创建服务器实例（不启动，只测试初始化）
            server = MultiDroneAlgorithmServer(
                drone_names=["UAV1"],
                use_learned_weights=test_case['use_learned_weights'],
                model_path=test_case['model_path'],
                enable_visualization=False  # 测试时禁用可视化
            )
            
            # 检查结果
            if test_case['use_learned_weights']:
                if server.weight_model is not None:
                    result = "✅ 成功"
                    print(f"结果: {result} - 模型已加载")
                else:
                    result = "⚠️ 降级"
                    print(f"结果: {result} - 模型加载失败，降级为固定权重")
            else:
                result = "✅ 成功"
                print(f"结果: {result} - 使用固定权重")
            
            results.append({
                "test": test_case['name'],
                "result": result,
                "success": True
            })
            
        except Exception as e:
            result = "❌ 失败"
            print(f"结果: {result}")
            print(f"错误: {str(e)}")
            results.append({
                "test": test_case['name'],
                "result": result,
                "success": False,
                "error": str(e)
            })
    
    # 输出汇总
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    for i, result in enumerate(results, 1):
        status = result['result']
        print(f"{i}. {result['test']}: {status}")
        if 'error' in result:
            print(f"   错误: {result['error']}")
    
    print(f"\n总计: {success_count}/{total_count} 通过")
    print("=" * 60)
    
    # 输出可用模型列表
    print("\n📦 可用模型文件:")
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        if model_files:
            for model_file in sorted(model_files):
                print(f"  - {model_file}")
        else:
            print("  ❌ 没有找到任何.zip模型文件")
    else:
        print(f"  ❌ 模型目录不存在: {models_dir}")
    
    print("\n💡 建议:")
    print("  1. 确保至少有一个模型文件存在（best_model.zip推荐）")
    print("  2. 如果没有模型，请先运行: python train_with_airsim_improved.py")
    print("  3. 使用 --use-learned-weights 时，系统会自动选择最佳模型")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_model_loading()
    except KeyboardInterrupt:
        print("\n\n测试中断")
    except Exception as e:
        print(f"\n\n测试出错: {str(e)}")
        import traceback
        traceback.print_exc()

