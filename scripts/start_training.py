#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DQN训练启动脚本 - Python版本
适用于无法运行批处理文件的情况
"""
import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("DQN训练 - 使用真实AirSim环境")
    print("=" * 60)
    print()
    print("本脚本将使用真实的Unity AirSim仿真环境训练DQN模型")
    print()
    print("重要提示:")
    print("  1. 请先启动Unity AirSim仿真场景")
    print("  2. 确保Unity场景中有无人机和环境")
    print("  3. 训练过程会实时使用仿真数据")
    print("  4. 训练时间较长，请保持Unity运行")
    print()
    print("=" * 60)
    print()
    
    # 检查Unity是否运行
    confirm = input("Unity已运行？(Y/N): ").strip().upper()
    if confirm != 'Y':
        print("\n请先启动Unity，然后重新运行本脚本")
        input("按Enter键退出...")
        return
    
    print("\n[OK] 继续训练准备...\n")
    
    # 检查依赖
    print("[1/3] 检查训练依赖...")
    missing_deps = []
    
    try:
        import torch
        print("[OK] PyTorch已安装")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import stable_baselines3
        print("[OK] Stable-Baselines3已安装")
    except ImportError:
        missing_deps.append("stable-baselines3")
    
    try:
        import gym
        print("[OK] Gym已安装")
    except ImportError:
        missing_deps.append("gym")
    
    try:
        import numpy
        print("[OK] NumPy已安装")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import shimmy
        print("[OK] Shimmy已安装")
    except ImportError:
        missing_deps.append("shimmy")
    
    if missing_deps:
        print(f"\n[错误] 缺少以下依赖: {', '.join(missing_deps)}")
        print("\n请运行以下命令安装:")
        print(f"  pip install {' '.join(missing_deps)}")
        input("\n按Enter键退出...")
        return
    
    print("[OK] 所有依赖检查通过\n")
    
    # 切换到训练脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, '..', 'multirotor', 'DQN_Weight')
    train_script = os.path.join(train_dir, 'train_with_airsim.py')
    
    if not os.path.exists(train_script):
        print(f"[错误] 训练脚本不存在: {train_script}")
        input("\n按Enter键退出...")
        return
    
    # 开始训练
    print("[2/3] 开始DQN训练")
    print("=" * 60)
    print()
    print("训练配置:")
    print("  - 环境: 真实Unity AirSim仿真")
    print("  - 无人机: 1台 (UAV1)")
    print("  - 训练步数: 100000 步")
    print("  - 可视化: 启用")
    print()
    print("训练控制: 按 Ctrl+C 可以随时停止训练")
    print("=" * 60)
    print()
    
    try:
        # 运行训练脚本
        os.chdir(train_dir)
        result = subprocess.run([sys.executable, 'train_with_airsim.py'])
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("[SUCCESS] 训练成功完成！")
            print("=" * 60)
            print("\n模型已保存到: models/")
        else:
            print("\n" + "=" * 60)
            print("[ERROR] 训练失败或被中断")
            print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n\n[中断] 训练被用户中断")
        print("模型已保存到检查点")
    
    except Exception as e:
        print(f"\n[错误] {e}")
    
    print("\n按Enter键退出...")
    input()

if __name__ == "__main__":
    main()

