"""
AirSim 碰撞处理快速使用示例

演示如何使用新的碰撞检测和防穿地重置功能
"""

from multirotor.AirsimServer.drone_controller import DroneController
import time

def example_basic_collision_check():
    """示例1：基础碰撞检测"""
    print("\n=== 示例1：基础碰撞检测 ===")
    
    controller = DroneController()
    controller.connect()
    
    # 检查无人机碰撞状态
    collision = controller.check_collision("UAV1")
    
    if collision['has_collided']:
        print(f"⚠️  检测到碰撞!")
        print(f"   碰撞对象: {collision['object_name']}")
        print(f"   穿透深度: {collision['penetration_depth']:.3f}米")
        print(f"   碰撞点: {collision['impact_point']}")
        print(f"   碰撞法向量: {collision['normal']}")
    else:
        print("✅ 无碰撞")


def example_safe_reset():
    """示例2：安全重置(防穿地)"""
    print("\n=== 示例2：安全重置 ===")
    
    controller = DroneController()
    controller.connect()
    
    # 执行安全重置(自动暂停仿真)
    print("执行安全重置...")
    if controller.reset():
        print("✅ 重置成功，无穿地风险")
    else:
        print("❌ 重置失败")


def example_collision_recovery():
    """示例3：碰撞后恢复"""
    print("\n=== 示例3：碰撞后恢复 ===")
    
    controller = DroneController()
    controller.connect()
    
    # 检查并恢复
    collision = controller.check_collision("UAV1")
    
    if collision['has_collided']:
        print(f"检测到碰撞，尝试恢复...")
        if controller.recover_from_collision("UAV1"):
            print("✅ 已尝试悬停稳定")
            
            # 再次检查
            time.sleep(1)
            collision = controller.check_collision("UAV1")
            if collision['has_collided']:
                print("⚠️  仍处于碰撞状态，需要强制重置")
            else:
                print("✅ 恢复成功")


def example_force_reset_position():
    """示例4：强制位置重置(用于穿地恢复)"""
    print("\n=== 示例4：强制位置重置 ===")
    
    controller = DroneController()
    controller.connect()
    
    # 检查是否穿地严重
    collision = controller.check_collision("UAV1")
    
    if collision['has_collided'] and collision['penetration_depth'] > 1.0:
        print(f"⚠️  严重穿地(深度: {collision['penetration_depth']:.2f}米)")
        print("执行强制位置重置...")
        
        # 忽略碰撞，强制传送到安全位置
        if controller.reset_vehicle_to_pose(
            vehicle_name="UAV1",
            position=(0, 0, -3),  # 原点上方3米
            ignore_collision=True  # 关键：忽略碰撞
        ):
            print("✅ 已强制重置到安全位置")
            
            # 验证
            time.sleep(0.5)
            state = controller.get_vehicle_state("UAV1")
            print(f"   当前位置: {state['position']}")


def example_training_integration():
    """示例5：训练循环集成"""
    print("\n=== 示例5：训练循环集成 ===")
    
    controller = DroneController()
    controller.connect()
    
    # 启用API控制
    controller.enable_api_control(True, "UAV1")
    controller.arm_disarm(True, "UAV1")
    controller.takeoff("UAV1")
    
    # 模拟训练循环
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        
        # 检查碰撞
        collision = controller.check_collision("UAV1")
        
        if collision['has_collided']:
            print(f"⚠️  Episode中断：碰撞 {collision['object_name']}")
            
            # 判断严重程度
            if collision['penetration_depth'] > 0.5:
                print("   严重碰撞，执行完全重置")
                controller.reset()  # 安全重置
                controller.enable_api_control(True, "UAV1")
                controller.arm_disarm(True, "UAV1")
                controller.takeoff("UAV1")
            else:
                print("   轻微碰撞，尝试恢复")
                controller.recover_from_collision("UAV1")
        else:
            print("✅ 无碰撞，继续训练")
        
        # 模拟一些动作
        time.sleep(0.5)


def example_complete_workflow():
    """示例6：完整的安全重置工作流"""
    print("\n=== 示例6：完整安全重置工作流 ===")
    
    controller = DroneController()
    controller.connect()
    
    print("步骤1: 检查当前状态")
    collision = controller.check_collision("UAV1")
    state = controller.get_vehicle_state("UAV1")
    
    print(f"  位置: {state['position']}")
    print(f"  碰撞: {'是' if collision['has_collided'] else '否'}")
    
    print("\n步骤2: 根据状态选择重置策略")
    
    if collision['has_collided']:
        if collision['penetration_depth'] > 1.0:
            print("  策略: 严重穿地 → 强制位置重置")
            controller.reset_vehicle_to_pose("UAV1", (0, 0, -3), ignore_collision=True)
        else:
            print("  策略: 轻微碰撞 → 先悬停恢复")
            controller.recover_from_collision("UAV1")
            time.sleep(0.5)
            
            # 检查恢复效果
            collision = controller.check_collision("UAV1")
            if collision['has_collided']:
                print("  恢复失败 → 执行完全重置")
                controller.reset()
            else:
                print("  ✅ 恢复成功")
    else:
        print("  策略: 无碰撞 → 标准安全重置")
        controller.reset()
    
    print("\n步骤3: 重新初始化")
    controller.enable_api_control(True, "UAV1")
    controller.arm_disarm(True, "UAV1")
    controller.takeoff("UAV1")
    
    print("✅ 完整重置流程执行完毕")


if __name__ == "__main__":
    print("=" * 60)
    print("AirSim 碰撞处理示例")
    print("=" * 60)
    
    # 运行所有示例
    try:
        example_basic_collision_check()
        example_safe_reset()
        example_collision_recovery()
        example_force_reset_position()
        example_training_integration()
        example_complete_workflow()
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("请确保AirSim仿真器正在运行")
    
    print("\n" + "=" * 60)
    print("所有示例执行完毕")
    print("=" * 60)
