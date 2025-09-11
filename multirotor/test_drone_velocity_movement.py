from AirsimServer.drone_controller import DroneController
import time

def test_drone_velocity_movement():
    # 初始化控制器并连接模拟器
    controller = DroneController()
    if not controller.connect():
        print("连接模拟器失败，退出测试")
        return

    # 测试的无人机列表
    drones = ["UAV1", "UAV2"]
    # 每台无人机的速度指令 (vx, vy, vz) 和持续时间(秒)
    velocity_commands = {
        "UAV1": (2, 2, -1, 2),   # x正方向2m/s, y正方向2m/s, z负方向1m/s, 持续5秒
        "UAV2": (-2, -2, -1, 2)  # x负方向2m/s, y负方向2m/s, z负方向1m/s, 持续5秒
    }

    try:
        controller.reset()
        # 初始化所有无人机（启用控制、解锁、起飞）
        for drone in drones:
            print(f"\n初始化{drone}...")
            controller.enable_api_control(True, drone)
            controller.arm_disarm(True, drone)
            controller.takeoff(drone)
            time.sleep(5)  # 等待起飞至一定高度

        # 记录初始位置
        initial_positions = {drone: controller.get_vehicle_state(drone)["position"] for drone in drones}
        for drone in drones:
            print(f"{drone}初始位置: {initial_positions[drone]}")

        # 按速度移动无人机
        for drone in drones:
            vx, vy, vz, duration = velocity_commands[drone]
            print(f"\n{drone}以速度({vx}, {vy}, {vz})移动，持续{duration}秒...")
            controller.move_by_velocity(vx, vy, vz, duration, drone)
            time.sleep(duration)  # 等待速度移动完成

        for drone in drones:
            vx, vy, vz, duration = velocity_commands[drone]
            print(f"\n{drone}以速度({vx}, {vy}, {vz})移动，持续{duration}秒...")
            controller.move_by_velocity(vx*-1, vy*-1, vz, duration, drone)
            time.sleep(duration)  # 等待速度移动完成

        for drone in drones:
            vx, vy, vz, duration = velocity_commands[drone]
            print(f"\n{drone}以速度({vx}, {vy}, {vz})移动，持续{duration}秒...")
            controller.move_by_velocity(vx, vy, vz, duration, drone)
            time.sleep(duration)  # 等待速度移动完成

        # 记录移动后位置
        final_positions = {drone: controller.get_vehicle_state(drone)["position"] for drone in drones}
        for drone in drones:
            print(f"{drone}移动后位置: {final_positions[drone]}")

        # 所有无人机降落
        for drone in drones:
            print(f"\n{drone}开始降落...")
            controller.land(drone)
            time.sleep(5)
            controller.arm_disarm(False, drone)
            controller.enable_api_control(False, drone)

    finally:
        print("\n测试结束")

if __name__ == "__main__":
    test_drone_velocity_movement()