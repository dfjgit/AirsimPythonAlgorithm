@echo off
echo ========================================
echo AirSim 随机位置飞行测试
echo ========================================
echo.
echo 此测试将：
echo 1. 连接到AirSim并起飞
echo 2. 生成5个随机目标点（10米范围内）
echo 3. 依次飞到每个点并记录结果
echo 4. 显示测试总结
echo.
echo 请确保Unity/AirSim已启动！
echo.
pause

cd /d "%~dp0"
python test_random_flight.py

echo.
echo ========================================
echo 测试完成！
echo 详细日志已保存到: test_random_flight.log
echo ========================================
pause

