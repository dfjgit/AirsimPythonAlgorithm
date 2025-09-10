#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试DataStorageManager功能的脚本"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入数据存储管理器
from multirotor.AirsimServer.data_storage import DataStorageManager

print("=== AirSim DroneServer 数据存储功能测试 ===")

# 创建数据存储管理器实例
data_manager = DataStorageManager()
print("✓ 数据存储管理器已创建")

def test_store_data():
    """测试存储数据功能"""
    print("\n测试存储数据功能...")
    # 测试正常存储
    result = data_manager.store_data("test_id_1", {"name": "test_data", "value": 123})
    print(f"  存储测试数据1: {result['status']} - {result['message']}")
    
    # 测试重复ID存储
    result = data_manager.store_data("test_id_1", {"name": "updated_data", "value": 456})
    print(f"  更新测试数据1: {result['status']} - {result['message']}")
    
    # 测试存储多个不同ID的数据
    data_manager.store_data("test_id_2", [1, 2, 3, 4, 5])
    data_manager.store_data("test_id_3", "这是字符串数据")
    print("  已存储多个测试数据")
    
    return True

def test_retrieve_data():
    """测试检索数据功能"""
    print("\n测试检索数据功能...")
    # 测试检索存在的数据
    result = data_manager.retrieve_data("test_id_1")
    print(f"  检索存在的数据(test_id_1): {result['status']} - {result['message']}")
    # 不打印数据内容
    
    # 测试检索不存在的数据
    result = data_manager.retrieve_data("non_existent_id")
    print(f"  检索不存在的数据: {result['status']} - {result['message']}")
    
    # 测试检索多个数据
    result_id2 = data_manager.retrieve_data("test_id_2")
    result_id3 = data_manager.retrieve_data("test_id_3")
    print(f"  检索多个数据: 成功检索{2 if result_id2['status'] == 'success' and result_id3['status'] == 'success' else 0}条数据")
    
    return True

def test_list_data():
    """测试列出所有数据ID功能"""
    print("\n测试列出所有数据ID功能...")
    result = data_manager.list_data_ids()
    print(f"  列出所有数据ID: {result['status']} - {result['message']}")
    if result['status'] == 'success':
        print(f"    数据ID列表: {result['data_ids']}")
    
    return True

def test_delete_data():
    """测试删除数据功能"""
    print("\n测试删除数据功能...")
    # 测试删除存在的数据
    result = data_manager.delete_data("test_id_1")
    print(f"  删除存在的数据(test_id_1): {result['status']} - {result['message']}")
    
    # 验证数据是否已删除
    result_after_delete = data_manager.retrieve_data("test_id_1")
    print(f"  验证数据是否已删除: {'成功' if result_after_delete['status'] == 'error' else '失败'}")
    
    # 测试删除不存在的数据
    result = data_manager.delete_data("non_existent_id")
    print(f"  删除不存在的数据: {result['status']} - {result['message']}")
    
    return True

def test_clear_all_data():
    """测试清除所有数据功能"""
    print("\n测试清除所有数据功能...")
    # 存储一些新数据用于清除测试
    data_manager.store_data("test_id_4", "临时数据1")
    data_manager.store_data("test_id_5", "临时数据2")
    
    # 清除所有数据
    result = data_manager.clear_all_data()
    print(f"  清除所有数据: {result['status']} - {result['message']}")
    
    # 验证数据是否已清除
    result_after_clear = data_manager.list_data_ids()
    print(f"  验证数据是否已清除: {'成功' if result_after_clear['count'] == 0 else '失败'}")
    
    return True

def main():
    # 运行所有测试
    test_store_data()
    test_retrieve_data()
    test_list_data()
    test_delete_data()
    test_clear_all_data()
    
    print("\n=== 数据存储功能测试完成 ===")

if __name__ == "__main__":
    main()