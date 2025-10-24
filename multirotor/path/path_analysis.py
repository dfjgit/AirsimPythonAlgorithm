#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径分析测试脚本
不连接AirSim，仅分析路径数据
"""

import json
import math
import logging
from typing import Dict, List, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PathAnalysis")

class PathAnalyzer:
    """路径分析器"""
    
    def __init__(self):
        self.path1_data = []
        self.path2_data = []
    
    def load_paths(self, path1_file: str, path2_file: str) -> bool:
        """加载两个路径文件"""
        try:
            # 加载路径1
            with open(path1_file, 'r', encoding='utf-8') as f:
                data1 = json.load(f)
                self.path1_data = data1.get("1", [])
            
            # 加载路径2
            with open(path2_file, 'r', encoding='utf-8') as f:
                data2 = json.load(f)
                self.path2_data = data2.get("1", [])
            
            logger.info(f"成功加载路径1: {len(self.path1_data)} 个点")
            logger.info(f"成功加载路径2: {len(self.path2_data)} 个点")
            return True
            
        except Exception as e:
            logger.error(f"加载路径文件失败: {str(e)}")
            return False
    
    def calculate_path_statistics(self, path_data: List[Dict[str, float]], path_name: str) -> Dict[str, float]:
        """计算路径统计信息"""
        if not path_data:
            return {}
        
        # 计算路径长度
        total_distance = 0.0
        for i in range(1, len(path_data)):
            p1 = path_data[i-1]
            p2 = path_data[i]
            distance = math.sqrt(
                (p2['x'] - p1['x'])**2 + 
                (p2['y'] - p1['y'])**2 + 
                (p2['z'] - p1['z'])**2
            )
            total_distance += distance
        
        # 计算高度变化
        heights = [point['z'] for point in path_data]
        min_height = min(heights)
        max_height = max(heights)
        height_range = max_height - min_height
        
        # 计算时间跨度
        times = [point.get('time', 0) for point in path_data]
        duration = max(times) - min(times) if times else 0
        
        # 计算起点和终点
        start_point = path_data[0]
        end_point = path_data[-1]
        straight_distance = math.sqrt(
            (end_point['x'] - start_point['x'])**2 + 
            (end_point['y'] - start_point['y'])**2 + 
            (end_point['z'] - start_point['z'])**2
        )
        
        stats = {
            'path_name': path_name,
            'point_count': len(path_data),
            'total_distance': total_distance,
            'straight_distance': straight_distance,
            'efficiency': straight_distance / total_distance if total_distance > 0 else 0,
            'min_height': min_height,
            'max_height': max_height,
            'height_range': height_range,
            'duration': duration,
            'avg_speed': total_distance / duration if duration > 0 else 0,
            'start_point': start_point,
            'end_point': end_point
        }
        
        return stats
    
    def analyze_path_differences(self) -> Dict[str, Any]:
        """分析路径差异"""
        if not self.path1_data or not self.path2_data:
            logger.error("路径数据不完整，无法分析")
            return {}
        
        # 计算统计信息
        stats1 = self.calculate_path_statistics(self.path1_data, "路径1")
        stats2 = self.calculate_path_statistics(self.path2_data, "路径2")
        
        # 计算差异
        analysis = {
            'path1_stats': stats1,
            'path2_stats': stats2,
            'differences': {
                'distance_diff': stats2['total_distance'] - stats1['total_distance'],
                'efficiency_diff': stats2['efficiency'] - stats1['efficiency'],
                'height_range_diff': stats2['height_range'] - stats1['height_range'],
                'duration_diff': stats2['duration'] - stats1['duration'],
                'speed_diff': stats2['avg_speed'] - stats1['avg_speed']
            },
            'recommendations': []
        }
        
        # 生成建议
        if stats1['efficiency'] > stats2['efficiency']:
            analysis['recommendations'].append("路径1比路径2更直接（效率更高）")
        else:
            analysis['recommendations'].append("路径2比路径1更直接（效率更高）")
        
        if stats1['total_distance'] < stats2['total_distance']:
            analysis['recommendations'].append("路径1总距离更短")
        else:
            analysis['recommendations'].append("路径2总距离更短")
        
        if stats1['height_range'] < stats2['height_range']:
            analysis['recommendations'].append("路径1高度变化更小，更稳定")
        else:
            analysis['recommendations'].append("路径2高度变化更小，更稳定")
        
        return analysis
    
    def print_analysis(self):
        """打印分析结果"""
        analysis = self.analyze_path_differences()
        if not analysis:
            return
        
        stats1 = analysis['path1_stats']
        stats2 = analysis['path2_stats']
        differences = analysis['differences']
        recommendations = analysis['recommendations']
        
        print("\n" + "="*60)
        print("                   路径分析报告")
        print("="*60)
        
        print(f"\n【路径1统计】")
        print(f"  路径点数: {stats1['point_count']}")
        print(f"  总距离: {stats1['total_distance']:.2f} m")
        print(f"  直线距离: {stats1['straight_distance']:.2f} m")
        print(f"  路径效率: {stats1['efficiency']:.3f}")
        print(f"  高度范围: {stats1['height_range']:.2f} m ({stats1['min_height']:.2f} ~ {stats1['max_height']:.2f})")
        print(f"  飞行时间: {stats1['duration']:.2f} s")
        print(f"  平均速度: {stats1['avg_speed']:.2f} m/s")
        print(f"  起点: ({stats1['start_point']['x']:.2f}, {stats1['start_point']['y']:.2f}, {stats1['start_point']['z']:.2f})")
        print(f"  终点: ({stats1['end_point']['x']:.2f}, {stats1['end_point']['y']:.2f}, {stats1['end_point']['z']:.2f})")
        
        print(f"\n【路径2统计】")
        print(f"  路径点数: {stats2['point_count']}")
        print(f"  总距离: {stats2['total_distance']:.2f} m")
        print(f"  直线距离: {stats2['straight_distance']:.2f} m")
        print(f"  路径效率: {stats2['efficiency']:.3f}")
        print(f"  高度范围: {stats2['height_range']:.2f} m ({stats2['min_height']:.2f} ~ {stats2['max_height']:.2f})")
        print(f"  飞行时间: {stats2['duration']:.2f} s")
        print(f"  平均速度: {stats2['avg_speed']:.2f} m/s")
        print(f"  起点: ({stats2['start_point']['x']:.2f}, {stats2['start_point']['y']:.2f}, {stats2['start_point']['z']:.2f})")
        print(f"  终点: ({stats2['end_point']['x']:.2f}, {stats2['end_point']['y']:.2f}, {stats2['end_point']['z']:.2f})")
        
        print(f"\n【差异分析】")
        print(f"  距离差异: {differences['distance_diff']:+.2f} m")
        print(f"  效率差异: {differences['efficiency_diff']:+.3f}")
        print(f"  高度范围差异: {differences['height_range_diff']:+.2f} m")
        print(f"  时间差异: {differences['duration_diff']:+.2f} s")
        print(f"  速度差异: {differences['speed_diff']:+.2f} m/s")
        
        print(f"\n【建议】")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*60)

def main():
    """主函数"""
    print("开始路径分析...")
    
    # 文件路径
    path1_file = "path1.json"
    path2_file = "path2.json"
    
    # 创建分析器
    analyzer = PathAnalyzer()
    
    # 加载路径
    if not analyzer.load_paths(path1_file, path2_file):
        print("加载路径文件失败")
        return
    
    # 进行分析
    analyzer.print_analysis()
    
    print("\n路径分析完成！")

if __name__ == "__main__":
    main()
