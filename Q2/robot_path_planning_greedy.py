"""
贪心算法 - 机器人路径规划
基于局部最优选择的路径规划算法

算法思路：
1. 在每一步选择时，评估向上和向右两个方向的收益
2. 收益函数考虑：到赛道中线的距离减少 + 向前位移奖励
3. 选择收益最大的方向移动
4. 重复直到到达赛道中线

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from track_detection_edge_based import EdgeBasedTrackDetector
import math

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class GreedyPathPlanner:
    def __init__(self, image_path):
        """
        初始化贪心路径规划器
        
        Args:
            image_path: 赛道图像路径
        """
        self.image_path = image_path
        self.step_size = 10  # 每步移动10像素
        self.start_pos = (100, 300)  # 起始位置
        
        # 加载赛道信息
        self.load_track_info()
        
        # 权重参数
        self.distance_weight = 1.0    # 距离中线权重
        self.forward_weight = 0.5     # 向前位移权重
        
    def load_track_info(self):
        """
        加载赛道边界信息
        """
        detector = EdgeBasedTrackDetector()
        results = detector.detect_track(self.image_path)
        
        self.original_image = results['original_image']
        self.track_mask = results['final_mask']
        self.left_boundary = np.array(results['left_boundary'])
        self.right_boundary = np.array(results['right_boundary'])
        
        # 计算赛道中线
        self.calculate_centerline()
        
    def calculate_centerline(self):
        """
        计算赛道中线
        """
        self.centerline = []
        
        if len(self.left_boundary) > 0 and len(self.right_boundary) > 0:
            # 获取y坐标范围
            min_y = max(self.left_boundary[:, 1].min(), self.right_boundary[:, 1].min())
            max_y = min(self.left_boundary[:, 1].max(), self.right_boundary[:, 1].max())
            
            for y in range(min_y, max_y + 1):
                # 找到该y坐标对应的左右边界点
                left_points = self.left_boundary[self.left_boundary[:, 1] == y]
                right_points = self.right_boundary[self.right_boundary[:, 1] == y]
                
                if len(left_points) > 0 and len(right_points) > 0:
                    left_x = left_points[0, 0]
                    right_x = right_points[0, 0]
                    center_x = (left_x + right_x) // 2
                    self.centerline.append([center_x, y])
        
        self.centerline = np.array(self.centerline)
    
    def get_distance_to_centerline(self, pos):
        """
        计算位置到赛道中线的最短距离
        
        Args:
            pos: 当前位置 (x, y)
            
        Returns:
            distance: 到中线的最短距离
        """
        if len(self.centerline) == 0:
            return float('inf')
        
        x, y = pos
        distances = np.sqrt((self.centerline[:, 0] - x)**2 + (self.centerline[:, 1] - y)**2)
        return np.min(distances)
    
    def is_on_track(self, pos):
        """
        判断位置是否在赛道上
        
        Args:
            pos: 位置 (x, y)
            
        Returns:
            bool: 是否在赛道上
        """
        x, y = pos
        if 0 <= x < self.track_mask.shape[1] and 0 <= y < self.track_mask.shape[0]:
            return self.track_mask[y, x] == 255
        return False
    
    def is_on_centerline(self, pos, tolerance=15):
        """
        判断是否到达赛道中线
        
        Args:
            pos: 当前位置
            tolerance: 容差范围
            
        Returns:
            bool: 是否到达中线
        """
        return self.get_distance_to_centerline(pos) <= tolerance
    
    def calculate_reward(self, current_pos, next_pos):
        """
        计算移动的奖励值
        
        Args:
            current_pos: 当前位置
            next_pos: 下一个位置
            
        Returns:
            reward: 奖励值
        """
        # 距离中线的改善
        current_dist = self.get_distance_to_centerline(current_pos)
        next_dist = self.get_distance_to_centerline(next_pos)
        distance_improvement = current_dist - next_dist
        
        # 向前位移（假设向右为前进方向）
        forward_movement = next_pos[0] - current_pos[0]
        
        # 综合奖励
        reward = (self.distance_weight * distance_improvement + 
                 self.forward_weight * forward_movement)
        
        # 如果移动到赛道外，给予惩罚
        if not self.is_on_track(next_pos):
            reward -= 100
        
        return reward
    
    def get_possible_moves(self, pos):
        """
        获取可能的移动方向
        
        Args:
            pos: 当前位置
            
        Returns:
            moves: 可能的移动列表
        """
        x, y = pos
        moves = []
        
        # 向上移动
        up_pos = (x, y - self.step_size)
        if y - self.step_size >= 0:
            moves.append(('up', up_pos))
        
        # 向右移动
        right_pos = (x + self.step_size, y)
        if x + self.step_size < self.original_image.shape[1]:
            moves.append(('right', right_pos))
        
        return moves
    
    def plan_path(self):
        """
        使用贪心算法规划路径
        
        Returns:
            path: 路径点列表
            directions: 移动方向列表
        """
        
        path = [self.start_pos]
        directions = []
        current_pos = self.start_pos
        
        max_steps = 1000  # 防止无限循环
        step_count = 0
        
        while not self.is_on_centerline(current_pos) and step_count < max_steps:
            # 获取可能的移动
            possible_moves = self.get_possible_moves(current_pos)
            
            if not possible_moves:
                print("无可用移动，路径规划结束")
                break
            
            # 计算每个移动的奖励
            best_move = None
            best_reward = float('-inf')
            
            for direction, next_pos in possible_moves:
                reward = self.calculate_reward(current_pos, next_pos)
                
                if reward > best_reward:
                    best_reward = reward
                    best_move = (direction, next_pos)
            
            # 执行最佳移动
            if best_move:
                direction, next_pos = best_move
                current_pos = next_pos
                path.append(current_pos)
                directions.append(direction)
                
                step_count += 1
                
                if step_count % 10 == 0:
                    dist = self.get_distance_to_centerline(current_pos)
                    print(f"步骤 {step_count}: 位置 {current_pos}, 距离中线 {dist:.2f}")
            else:
                break
        
        print(f"路径规划完成，共 {len(path)} 步")
        print(f"最终位置: {current_pos}")
        print(f"最终距离中线: {self.get_distance_to_centerline(current_pos):.2f}")
        
        return path, directions
    
    def visualize_path(self, path, output_name="greedy_path_result.png"):
        """
        可视化路径规划结果
        
        Args:
            path: 路径点列表
            output_name: 输出文件名
        """
        # 创建可视化图像
        result_image = self.original_image.copy()
        
        # 绘制赛道中线
        if len(self.centerline) > 0:
            for i in range(len(self.centerline) - 1):
                pt1 = tuple(self.centerline[i])
                pt2 = tuple(self.centerline[i + 1])
                cv2.line(result_image, pt1, pt2, (0, 255, 255), 2)  # 黄色中线
        
        # 绘制路径
        for i in range(len(path) - 1):
            pt1 = tuple(map(int, path[i]))
            pt2 = tuple(map(int, path[i + 1]))
            cv2.line(result_image, pt1, pt2, (0, 0, 255), 3)  # 红色路径
        
        # 绘制路径点
        for i, pos in enumerate(path):
            x, y = map(int, pos)
            if i == 0:
                cv2.circle(result_image, (x, y), 8, (255, 0, 0), -1)  # 蓝色起点
                cv2.putText(result_image, 'START', (x-20, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            elif i == len(path) - 1:
                cv2.circle(result_image, (x, y), 8, (0, 255, 0), -1)  # 绿色终点
                cv2.putText(result_image, 'END', (x-15, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)  # 红色路径点
        
        
        # 保存结果
        cv2.imwrite(output_name, result_image)
        
        return result_image
    
    def analyze_path(self, path, directions):
        """
        分析路径性能
        
        Args:
            path: 路径点列表
            directions: 移动方向列表
        """
        
        # 基本统计
        total_steps = len(path) - 1
        forward_displacement = path[-1][0] - path[0][0]
        final_distance = self.get_distance_to_centerline(path[-1])
        
        # 方向统计
        up_count = directions.count('up')
        right_count = directions.count('right')
        
        print(f"总步数: {total_steps}")
        print(f"向前位移: {forward_displacement} 像素")
        print(f"最终距离中线: {final_distance:.2f} 像素")
        print(f"向上移动: {up_count} 次")
        print(f"向右移动: {right_count} 次")
        
        
        return {
            'total_steps': total_steps,
            'forward_displacement': forward_displacement,
            'final_distance': final_distance,
            'up_count': up_count,
            'right_count': right_count,
        }

def main():
    """
    主函数
    """
    
    # 创建路径规划器
    planner = GreedyPathPlanner("saidao_2.png")
    
    # 规划路径
    path, directions = planner.plan_path()
    
    # 可视化结果
    planner.visualize_path(path, "greedy_path_result.png")
    
    # 分析路径性能
    analysis = planner.analyze_path(path, directions)
    

if __name__ == "__main__":
    main()