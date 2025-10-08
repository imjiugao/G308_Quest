"""
优化的赛道边界检测程序

设计思路：
本程序采用基于轮廓分析和几何特征的优化算法：

1. 高效预处理：使用高斯滤波和自适应直方图均衡化
2. 智能颜色分割：结合LAB和HSV颜色空间的双重分割
3. 连通组件分析：快速识别和筛选候选区域
4. 几何形状验证：基于面积、长宽比等特征筛选赛道
5. 边界优化：使用多项式拟合获得平滑边界线
6. 实时性优化：减少不必要的计算，提高处理速度

核心优势：
- 计算效率高，适合实时处理
- 鲁棒性强，对光照变化不敏感
- 精度高，边界检测准确
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedTrackDetector:
    def __init__(self):
        """
        初始化优化的赛道检测器
        """
        # LAB颜色空间中黄色的范围（更精确）
        self.lab_lower = np.array([20, 15, 40])   # L, A, B下界
        self.lab_upper = np.array([100, 35, 80])  # L, A, B上界
        
        # HSV颜色空间中黄色的范围（备用）
        self.hsv_lower = np.array([15, 50, 50])
        self.hsv_upper = np.array([35, 255, 255])
        
        # 形态学操作参数
        self.kernel_size = 5
        self.min_area = 5000      # 最小区域面积
        self.max_area = 500000    # 最大区域面积
        
    def preprocess_image(self, image):
        """
        高效的图像预处理
        
        Args:
            image: 输入的BGR图像
            
        Returns:
            processed: 预处理后的图像
        """
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 自适应直方图均衡化（仅对亮度通道）
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return processed
    
    def dual_color_segmentation(self, image):
        """
        双重颜色空间分割
        结合LAB和HSV颜色空间提高分割精度
        
        Args:
            image: 输入图像
            
        Returns:
            combined_mask: 组合掩码
            lab_mask: LAB空间掩码
            hsv_mask: HSV空间掩码
        """
        # LAB颜色空间分割
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_mask = cv2.inRange(lab, self.lab_lower, self.lab_upper)
        
        # HSV颜色空间分割
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # 组合两个掩码（取交集以提高精度）
        combined_mask = cv2.bitwise_and(lab_mask, hsv_mask)
        
        # 如果交集太小，则使用并集
        if np.sum(combined_mask) < self.min_area:
            combined_mask = cv2.bitwise_or(lab_mask, hsv_mask)
        
        return combined_mask, lab_mask, hsv_mask
    
    def connected_component_analysis(self, mask):
        """
        连通组件分析
        快速识别和筛选候选区域
        
        Args:
            mask: 输入二值掩码
            
        Returns:
            filtered_mask: 筛选后的掩码
            stats: 连通组件统计信息
        """
        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 创建筛选后的掩码
        filtered_mask = np.zeros_like(mask)
        
        # 筛选符合条件的组件
        for i in range(1, num_labels):  # 跳过背景（标签0）
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 面积筛选
            if self.min_area <= area <= self.max_area:
                # 长宽比筛选（赛道通常比较长）
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 1.2:  # 长宽比大于1.2
                    # 将该组件添加到筛选后的掩码中
                    filtered_mask[labels == i] = 255
        
        return filtered_mask, stats
    
    def morphological_optimization(self, mask):
        """
        形态学优化
        
        Args:
            mask: 输入掩码
            
        Returns:
            optimized_mask: 优化后的掩码
        """
        # 定义结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        
        # 开运算去除噪声
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 闭运算填补空洞
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 轻微膨胀平滑边界
        optimized_mask = cv2.dilate(closed, kernel, iterations=1)
        
        return optimized_mask
    
    def extract_main_track_region(self, mask):
        """
        提取主要的赛道区域
        
        Args:
            mask: 输入掩码
            
        Returns:
            main_track_mask: 主赛道掩码
            main_contour: 主赛道轮廓
        """
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask, None
        
        # 找到面积最大的轮廓作为主赛道
        main_contour = max(contours, key=cv2.contourArea)
        
        # 创建主赛道掩码
        main_track_mask = np.zeros_like(mask)
        cv2.fillPoly(main_track_mask, [main_contour], 255)
        
        return main_track_mask, main_contour
    
    def polynomial_boundary_fitting(self, mask):
        """
        多项式边界拟合
        获得平滑的边界线
        
        Args:
            mask: 赛道掩码
            
        Returns:
            left_boundary: 左边界点
            right_boundary: 右边界点
            left_poly: 左边界多项式系数
            right_poly: 右边界多项式系数
        """
        height, width = mask.shape
        left_boundary = []
        right_boundary = []
        
        # 从上到下扫描，找到边界点
        for y in range(0, height, 2):  # 每隔2行采样，提高效率
            row = mask[y, :]
            white_pixels = np.where(row == 255)[0]
            
            if len(white_pixels) > 0:
                left_x = white_pixels[0]
                right_x = white_pixels[-1]
                
                left_boundary.append([left_x, y])
                right_boundary.append([right_x, y])
        
        # 多项式拟合（2次多项式）
        left_poly = None
        right_poly = None
        
        if len(left_boundary) > 10:
            left_points = np.array(left_boundary)
            left_poly = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
        
        if len(right_boundary) > 10:
            right_points = np.array(right_boundary)
            right_poly = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
        
        return left_boundary, right_boundary, left_poly, right_poly
    
    def create_smooth_boundaries(self, mask, left_poly, right_poly):
        """
        创建平滑的边界线
        
        Args:
            mask: 输入掩码
            left_poly: 左边界多项式系数
            right_poly: 右边界多项式系数
            
        Returns:
            smooth_left: 平滑左边界点
            smooth_right: 平滑右边界点
        """
        height = mask.shape[0]
        smooth_left = []
        smooth_right = []
        
        # 生成平滑的边界点
        for y in range(height):
            if left_poly is not None:
                x_left = int(np.polyval(left_poly, y))
                if 0 <= x_left < mask.shape[1]:
                    smooth_left.append([x_left, y])
            
            if right_poly is not None:
                x_right = int(np.polyval(right_poly, y))
                if 0 <= x_right < mask.shape[1]:
                    smooth_right.append([x_right, y])
        
        return smooth_left, smooth_right
    
    def detect_track(self, image_path):
        """
        主要的赛道检测函数
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            results: 检测结果字典
        """
        start_time = time.time()
        
        # 读取图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        print(f"开始处理图像: {image_path}")
        print(f"图像尺寸: {original_image.shape}")
        
        # 1. 图像预处理
        print("步骤1: 图像预处理...")
        processed_image = self.preprocess_image(original_image)
        
        # 2. 双重颜色分割
        print("步骤2: 双重颜色分割...")
        combined_mask, lab_mask, hsv_mask = self.dual_color_segmentation(processed_image)
        
        # 3. 连通组件分析
        print("步骤3: 连通组件分析...")
        filtered_mask, stats = self.connected_component_analysis(combined_mask)
        
        # 4. 形态学优化
        print("步骤4: 形态学优化...")
        optimized_mask = self.morphological_optimization(filtered_mask)
        
        # 5. 提取主赛道区域
        print("步骤5: 提取主赛道区域...")
        main_track_mask, main_contour = self.extract_main_track_region(optimized_mask)
        
        # 6. 边界拟合
        print("步骤6: 多项式边界拟合...")
        left_boundary, right_boundary, left_poly, right_poly = self.polynomial_boundary_fitting(main_track_mask)
        
        # 7. 创建平滑边界
        print("步骤7: 创建平滑边界...")
        smooth_left, smooth_right = self.create_smooth_boundaries(main_track_mask, left_poly, right_poly)
        
        # 8. 创建可视化结果
        print("步骤8: 创建可视化...")
        result_image = self.create_visualization(
            original_image, main_track_mask, smooth_left, smooth_right
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 组织返回结果
        results = {
            'original_image': original_image,
            'processed_image': processed_image,
            'combined_mask': combined_mask,
            'lab_mask': lab_mask,
            'hsv_mask': hsv_mask,
            'filtered_mask': filtered_mask,
            'optimized_mask': optimized_mask,
            'main_track_mask': main_track_mask,
            'result_image': result_image,
            'left_boundary': left_boundary,
            'right_boundary': right_boundary,
            'smooth_left': smooth_left,
            'smooth_right': smooth_right,
            'left_poly': left_poly,
            'right_poly': right_poly,
            'main_contour': main_contour,
            'processing_time': processing_time
        }
        
        print(f"赛道检测完成！处理时间: {processing_time:.2f}秒")
        return results
    
    def create_visualization(self, original_image, mask, left_boundary, right_boundary):
        """
        创建可视化结果
        
        Args:
            original_image: 原始图像
            mask: 赛道掩码
            left_boundary: 左边界点
            right_boundary: 右边界点
            
        Returns:
            result_image: 可视化结果
        """
        result_image = original_image.copy()
        
        # 创建半透明的赛道区域叠加
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask == 255] = [0, 255, 0]  # 绿色表示赛道区域
        result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
        
        # 绘制平滑的边界线
        if len(left_boundary) > 1:
            left_points = np.array(left_boundary, dtype=np.int32)
            cv2.polylines(result_image, [left_points], False, (0, 0, 255), 3)  # 红色左边界
        
        if len(right_boundary) > 1:
            right_points = np.array(right_boundary, dtype=np.int32)
            cv2.polylines(result_image, [right_points], False, (255, 0, 0), 3)  # 蓝色右边界
        
        return result_image
    
    def save_results(self, results, output_prefix="optimized_track"):
        """
        保存处理结果
        
        Args:
            results: 处理结果字典
            output_prefix: 输出文件前缀
        """
        # 保存主要结果
        cv2.imwrite(f"{output_prefix}_mask.png", results['main_track_mask'])
        cv2.imwrite(f"{output_prefix}_result.png", results['result_image'])
        
        print(f"结果已保存:")
        print(f"- 赛道掩码: {output_prefix}_mask.png")
        print(f"- 可视化结果: {output_prefix}_result.png")
        
        # 创建对比展示图
        self.create_comparison_visualization(results, output_prefix)
    
    def create_comparison_visualization(self, results, output_prefix):
        """
        创建对比可视化
        
        Args:
            results: 处理结果字典
            output_prefix: 输出文件前缀
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        axes[0, 0].imshow(cv2.cvtColor(results['original_image'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('原始图像', fontsize=12)
        axes[0, 0].axis('off')
        
        # 预处理图像
        axes[0, 1].imshow(cv2.cvtColor(results['processed_image'], cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('预处理图像', fontsize=12)
        axes[0, 1].axis('off')
        
        # LAB掩码
        axes[0, 2].imshow(results['lab_mask'], cmap='gray')
        axes[0, 2].set_title('LAB颜色分割', fontsize=12)
        axes[0, 2].axis('off')
        
        # HSV掩码
        axes[1, 0].imshow(results['hsv_mask'], cmap='gray')
        axes[1, 0].set_title('HSV颜色分割', fontsize=12)
        axes[1, 0].axis('off')
        
        # 最终掩码
        axes[1, 1].imshow(results['main_track_mask'], cmap='gray')
        axes[1, 1].set_title('最终赛道掩码', fontsize=12)
        axes[1, 1].axis('off')
        
        # 最终结果
        axes[1, 2].imshow(cv2.cvtColor(results['result_image'], cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('边界检测结果', fontsize=12)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    主函数
    """
    # 创建优化的赛道检测器
    detector = OptimizedTrackDetector()
    
    # 输入图像路径
    input_image = "saidao.jpeg"
    
    try:
        print("=" * 60)
        print("优化的赛道边界检测算法")
        print("=" * 60)
        
        # 执行赛道检测
        results = detector.detect_track(input_image)
        
        # 保存结果
        detector.save_results(results, "optimized_track")
        
        # 输出详细统计信息
        print(f"\n处理统计信息:")
        print(f"- 处理时间: {results['processing_time']:.2f}秒")
        print(f"- 左边界点数量: {len(results['left_boundary'])}")
        print(f"- 右边界点数量: {len(results['right_boundary'])}")
        print(f"- 平滑左边界点数量: {len(results['smooth_left'])}")
        print(f"- 平滑右边界点数量: {len(results['smooth_right'])}")
        
        # 计算赛道区域统计
        track_area = np.sum(results['main_track_mask'] == 255)
        total_area = results['main_track_mask'].shape[0] * results['main_track_mask'].shape[1]
        coverage_ratio = track_area / total_area * 100
        
        print(f"- 赛道区域像素数量: {track_area}")
        print(f"- 赛道覆盖率: {coverage_ratio:.2f}%")
        
        # 多项式系数信息
        if results['left_poly'] is not None:
            print(f"- 左边界多项式系数: {results['left_poly']}")
        if results['right_poly'] is not None:
            print(f"- 右边界多项式系数: {results['right_poly']}")
        
        print("\n算法特点:")
        print("- 使用LAB和HSV双重颜色空间分割")
        print("- 连通组件分析快速筛选候选区域")
        print("- 多项式拟合获得平滑边界线")
        print("- 优化的形态学操作提高精度")
        print("- 高效的处理流程适合实时应用")
        
        print("\n处理完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()