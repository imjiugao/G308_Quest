"""
基于边缘检测的赛道区域分割程序

设计思路：
本程序采用与传统颜色分割不同的算法思路，主要基于边缘检测和几何形状分析：

1. 图像预处理：使用双边滤波保持边缘清晰的同时去除噪声
2. 多尺度边缘检测：结合Canny边缘检测和Sobel算子，增强边缘信息
3. 霍夫直线检测：检测赛道的直线边界特征
4. 区域生长算法：从种子点开始，基于颜色相似性扩展赛道区域
5. 凸包分析：利用赛道的凸多边形特性进行形状约束
6. 自适应阈值：根据图像局部特性动态调整分割参数

核心创新点：
- 结合边缘信息和区域信息的混合分割策略
- 使用几何约束提高分割精度
- 自适应参数调整机制
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class EdgeBasedTrackDetector:
    def __init__(self):
        """
        初始化基于边缘检测的赛道检测器
        """
        # Canny边缘检测参数
        self.canny_low = 50      # Canny低阈值
        self.canny_high = 150    # Canny高阈值
        
        # 霍夫直线检测参数
        self.hough_threshold = 100    # 霍夫变换阈值
        self.min_line_length = 50     # 最小直线长度
        self.max_line_gap = 10        # 最大直线间隙
        
        # 区域生长参数
        self.color_tolerance = 30     # 颜色容差
        self.region_min_size = 1000   # 最小区域大小
        
    def bilateral_filter_preprocessing(self, image):
        """
        使用双边滤波进行图像预处理
        双边滤波能够在去噪的同时保持边缘信息
        
        Args:
            image: 输入的BGR图像
            
        Returns:
            filtered_image: 滤波后的图像
        """
        # 双边滤波：d=9表示邻域直径，sigmaColor和sigmaSpace控制滤波强度
        filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        
        return filtered
    
    def multi_scale_edge_detection(self, image):
        """
        多尺度边缘检测
        结合Canny边缘检测和Sobel算子，获得更丰富的边缘信息
        
        Args:
            image: 输入图像
            
        Returns:
            combined_edges: 组合的边缘图像
            canny_edges: Canny边缘
            sobel_edges: Sobel边缘
        """
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Canny边缘检测
        canny_edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # Sobel边缘检测（X和Y方向）
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_edges = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)
        
        # 对Sobel结果进行阈值处理
        _, sobel_edges = cv2.threshold(sobel_edges, 50, 255, cv2.THRESH_BINARY)
        
        # 组合两种边缘检测结果
        combined_edges = cv2.bitwise_or(canny_edges, sobel_edges)
        
        return combined_edges, canny_edges, sobel_edges
    
    def hough_line_detection(self, edges):
        """
        使用霍夫变换检测直线
        赛道通常具有明显的直线边界特征
        
        Args:
            edges: 边缘图像
            
        Returns:
            lines: 检测到的直线列表
            line_image: 绘制直线的图像
        """
        # 创建用于绘制直线的图像
        line_image = np.zeros_like(edges)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(
            edges, 
            rho=1,                          # 距离分辨率
            theta=np.pi/180,                # 角度分辨率
            threshold=self.hough_threshold, # 阈值
            minLineLength=self.min_line_length,  # 最小直线长度
            maxLineGap=self.max_line_gap    # 最大直线间隙
        )
        
        # 绘制检测到的直线
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
        
        return lines, line_image
    
    def adaptive_color_clustering(self, image, n_clusters=5):
        """
        自适应颜色聚类
        使用K-means聚类分析图像中的主要颜色
        
        Args:
            image: 输入图像
            n_clusters: 聚类数量
            
        Returns:
            clustered_image: 聚类后的图像
            cluster_centers: 聚类中心
        """
        # 将图像重塑为像素向量
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        cluster_centers = kmeans.cluster_centers_
        
        # 将聚类结果重塑回图像形状
        clustered_data = cluster_centers[labels]
        clustered_image = clustered_data.reshape(image.shape)
        clustered_image = np.uint8(clustered_image)
        
        return clustered_image, cluster_centers
    
    def region_growing(self, image, seed_points, tolerance):
        """
        区域生长算法
        从种子点开始，基于颜色相似性扩展区域
        
        Args:
            image: 输入图像
            seed_points: 种子点列表
            tolerance: 颜色容差
            
        Returns:
            region_mask: 生长区域的掩码
        """
        height, width = image.shape[:2]
        region_mask = np.zeros((height, width), dtype=np.uint8)
        visited = np.zeros((height, width), dtype=bool)
        
        # 对每个种子点执行区域生长
        for seed_point in seed_points:
            if seed_point is None:
                continue
                
            seed_x, seed_y = seed_point
            if visited[seed_y, seed_x]:
                continue
                
            # 获取种子点的颜色值
            seed_color = image[seed_y, seed_x]
            
            # 使用栈进行区域生长
            stack = [(seed_x, seed_y)]
            region_pixels = []
            
            while stack:
                x, y = stack.pop()
                
                # 检查边界条件
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue
                if visited[y, x]:
                    continue
                
                # 计算颜色差异
                current_color = image[y, x]
                color_diff = np.linalg.norm(current_color.astype(float) - seed_color.astype(float))
                
                # 如果颜色相似，加入区域
                if color_diff <= tolerance:
                    visited[y, x] = True
                    region_pixels.append((x, y))
                    region_mask[y, x] = 255
                    
                    # 添加邻接像素到栈中
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((x + dx, y + dy))
            
            # 如果区域太小，则移除
            if len(region_pixels) < self.region_min_size:
                for px, py in region_pixels:
                    region_mask[py, px] = 0
        
        return region_mask
    
    def find_track_seed_points(self, image, clustered_image):
        """
        寻找赛道区域的种子点
        基于颜色聚类结果和图像中心区域分析
        
        Args:
            image: 原始图像
            clustered_image: 聚类后的图像
            
        Returns:
            seed_points: 种子点列表
        """
        height, width = image.shape[:2]
        seed_points = []
        
        # 在图像中心区域寻找黄色像素作为种子点
        center_y, center_x = height // 2, width // 2
        search_radius = min(height, width) // 4
        
        # 转换到HSV空间进行黄色检测
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义黄色范围（更宽松的范围）
        lower_yellow = np.array([15, 30, 30])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 在中心区域寻找黄色像素
        for dy in range(-search_radius, search_radius, 10):
            for dx in range(-search_radius, search_radius, 10):
                y, x = center_y + dy, center_x + dx
                if 0 <= y < height and 0 <= x < width:
                    if yellow_mask[y, x] > 0:
                        seed_points.append((x, y))
        
        # 如果没有找到足够的种子点，使用图像中心点
        if len(seed_points) < 3:
            seed_points = [(center_x, center_y)]
        
        return seed_points
    
    def convex_hull_constraint(self, mask):
        """
        使用凸包约束优化分割结果
        赛道通常具有凸多边形的形状特征
        
        Args:
            mask: 输入掩码
            
        Returns:
            constrained_mask: 凸包约束后的掩码
        """
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算凸包
        hull = cv2.convexHull(largest_contour)
        
        # 创建凸包掩码
        constrained_mask = np.zeros_like(mask)
        cv2.fillPoly(constrained_mask, [hull], 255)
        
        return constrained_mask
    
    def morphological_refinement(self, mask):
        """
        形态学细化处理
        
        Args:
            mask: 输入掩码
            
        Returns:
            refined_mask: 细化后的掩码
        """
        # 定义结构元素
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # 开运算去除小噪声
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # 闭运算填补空洞
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)
        
        # 膨胀操作平滑边界
        refined_mask = cv2.dilate(closed, kernel_small, iterations=1)
        
        return refined_mask
    
    def extract_boundary_lines(self, mask):
        """
        提取赛道边界线
        使用轮廓分析和直线拟合
        
        Args:
            mask: 赛道掩码
            
        Returns:
            left_boundary: 左边界点
            right_boundary: 右边界点
            boundary_lines: 边界直线参数
        """
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [], [], []
        
        # 获取最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        left_boundary = []
        right_boundary = []
        
        # 扫描每一行，找到左右边界
        for row in range(y, y + h):
            # 获取该行的掩码
            row_mask = mask[row, :]
            white_pixels = np.where(row_mask == 255)[0]
            
            if len(white_pixels) > 0:
                left_x = white_pixels[0]
                right_x = white_pixels[-1]
                
                left_boundary.append([left_x, row])
                right_boundary.append([right_x, row])
        
        # 使用直线拟合边界点
        boundary_lines = []
        if len(left_boundary) > 10:
            left_points = np.array(left_boundary)
            # 使用最小二乘法拟合直线
            [vx, vy, x, y] = cv2.fitLine(left_points, cv2.DIST_L2, 0, 0.01, 0.01)
            boundary_lines.append(('left', vx, vy, x, y))
        
        if len(right_boundary) > 10:
            right_points = np.array(right_boundary)
            [vx, vy, x, y] = cv2.fitLine(right_points, cv2.DIST_L2, 0, 0.01, 0.01)
            boundary_lines.append(('right', vx, vy, x, y))
        
        return left_boundary, right_boundary, boundary_lines
    
    def detect_track(self, image_path):
        """
        主要的赛道检测函数
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            results: 包含所有处理结果的字典
        """
        # 读取图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        print(f"开始处理图像: {image_path}")
        print(f"图像尺寸: {original_image.shape}")
        
        # 1. 双边滤波预处理
        print("步骤1: 双边滤波预处理...")
        filtered_image = self.bilateral_filter_preprocessing(original_image)
        
        # 2. 多尺度边缘检测
        print("步骤2: 多尺度边缘检测...")
        combined_edges, canny_edges, sobel_edges = self.multi_scale_edge_detection(filtered_image)
        
        # 3. 霍夫直线检测
        print("步骤3: 霍夫直线检测...")
        lines, line_image = self.hough_line_detection(combined_edges)
        
        # 4. 自适应颜色聚类
        print("步骤4: 自适应颜色聚类...")
        clustered_image, cluster_centers = self.adaptive_color_clustering(filtered_image)
        
        # 5. 寻找种子点
        print("步骤5: 寻找赛道种子点...")
        seed_points = self.find_track_seed_points(original_image, clustered_image)
        print(f"找到 {len(seed_points)} 个种子点")
        
        # 6. 区域生长
        print("步骤6: 区域生长...")
        region_mask = self.region_growing(filtered_image, seed_points, self.color_tolerance)
        
        # 7. 凸包约束
        print("步骤7: 凸包约束...")
        hull_mask = self.convex_hull_constraint(region_mask)
        
        # 8. 形态学细化
        print("步骤8: 形态学细化...")
        final_mask = self.morphological_refinement(hull_mask)
        
        # 9. 提取边界线
        print("步骤9: 提取边界线...")
        left_boundary, right_boundary, boundary_lines = self.extract_boundary_lines(final_mask)
        
        # 创建结果可视化
        result_image = self.create_visualization(
            original_image, final_mask, left_boundary, right_boundary, boundary_lines
        )
        
        # 组织返回结果
        results = {
            'original_image': original_image,
            'filtered_image': filtered_image,
            'combined_edges': combined_edges,
            'canny_edges': canny_edges,
            'sobel_edges': sobel_edges,
            'line_image': line_image,
            'clustered_image': clustered_image,
            'region_mask': region_mask,
            'hull_mask': hull_mask,
            'final_mask': final_mask,
            'result_image': result_image,
            'left_boundary': left_boundary,
            'right_boundary': right_boundary,
            'boundary_lines': boundary_lines,
            'seed_points': seed_points,
            'cluster_centers': cluster_centers
        }
        
        print("赛道检测完成！")
        return results
    
    def create_visualization(self, original_image, mask, left_boundary, right_boundary, boundary_lines):
        """
        创建可视化结果
        
        Args:
            original_image: 原始图像
            mask: 最终掩码
            left_boundary: 左边界点
            right_boundary: 右边界点
            boundary_lines: 边界直线参数
            
        Returns:
            result_image: 可视化结果
        """
        result_image = original_image.copy()
        
        # 创建彩色掩码叠加
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask == 255] = [0, 255, 0]  # 绿色表示赛道区域
        result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
        
        # 绘制边界点
        for point in left_boundary:
            cv2.circle(result_image, tuple(point), 2, (0, 0, 255), -1)  # 红色左边界
        
        for point in right_boundary:
            cv2.circle(result_image, tuple(point), 2, (255, 0, 0), -1)  # 蓝色右边界
        
        # 绘制拟合的边界直线
        height = original_image.shape[0]
        for line_info in boundary_lines:
            side, vx, vy, x, y = line_info
            
            # 计算直线的两个端点
            t1 = -y / vy if vy != 0 else 0
            t2 = (height - y) / vy if vy != 0 else 0
            
            pt1 = (int(x + vx * t1), int(y + vy * t1))
            pt2 = (int(x + vx * t2), int(y + vy * t2))
            
            color = (0, 255, 255) if side == 'left' else (255, 255, 0)  # 黄色直线
            cv2.line(result_image, pt1, pt2, color, 3)
        
        return result_image
    
    def save_results(self, results, output_prefix="edge_based_track"):
        """
        保存所有处理结果
        
        Args:
            results: 处理结果字典
            output_prefix: 输出文件前缀
        """
        # 保存主要结果
        cv2.imwrite(f"{output_prefix}_mask.png", results['final_mask'])
        cv2.imwrite(f"{output_prefix}_result.png", results['result_image'])
        cv2.imwrite(f"{output_prefix}_edges.png", results['combined_edges'])
        cv2.imwrite(f"{output_prefix}_clustered.png", results['clustered_image'])
        
        print(f"结果已保存:")
        print(f"- 最终掩码: {output_prefix}_mask.png")
        print(f"- 可视化结果: {output_prefix}_result.png")
        print(f"- 边缘检测: {output_prefix}_edges.png")
        print(f"- 颜色聚类: {output_prefix}_clustered.png")
        
        # 创建综合展示图
        self.create_comprehensive_visualization(results, output_prefix)
    
    def create_comprehensive_visualization(self, results, output_prefix):
        """
        创建综合可视化展示
        
        Args:
            results: 处理结果字典
            output_prefix: 输出文件前缀
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # 原始图像
        axes[0, 0].imshow(cv2.cvtColor(results['original_image'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('原始图像', fontsize=12)
        axes[0, 0].axis('off')
        
        # 双边滤波结果
        axes[0, 1].imshow(cv2.cvtColor(results['filtered_image'], cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('双边滤波', fontsize=12)
        axes[0, 1].axis('off')
        
        # 颜色聚类结果
        axes[0, 2].imshow(cv2.cvtColor(results['clustered_image'], cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('颜色聚类', fontsize=12)
        axes[0, 2].axis('off')
        
        # Canny边缘
        axes[1, 0].imshow(results['canny_edges'], cmap='gray')
        axes[1, 0].set_title('Canny边缘检测', fontsize=12)
        axes[1, 0].axis('off')
        
        # Sobel边缘
        axes[1, 1].imshow(results['sobel_edges'], cmap='gray')
        axes[1, 1].set_title('Sobel边缘检测', fontsize=12)
        axes[1, 1].axis('off')
        
        # 组合边缘
        axes[1, 2].imshow(results['combined_edges'], cmap='gray')
        axes[1, 2].set_title('组合边缘检测', fontsize=12)
        axes[1, 2].axis('off')
        
        # 区域生长结果
        axes[2, 0].imshow(results['region_mask'], cmap='gray')
        axes[2, 0].set_title('区域生长', fontsize=12)
        axes[2, 0].axis('off')
        
        # 最终掩码
        axes[2, 1].imshow(results['final_mask'], cmap='gray')
        axes[2, 1].set_title('最终掩码', fontsize=12)
        axes[2, 1].axis('off')
        
        # 最终结果
        axes[2, 2].imshow(cv2.cvtColor(results['result_image'], cv2.COLOR_BGR2RGB))
        axes[2, 2].set_title('最终结果', fontsize=12)
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    主函数
    """
    # 创建基于边缘检测的赛道检测器
    detector = EdgeBasedTrackDetector()
    
    # 输入图像路径
    input_image = "saidao.jpeg"
    
    try:
        # 执行赛道检测
        print("=" * 50)
        print("基于边缘检测的赛道区域分割")
        print("=" * 50)
        
        results = detector.detect_track(input_image)
        
        # 保存结果
        detector.save_results(results, "edge_based_track")
        
        # 输出统计信息
        print(f"\n处理统计信息:")
        print(f"- 左边界点数量: {len(results['left_boundary'])}")
        print(f"- 右边界点数量: {len(results['right_boundary'])}")
        print(f"- 检测到的边界直线数量: {len(results['boundary_lines'])}")
        print(f"- 种子点数量: {len(results['seed_points'])}")
        print(f"- 颜色聚类中心数量: {len(results['cluster_centers'])}")
        
        # 计算赛道区域面积
        track_area = np.sum(results['final_mask'] == 255)
        total_area = results['final_mask'].shape[0] * results['final_mask'].shape[1]
        coverage_ratio = track_area / total_area * 100
        
        print(f"- 赛道区域像素数量: {track_area}")
        print(f"- 赛道覆盖率: {coverage_ratio:.2f}%")
        
        print("\n处理完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()