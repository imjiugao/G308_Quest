"""
赛道区域分割程序

设计思路：
1. 使用HSV颜色空间进行黄色区域检测
2. 通过颜色阈值分割提取黄色赛道区域
3. 使用形态学操作去除噪声和填补空洞
4. 应用轮廓检测找到最大的连通区域作为赛道
5. 识别并绘制赛道边界线
6. 生成二值掩码图像，白色表示赛道区域，黑色表示其他区域
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class TrackSegmentation:
    def __init__(self):
        """
        初始化赛道分割类
        """
        # 定义黄色在HSV空间的颜色范围
        # HSV中黄色的色调范围大约在20-30之间
        self.lower_yellow = np.array([15, 50, 50])   # 黄色下界：色调15，饱和度50，亮度50
        self.upper_yellow = np.array([35, 255, 255]) # 黄色上界：色调35，饱和度255，亮度255
        
    def preprocess_image(self, image):
        """
        图像预处理
        
        Args:
            image: 输入的BGR图像
            
        Returns:
            processed_image: 预处理后的图像
        """
        # 高斯模糊去噪，减少图像噪声对分割的影响
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 转换到HSV颜色空间，便于颜色分割
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return hsv
    
    def color_segmentation(self, hsv_image):
        """
        基于颜色的分割
        
        Args:
            hsv_image: HSV格式的图像
            
        Returns:
            mask: 二值掩码，白色表示黄色区域
        """
        # 创建黄色区域的掩码
        # inRange函数会将在指定范围内的像素设为255（白色），其他设为0（黑色）
        mask = cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow)
        
        return mask
    
    def morphological_operations(self, mask):
        """
        形态学操作优化掩码
        
        Args:
            mask: 输入的二值掩码
            
        Returns:
            processed_mask: 经过形态学处理的掩码
        """
        # 定义形态学操作的结构元素（核）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 开运算：先腐蚀后膨胀，去除小的噪声点
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 闭运算：先膨胀后腐蚀，填补内部的小洞
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def find_largest_contour(self, mask):
        """
        找到最大的轮廓区域
        
        Args:
            mask: 二值掩码图像
            
        Returns:
            final_mask: 只包含最大轮廓的掩码
            largest_contour: 最大轮廓
        """
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask, None
        
        # 找到面积最大的轮廓，假设这是赛道区域
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 创建新的掩码，只包含最大轮廓
        final_mask = np.zeros_like(mask)
        cv2.fillPoly(final_mask, [largest_contour], 255)
        
        return final_mask, largest_contour
    
    def detect_track_boundaries(self, mask):
        """
        检测赛道边界线
        
        Args:
            mask: 赛道区域掩码
            
        Returns:
            left_boundary: 左边界点列表
            right_boundary: 右边界点列表
        """
        height, width = mask.shape
        left_boundary = []
        right_boundary = []
        
        # 从上到下扫描每一行，找到赛道的左右边界
        for y in range(height):
            row = mask[y, :]
            # 找到该行中白色像素的位置
            white_pixels = np.where(row == 255)[0]
            
            if len(white_pixels) > 0:
                # 左边界是最左边的白色像素
                left_x = white_pixels[0]
                # 右边界是最右边的白色像素
                right_x = white_pixels[-1]
                
                left_boundary.append((left_x, y))
                right_boundary.append((right_x, y))
        
        return left_boundary, right_boundary
    
    def draw_boundaries(self, image, left_boundary, right_boundary):
        """
        在图像上绘制边界线
        
        Args:
            image: 输入图像
            left_boundary: 左边界点列表
            right_boundary: 右边界点列表
            
        Returns:
            result_image: 绘制了边界线的图像
        """
        result_image = image.copy()
        
        # 绘制左边界线（红色）
        if len(left_boundary) > 1:
            for i in range(len(left_boundary) - 1):
                cv2.line(result_image, left_boundary[i], left_boundary[i + 1], (0, 0, 255), 2)
        
        # 绘制右边界线（蓝色）
        if len(right_boundary) > 1:
            for i in range(len(right_boundary) - 1):
                cv2.line(result_image, right_boundary[i], right_boundary[i + 1], (255, 0, 0), 2)
        
        return result_image
    
    def create_visualization(self, original_image, mask, left_boundary, right_boundary):
        """
        创建分割结果的可视化
        
        Args:
            original_image: 原始图像
            mask: 赛道掩码
            left_boundary: 左边界点列表
            right_boundary: 右边界点列表
            
        Returns:
            result: 可视化结果图像
            boundary_image: 边界线图像
        """
        # 创建彩色掩码：赛道区域显示为绿色
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask == 255] = [0, 255, 0]  # 绿色表示赛道区域
        
        # 将原图与彩色掩码叠加
        result = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
        
        # 创建边界线图像
        boundary_image = self.draw_boundaries(original_image, left_boundary, right_boundary)
        
        return result, boundary_image
    
    def segment_track(self, image_path):
        """
        主要的赛道分割函数
        
        Args:
            image_path: 输入图像路径            
        Returns:
            original_image: 原始图像
            track_mask: 赛道区域掩码
            result_image: 分割结果可视化
            boundary_image: 边界线图像
            left_boundary: 左边界点列表
            right_boundary: 右边界点列表
        """
        # 读取图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        print(f"图像尺寸: {original_image.shape}")
        
        # 1. 图像预处理
        hsv_image = self.preprocess_image(original_image)
        
        # 2. 颜色分割
        color_mask = self.color_segmentation(hsv_image)
        
        # 3. 形态学操作
        morphed_mask = self.morphological_operations(color_mask)
        
        # 4. 找到最大轮廓
        track_mask, largest_contour = self.find_largest_contour(morphed_mask)
        
        # 5. 检测边界线
        left_boundary, right_boundary = self.detect_track_boundaries(track_mask)
        
        # 6. 创建结果可视化
        result_image, boundary_image = self.create_visualization(
            original_image, track_mask, left_boundary, right_boundary
        )
        
        return original_image, track_mask, result_image, boundary_image, left_boundary, right_boundary
    
    def save_results(self, original_image, track_mask, result_image, boundary_image, output_prefix="output"):
        """
        保存处理结果
        
        Args:
            original_image: 原始图像
            track_mask: 赛道掩码
            result_image: 可视化结果
            boundary_image: 边界线图像
            output_prefix: 输出文件前缀
        """
        # 保存二值掩码
        cv2.imwrite(f"{output_prefix}_mask.png", track_mask)
        
        # 保存可视化结果
        cv2.imwrite(f"{output_prefix}_result.png", result_image)
        
        # 保存边界线图像
        cv2.imwrite(f"{output_prefix}_boundary.png", boundary_image)
        
        print(f"结果已保存:")
        print(f"- 掩码图像: {output_prefix}_mask.png")
        print(f"- 可视化结果: {output_prefix}_result.png")
        print(f"- 边界线图像: {output_prefix}_boundary.png")

def main():
    """
    主函数
    """
    # 创建赛道分割对象
    segmenter = TrackSegmentation()
    
    # 输入图像路径
    input_image = "saidao.jpeg"
    
    try:
        # 执行赛道分割
        print("开始处理赛道图像...")
        original, mask, result, boundary, left_bound, right_bound = segmenter.segment_track(input_image)
        
        # 保存结果
        segmenter.save_results(original, mask, result, boundary, "track_segmentation")
        
        # 使用matplotlib显示结果
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 显示原始图像
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('原始图像', fontsize=14)
        axes[0, 0].axis('off')
        
        # 显示分割掩码
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('赛道区域掩码', fontsize=14)
        axes[0, 1].axis('off')
        
        # 显示可视化结果
        axes[1, 0].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('分割结果可视化', fontsize=14)
        axes[1, 0].axis('off')
        
        # 显示边界线检测结果
        axes[1, 1].imshow(cv2.cvtColor(boundary, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('边界线检测结果', fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('track_segmentation_collect.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出边界点统计信息
        print(f"\n边界检测统计:")
        print(f"- 左边界点数量: {len(left_bound)}")
        print(f"- 右边界点数量: {len(right_bound)}")
        
        print("处理完成！")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    main()