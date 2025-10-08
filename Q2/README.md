# G308_Quest2
# 三种赛道区域分割程序

## 项目简介
- 方案A（颜色分割）：基于 HSV 颜色空间的经典黄色检测
- 方案B（边缘几何）：结合边缘检测、霍夫直线、区域生长与凸包约束
- 方案C（优化检测）：LAB+HSV 双重分割、连通组件筛选与多项式边界拟合，兼顾精度与速度

## 三种方案概览

### 方案A：HSV颜色分割（track_segmentation.py）
- 思路：HSV 转换 → inRange 阈值分割 → 形态学开闭 → 最大轮廓 → 边界扫描
- 输出文件：
- `track_segmentation_mask.png`：二值掩码（白=赛道）
- `track_segmentation_result.png`：赛道区域绿色叠加
- `track_segmentation_boundary.png`：左右边界线（红/蓝）
- `track_segmentation_collect.png`：四合一展示


### 方案B：基于边缘与几何（track_detection_edge_based.py）
- 思路：双边滤波 → Canny+Sobel 多尺度边缘 → 霍夫直线 → KMeans颜色聚类 → 基于黄色/聚类的种子点 → 区域生长 → 凸包约束 → 形态学细化 → 边界拟合
- 输出文件：
- `edge_based_track_mask.png`：最终掩码
- `edge_based_track_result.png`：赛道区域与边界可视化
- `edge_based_track_edges.png`：组合边缘图
- `edge_based_track_clustered.png`：颜色聚类结果
- `edge_based_track_comprehensive.png`：九宫格综合展示

- 关键参数（类 EdgeBasedTrackDetector）：
```python
# Canny
self.canny_low = 50
self.canny_high = 150
# Hough
self.hough_threshold = 100
self.min_line_length = 50
self.max_line_gap = 10
# 区域生长
self.color_tolerance = 30
self.region_min_size = 1000
```

### 方案C：优化的边界检测（track_detection_optimized.py）
- 思路：高斯滤波+CLAHE → LAB 与 HSV 双重颜色分割（交/并集自适应） → 连通组件（面积/长宽比筛选） → 形态学优化 → 主赛道提取 → 二次多项式边界拟合 → 平滑边界绘制
- 优势：高效、对光照变化鲁棒，边界平滑且准确，适合实时
- 适用场景：需要速度与稳定性兼顾的应用
- 输出文件：
- `optimized_track_mask.png`：主赛道掩码
- `optimized_track_result.png`：平滑边界可视化
- `optimized_track_comparison.png`：六宫格对比

- 关键参数（类 OptimizedTrackDetector）：
```python
# LAB黄色范围
self.lab_lower = np.array([20, 15, 40])
self.lab_upper = np.array([100, 35, 80])
# HSV备用范围
self.hsv_lower = np.array([15, 50, 50])
self.hsv_upper = np.array([35, 255, 255])
# 面积/形态学
self.kernel_size = 5
self.min_area = 5000
self.max_area = 500000
```

## 文件说明
- `track_segmentation.py`：方案A（HSV颜色分割）
- `track_detection_edge_based.py`：方案B（边缘与几何约束）
- `track_detection_optimized.py`：方案C（优化的边界检测）
- `requirements.txt`：依赖列表
- `README.md`：说明文档