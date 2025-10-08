# G308_Quest5

使用Python和OpenCV实现Apriltag码的实时识别

##设计思路：

1. 使用OpenCV库获取摄像头视频流
2. 使用pupil_apriltags库进行Apriltag码的检测
3. 对检测到的每个Apriltag码进行标记和信息显示
4. 实时显示检测结果

## 功能

- 同时检测多个Apriltag码
- 显示Apriltag码的ID和边界
- 显示Apriltag码数量

## 示例输出

程序运行时，会在视频窗口中显示：
- 检测到的Apriltag码总数
- 每个Apriltag码的ID
- 每个Apriltag码的边界和角点
- 每个Apriltag码的决策边界值(decision margin)