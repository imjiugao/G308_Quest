"""
Apriltag码识别程序

设计思路：
1. 使用OpenCV库获取摄像头视频流
2. 使用pupil_apriltags库进行Apriltag码的检测
3. 对检测到的每个Apriltag码进行标记和信息显示
4. 实时显示检测结果

"""

import cv2
import numpy as np
from pupil_apriltags import Detector

def main():
    """
    主函数，实现Apriltag码的检测和显示
    """
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 获取摄像头的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 初始化Apriltag检测器
    # 参数说明：
    # - families: 要检测的Apriltag码的类型，这里使用tag36h11
    # - nthreads: 使用的线程数
    # - quad_decimate: 图像缩小因子，用于加速检测
    # - quad_sigma: 应用于输入图像的高斯模糊sigma值
    # - refine_edges: 是否细化边缘
    # - decode_sharpening: 解码时的锐化程度
    # - debug: 是否启用调试模式
    detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    
    print("按'q'键退出程序")
    
    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        
        # 如果读取失败，退出循环
        if not ret:
            print("无法获取图像帧")
            break
        
        # 将图像转换为灰度图，Apriltag检测需要灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测Apriltag码
        # 返回的结果是一个包含多个检测结果的列表
        # 每个检测结果包含tag_id, hamming, decision_margin, homography等信息
        results = detector.detect(gray)
        
        # 显示检测到的Apriltag码的数量
        cv2.putText(frame, f"检测到 {len(results)} 个Apriltag码", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 处理每个检测到的Apriltag码
        for result in results:
            # 获取tag的ID
            tag_id = result.tag_id
            
            # 获取tag的四个角点坐标
            # 这些坐标是浮点数，需要转换为整数用于绘图
            corners = result.corners.astype(np.int32)
            
            # 绘制tag的边界
            # 使用绿色线条连接四个角点
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            
            # 计算tag的中心点
            center = np.mean(corners, axis=0).astype(np.int32)
            
            # 在tag中心显示tag的ID
            cv2.putText(frame, f"ID: {tag_id}", (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 显示每个角点的坐标
            for i, corner in enumerate(corners):
                cv2.circle(frame, (corner[0], corner[1]), 4, (255, 0, 0), -1)
                cv2.putText(frame, f"{i}", (corner[0] + 5, corner[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 显示tag的姿态信息
            pose_info = f"Margin: {result.decision_margin:.2f}"
            cv2.putText(frame, pose_info, (center[0] - 10, center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # 显示结果图像
        cv2.imshow("Apriltag检测", frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()