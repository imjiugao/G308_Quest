'''
model:预训练模型
data:数据集配置文件路径
epochs:训练轮数
imgsz:图像尺寸
batch:批次大小
device:指定GPU
workers:数据加载线程数
'''
from ultralytics import YOLO
import torch.multiprocessing as mp
from multiprocessing import freeze_support

def train_model():
    model = YOLO("yolov8m.pt")
    
    results = model.train(
        data="C:/Users/ASUS/Desktop/G308/G3082026/zyzf/Q5/dataset.yaml",
        epochs=300,
        imgsz=640,
        batch=16,
        device=0,  
        workers=4,  
        name="vdetect"
    )

if __name__ == '__main__':
    # Windows 多进程设置
    freeze_support()
    
    # 设置启动方法为 spawn
    mp.set_start_method('spawn', force=True)
    
    # 在独立进程中运行训练
    p = mp.Process(target=train_model)
    p.start()
    p.join()