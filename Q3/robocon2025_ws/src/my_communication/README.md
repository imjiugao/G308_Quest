# my_communication ROS Package

实现了基于话题（Topic）的简单消息发布与订阅。

## 功能概述

本包提供了C++和Python两种语言版本的示例节点：
- **发布者 (Publisher)**：周期性地向 `/student_id` 话题发布包含学号的字符串消息
- **订阅者 (Subscriber)**：订阅 `/student_id` 话题并打印接收到的消息

## 节点说明

### 发布者节点
- **cpp_publisher**: C++版本的发布者节点
- **python_publisher.py**: Python版本的发布者节点

### 订阅者节点
- **cpp_subscriber**: C++版本的订阅者节点
- **python_subscriber.py**: Python版本的订阅者节点

## 消息类型
- 使用标准消息类型：`std_msgs/String`

## 安装和运行

### 编译工作空间
```bash
cd ~/ROS/robocon2025_ws
catkin_make
source devel/setup.bash
```

### 运行方法

1. 启动ROS核心：
   ```bash
   roscore
   ```

2. 运行发布者：
   ```bash
   # C++版本
   rosrun my_communication cpp_publisher
   # Python版本
   rosrun my_communication python_publisher.py
   ```

3. 运行订阅者：
   ```bash
   # C++版本
   rosrun my_communication cpp_subscriber
   # Python版本
   rosrun my_communication python_subscriber.py
   ```


## 文件结构
```
my_communication/
├── CMakeLists.txt          # C++编译配置
├── package.xml             # 包依赖配置
├── README.md              # 本文档
├── src/
│   ├── cpp_publisher.cpp   # C++发布者源码
│   └── cpp_subscriber.cpp  # C++订阅者源码
├── scripts/
│   ├── python_publisher.py # Python发布者源码
│   └── python_subscriber.py # Python订阅者源码
└── launch/
    └── run_all.launch     # 启动文件
```

## 测试验证

```bash
# 查看活跃话题
rostopic list
# 实时查看消息内容
rostopic echo /student_id
# 可视化节点连接
rqt_graph
```

