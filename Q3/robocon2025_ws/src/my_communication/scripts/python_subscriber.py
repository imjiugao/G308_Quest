#!/usr/bin/env python3
import rospy
from std_msgs.msg import String # 我们要接收的消息类型

# 这是一个回调函数
# 当收到名为 ‘student_id’ 的话题上有新消息时，这个函数会被自动调用
def callback(msg):
    # 在终端打印接收到的消息
    rospy.loginfo("Python Received: %s", msg.data)

def listener():
    # 初始化ROS节点，节点名为 "python_subscriber"
    rospy.init_node('python_subscriber', anonymous=True)
    
    # 创建一个Subscriber，订阅名为 "student_id" 的话题
    # 当收到新消息时，ROS会自动调用 callback 函数
    rospy.Subscriber('student_id', String, callback)
    
    # rospy.spin() 使Python代码不会退出，直到节点被显式关闭
    # 它只是保持程序运行以便回调函数可以被一直调用
    rospy.spin()

if __name__ == '__main__':
    listener()