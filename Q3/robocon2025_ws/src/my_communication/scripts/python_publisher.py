#!/usr/bin/env python3
import rospy
from std_msgs.msg import String # 发布的消息类型

def publisher():
    # 初始化ROS节点，节点名为 "python_publisher"
    rospy.init_node('python_publisher', anonymous=True)
    
    # 创建一个Publisher，发布到话题 "student_id" 上
    # 消息类型是String，队列大小是10
    pub = rospy.Publisher('student_id', String, queue_size=10)
    
    # 设置循环的频率（这里1秒1次）
    rate = rospy.Rate(1)
    
    # 在ROS正常运行的情况下循环
    while not rospy.is_shutdown():
        # 设置要发布的消息内容
        student_id = "2024111040 from Python"
        
        # 在终端打印日志信息
        rospy.loginfo("Python Published: " + student_id)
        
        # 发布消息
        pub.publish(student_id)
        
        # 按照循环频率延时
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass