#include "ros/ros.h"
#include "std_msgs/String.h" // 我们要接收的消息类型

// 这是一个回调函数
// 当收到名为 ‘student_id’ 的话题上有新消息时，这个函数会被自动调用
void chatterCallback(const std_msgs::String::ConstPtr& msg)
{
    // 在终端打印接收到的消息
    ROS_INFO("C++ Received: %s", msg->data.c_str());
}

int main(int argc, char **argv)
{
    // 初始化ROS节点，节点名为 "cpp_subscriber"
    ros::init(argc, argv, "cpp_subscriber", ros::init_options::AnonymousName);
    
    // 创建节点句柄
    ros::NodeHandle nh;
    
    // 创建一个Subscriber，订阅名为 "student_id" 的话题
    // 当收到新消息时，ROS会自动调用 chatterCallback 函数
    ros::Subscriber sub = nh.subscribe("student_id", 10, chatterCallback);
    
    // ros::spin() 进入自循环，可以尽可能快的调用消息回调函数
    // 当收到ROS的停止信号（如Ctrl+C）时，它会退出
    ros::spin();
    
    return 0;
}