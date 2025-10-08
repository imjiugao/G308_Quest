#include "ros/ros.h"
#include "std_msgs/String.h" // 我们要发布的消息类型

int main(int argc, char **argv)
{
    // 初始化ROS节点，节点名为 "cpp_publisher"
    ros::init(argc, argv, "cpp_publisher", ros::init_options::AnonymousName);
    
    // 创建节点句柄
    ros::NodeHandle nh;
    
    // 创建一个Publisher，发布到话题 "student_id" 上
    // 消息类型是std_msgs::String，队列大小是10
    ros::Publisher pub = nh.advertise<std_msgs::String>("student_id", 10);
    
    // 设置循环的频率（这里1秒1次）
    ros::Rate loop_rate(1);
    
    // 在ROS正常运行的情况下循环
    while (ros::ok())
    {
        // 创建一个要发布的消息对象
        std_msgs::String msg;
        
        // 设置消息的数据内容
        msg.data = "2024111040 from C++";
        
        // 发布消息
        pub.publish(msg);
        
        // 在终端打印日志信息
        ROS_INFO("C++ Published: %s", msg.data.c_str());
        
        // 按照循环频率延时
        loop_rate.sleep();
    }
    
    return 0;
}
