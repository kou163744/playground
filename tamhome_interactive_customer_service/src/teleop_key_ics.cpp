#include <stdio.h>
#include <signal.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <interactive_customer_service/Conversation.h>
#include <interactive_customer_service/RobotStatus.h>
#include <nodelet/nodelet.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace cv;

class InteractiveCustomerServiceTeleopKey
{
private:
  const std::string MSG_ARE_YOU_READY = "Are_you_ready?";
  const std::string MSG_I_AM_READY    = "I_am_ready";
  
  ros::Publisher pub_msg_;

  void messageCallback(const interactive_customer_service::Conversation::ConstPtr& message)
  {
    if(message->type.c_str()==MSG_ARE_YOU_READY)
    {
      sendMessage(pub_msg_, MSG_I_AM_READY);
    }
    else
    {
      ROS_INFO("Subscribe message:%s, %s", message->type.c_str(), message->detail.c_str());
    }
  }
  
  void robotStatusCallback(const interactive_customer_service::RobotStatus::ConstPtr& status)
  {
    ROS_DEBUG("Subscribe robot status:%s, %d, %s", status->state.c_str(), status->speaking, status->grasped_item.c_str());
  }

  void customerImageCallback(const sensor_msgs::ImageConstPtr& image)
  {
    ROS_INFO("Subscribe Image");
    
    cv_bridge::CvImagePtr cv_ptr;
    
    try
    {
      cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);

      // Display Image
      cv::Mat image_mat = cv_ptr->image;
      cv::imshow("Customer Image", image_mat);
      cv::waitKey(3000);
      cv::destroyAllWindows();

      // Save the customer image to home directory
      cv::imwrite("../CustomerImage.jpg", image_mat); // current path=/home/username/.ros
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
  }

  void sendMessage(ros::Publisher &publisher, const std::string &type, const std::string &detail="")
  {
    ROS_INFO("Send message:%s, %s", type.c_str(), detail.c_str());

    interactive_customer_service::Conversation message;
    message.type   = type;
    message.detail = detail;
    publisher.publish(message);
  }

  static void rosSigintHandler(int sig)
  {
    ros::shutdown();
  }

public:
  int run(int argc, char **argv)
  {
    ros::NodeHandle node_handle;

    // Override the default ros sigint handler.
    // This must be set after the first NodeHandle is created.
    signal(SIGINT, rosSigintHandler);
  
    ros::Rate loop_rate(10);

    ros::Subscriber sub_msg   = node_handle.subscribe<interactive_customer_service::Conversation>("/interactive_customer_service/message/customer", 100, &InteractiveCustomerServiceTeleopKey::messageCallback, this);
    ros::Subscriber sub_state = node_handle.subscribe<interactive_customer_service::RobotStatus >("/interactive_customer_service/robot_status",     100, &InteractiveCustomerServiceTeleopKey::robotStatusCallback, this);
    ros::Subscriber sub_image = node_handle.subscribe<sensor_msgs::Image>                        ("/interactive_customer_service/customer_image",   100, &InteractiveCustomerServiceTeleopKey::customerImageCallback, this);

    pub_msg_ = node_handle.advertise<interactive_customer_service::Conversation>("/interactive_customer_service/message/robot", 10);

    ros::spin();

    return EXIT_SUCCESS;
  }
};


int main(int argc, char **argv)
{
  ros::init(argc, argv, "interactive_customer_teleop_key");

  InteractiveCustomerServiceTeleopKey teleopkey;
  return teleopkey.run(argc, argv);
};

