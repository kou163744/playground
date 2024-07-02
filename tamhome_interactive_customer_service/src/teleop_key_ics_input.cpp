#include <stdio.h>
#include <signal.h>
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <iostream>
#include <fstream>
#include <string>
#include <interactive_customer_service/Conversation.h>

class InteractiveCustomerServiceTeleopKeyInput
{
private:
  const std::string MSG_ROBOT_MESSAGE = "robot_message";
  const std::string MSG_TAKE_ITEM     = "take_item";
  const std::string MSG_GIVE_ITEM     = "give_item";
  const std::string MSG_GIVE_UP       = "Give_up";
  
  const std::string MSG1 = "There are several candidates. Is it green?";
  const std::string MSG2 = "Is this what you want?";
  
  const std::string OBJ1 = "super_big_choco-1000";
  const std::string OBJ2 = "11_xylitol-1000";
  const std::string OBJ3 = "chipstar_consomme-2000";
  const std::string OBJ4 = "irohasu-3000";
  const std::string OBJ5 = "donbee_soba";
  
  ros::Publisher pub_msg_;
  
  void showHelp(int goods_num)
  {
    std::cout << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Operate by Keyboard" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "m1: RobotMessage '" << MSG1 << "'" << std::endl;
    std::cout << "m2: RobotMessage '" << MSG2 << "'" << std::endl;
    std::cout << "t1: Take " << OBJ1 << std::endl;
    std::cout << "t2: Take " << OBJ2 << std::endl;
    std::cout << "t3: Take " << OBJ3 << std::endl;
    std::cout << "t4: Take " << OBJ4 << std::endl;
    std::cout << "t5: Take " << OBJ5 << std::endl;
    std::cout << "1-" << goods_num << ": Take Item by Number" << std::endl;
    std::cout << "g: Give Item " << std::endl;
    std::cout << "giveup: Give Up " << std::endl;
    std::cout << "---------------------------" << std::endl;
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
    
    exit(sig);
  }

public:
  int run(int argc, char **argv)
  {
    ros::NodeHandle node_handle;

    std::string goods_list_path;

    node_handle.param<std::string>("/teleop_key_ics_input/goods_list_path", goods_list_path, "GoodsList.txt");

    std::ifstream ifs(goods_list_path.c_str());

    if (ifs.fail()) 
    {
        std::cerr << "Failed to open the GoodsList file." << std::endl;
        return -1;
    }
    
    const int max_goods_num = 1000;
    int goods_num = 0;
    std::string goods_name;
    std::string goods_name_list[max_goods_num];

    // Print Goods List
    std::cout << "Goods List" << std::endl;

    while (std::getline(ifs, goods_name)) 
    {
        std::cout << "#" << (goods_num+1) << " :" << goods_name << std::endl;
        goods_name_list[goods_num] = goods_name;
        goods_num++;
    }
    ifs.close();
    
    // Override the default ros sigint handler.
    // This must be set after the first NodeHandle is created.
    signal(SIGINT, rosSigintHandler);
  
    ros::Rate loop_rate(10);

    pub_msg_ = node_handle.advertise<interactive_customer_service::Conversation>("/interactive_customer_service/message/robot", 10);

    std::string input_line;

    while (ros::ok())
    {
      showHelp(goods_num);

      std::cout << "input: ";

      std::getline(std::cin, input_line);

      if     (input_line=="m1"){ sendMessage(pub_msg_, MSG_ROBOT_MESSAGE, MSG1); }
      else if(input_line=="m2"){ sendMessage(pub_msg_, MSG_ROBOT_MESSAGE, MSG2); }
      else if(input_line=="t1"){ sendMessage(pub_msg_, MSG_TAKE_ITEM, OBJ1); }
      else if(input_line=="t2"){ sendMessage(pub_msg_, MSG_TAKE_ITEM, OBJ2); }
      else if(input_line=="t3"){ sendMessage(pub_msg_, MSG_TAKE_ITEM, OBJ3); }
      else if(input_line=="t4"){ sendMessage(pub_msg_, MSG_TAKE_ITEM, OBJ4); }
      else if(input_line=="t5"){ sendMessage(pub_msg_, MSG_TAKE_ITEM, OBJ5); }
      else if(input_line=="g"){ sendMessage(pub_msg_, MSG_GIVE_ITEM); }
      else if(input_line=="giveup"){ sendMessage(pub_msg_, MSG_GIVE_UP); }
      else
      {
        try
        {
          int input_num;
    
          input_num = std::stoi(input_line);

          if(input_num>0 && input_num<=goods_num)
          {
            sendMessage(pub_msg_, MSG_TAKE_ITEM, goods_name_list[input_num-1].c_str());
          }
          else
          {
            std::cout << "!!! Invalid input !!!" << std::endl;
          }
        }
        catch (const std::invalid_argument& e) 
        {
          std::cout << "!!! Invalid input !!!" << std::endl;
        }
        catch (const std::out_of_range & e) 
        {
          std::cout << "!!! Invalid input !!!" << std::endl;
        }
      }
        std::cin.clear();
//      std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');

      ros::spinOnce();

      loop_rate.sleep();
    }

    return EXIT_SUCCESS;
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "interactive_customer_teleop_key_input");
  InteractiveCustomerServiceTeleopKeyInput teleopkeyinput;
  return teleopkeyinput.run(argc, argv);
}

