cmake_minimum_required(VERSION 2.8.3)
project(interactive_customer_service)

## Add support for C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  trajectory_msgs
  tf
  message_generation
  cv_bridge
)
find_package(OpenCV REQUIRED)

################################################
## Declare ROS messages, services and actions ##
################################################

add_message_files(
  FILES
  Conversation.msg
  RobotStatus.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime cv_bridge
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

include_directories(include ${catkin_INCLUDE_DIRS})
link_directories(/usr/local/lib)

add_executable(interactive_customer_service_sample src/interactive_customer_service_sample.cpp)
target_link_libraries(interactive_customer_service_sample ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(interactive_customer_service_sample ${PROJECT_NAME}_generate_messages_cpp)

add_executable(teleop_key_ics src/teleop_key_ics.cpp)
target_link_libraries(teleop_key_ics ${catkin_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(teleop_key_ics ${PROJECT_NAME}_generate_messages_cpp)

add_executable(teleop_key_ics_input src/teleop_key_ics_input.cpp)
target_link_libraries(teleop_key_ics_input ${catkin_LIBRARIES})
