#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import smach
from tamlib.utils import Logger
from interactive_customer_service.msg import Conversation, RobotStatus


class Pickup(smach.State, Logger):
    def __init__(self, outcomes):
        smach.State.__init__(
            self, outcomes=outcomes,
            input_keys=[],
            output_keys=[]
        )
        Logger.__init__(self, loglevel="INFO")

        self.take_item_msg = "take_item"
        self.take_item_pub = rospy.Publisher("/interactive_customer_service/message/robot", Conversation, queue_size=10)

    def execute(self, userdata):
        """物体検出と把持をする関数
        """
        target_item = rospy.get_param("/interactive_customer_service/orderd_item", "none")
        target_id = rospy.get_param("/interactive_customer_service/orderd_id", 999)

        msg = Conversation()
        msg.type = self.take_item_msg
        msg.detail = target_item

        self.take_item_pub.publish(msg)
        rospy.sleep(2)

        start_time = rospy.Time.now()
        timeout = rospy.Duration(25)

        while not rospy.is_shutdown():
            try:
                if rospy.Time.now() - start_time > timeout:
                    self.loginfo("タイムアウトによりループを抜ける")
                    break

                robot_status = rospy.wait_for_message("/interactive_customer_service/robot_status", RobotStatus, timeout=5)
                if robot_status.state == "in_conversation":
                    break
                elif robot_status.state == "moving":
                    self.logdebug("移動中に付き，ステート遷移を一時停止")
                else:
                    continue
            except Exception as e:
                self.logwarn(e)

        rospy.sleep(1)
        return "next"
