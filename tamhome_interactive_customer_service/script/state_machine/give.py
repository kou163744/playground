#!/usr/bin/env python3
# coding:utf-8

import rospy
import smach
from tamlib.utils import Logger
from interactive_customer_service.msg import Conversation


class Give(smach.State, Logger):
    def __init__(self, outcomes):
        smach.State.__init__(
            self, outcomes=outcomes,
            input_keys=[],
            output_keys=[]
        )
        Logger.__init__(self, loglevel="INFO")

        self.give_item_msg = "give_item"
        self.give_item_pub = rospy.Publisher("/interactive_customer_service/message/robot", Conversation, queue_size=10)

    def execute(self, userdata):
        """商品をかごに入れる
        """
        self.loginfo("商品をかごに入れる")
        msg = Conversation()
        msg.type = self.give_item_msg
        msg.detail = self.give_item_msg

        for _ in range(5):
            self.give_item_pub.publish(msg)
            rospy.sleep(0.5)

        return "next"
