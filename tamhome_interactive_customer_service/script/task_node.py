#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
import smach
import rosnode

from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from state_machine import standard
from state_machine import interaction, pickup, give


class InteractiveCleanupStateMachine():
    def __init__(self) -> None:
        """
        必要なモジュールを初期化
        """
        self.start_state = rospy.get_param("~start_state", "Start")
        rospy.Service("/set/init_state", Trigger, self.cb_set_state)

        # ステートマシンの宣言
        self.sm = smach.StateMachine(outcomes=["exit"])

        with self.sm:
            smach.StateMachine.add(
                "Init",
                standard.Init(["next", "except"]),
                transitions={"next": "Wait4Start", "except": "Except"},
            )
            smach.StateMachine.add(
                "Wait4Start",
                standard.Wait4Start(["next", "except"]),
                transitions={"next": self.start_state, "except": "Except"},
            )
            smach.StateMachine.add(
                "Start",
                standard.Start(["next", "except"]),
                transitions={"next": "Interaction", "except": "Except"},
            )

            # Interactive customer service
            smach.StateMachine.add(
                "Interaction",
                interaction.Interaction(["next", "except"]),
                transitions={
                    "next": "Pickup",
                    "except": "Except",
                },
            )
            smach.StateMachine.add(
                "Pickup",
                pickup.Pickup(["next", "loop", "except"]),
                transitions={
                    "next": "Give",
                    "loop": "Pickup",
                    "except": "Except",
                },
            )
            smach.StateMachine.add(
                "Give",
                give.Give(["next", "loop", "except"]),
                transitions={
                    "next": "Finish",
                    "loop": "Give",
                    "except": "Except",
                },
            )

            smach.StateMachine.add(
                "Finish",
                standard.Finish(["finish"]),
                transitions={"finish": "Init"},
            )
            smach.StateMachine.add(
                "Except",
                standard.Except(["except", "recovery"]),
                transitions={
                    "except": "exit",
                    "recovery": "Init"
                },
            )

    def delete(self) -> None:
        del self.sm

    def run(self) -> None:
        self.sm.execute()

    def cb_set_state(self, req: TriggerRequest) -> TriggerResponse:
        # target_state = "Init"
        # # self.loginfo(f"[set_state] set inital state by timeout: {target_state}")
        # if self.sm.is_running():
        #     self.sm.request_preempt()
        #     rospy.sleep(3)
        # self.sm.set_initial_state([target_state])
        # self.sm.execute()
        try:
            node_name = "/state_machine"
            rosnode.kill_nodes([node_name])
            return TriggerResponse(success=True, message=f"State machine reset")
        except Exception as e:
            self.logwarn(e)
            return TriggerResponse(success=False, message=f"Please restart")


def main():
    rospy.init_node(os.path.basename(__file__).split(".")[0])

    cls = InteractiveCleanupStateMachine()
    rospy.on_shutdown(cls.delete)
    try:
        cls.run()
    except rospy.exceptions.ROSException as e:
        rospy.logerr("[" + rospy.get_name() + "]: FAILURE")
        rospy.logerr("[" + rospy.get_name() + "]: " + str(e))


if __name__ == "__main__":
    main()
