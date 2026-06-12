#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indoor Navigation Evaluator for Drone with NavDP
=================================================
Entry point for indoor VLN navigation using:
- RealSense D435i camera
- Gemini-2.5-pro for strategic decisions
- Qwen2.5-VL-32B for tactical bbox detection
- NavDP for trajectory planning
"""

import rospy
from vln_core.logger import logger
from vln_core.indoor_2_model import IndoorNavEvaluator


def ros_main():
    rospy.init_node("indoor_evaluator", anonymous=False)

    loop_hz = float(rospy.get_param("~loop_hz", 10.0))
    if loop_hz <= 0:
        loop_hz = 10.0
    rate = rospy.Rate(loop_hz)

    logger.info(f"[indoor_evaluator] ROS loop started. loop_hz={loop_hz}")

    evaluator = IndoorNavEvaluator()

    while not rospy.is_shutdown():
        try:
            evaluator.run()
            if getattr(evaluator, "nav_done", False):
                rospy.loginfo("Indoor navigation task completed")
                break
        except Exception as e:
            rospy.logerr_throttle(1.0, f"[indoor_evaluator] evaluator.run() exception: {e}")
            import traceback
            traceback.print_exc()
        rate.sleep()


if __name__ == "__main__":
    ros_main()
