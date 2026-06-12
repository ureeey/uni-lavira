"""Dual-arm gimbal controller for the Cobot Magic panorama scan.

This is NOT a manipulation controller. The two arms hold the side cameras and
sweep them through a fixed joint sequence so the robot can capture a panorama
(see ``RobotController.collect_panoramic_images``). The class publishes smoothed
joint commands on ``/master/joint_left`` / ``/master/joint_right`` and tracks the
puppet arm state on ``/puppet/joint_left`` / ``/puppet/joint_right``.

Behaviour is ported verbatim from the original ``AutomatedController``: a
background control loop interpolates each joint toward its target by a fixed
per-step increment (``Config.ARM_BASE_STEPS`` scaled by ``Config.ARM_SPEED``),
publishing at ``Config.ARM_PUBLISH_RATE`` Hz.

Import safety
-------------
``rospy`` and the ROS message types are imported lazily inside ``__init__`` /
methods so ``import robot.arm_controller`` succeeds on a machine without ROS.
"""
from __future__ import annotations

import math
import threading
from typing import List, Optional

import numpy as np

from config import Config
from utils import print_info


class AutomatedController:
    """Dual-arm camera gimbal driven by smoothed joint-space goals."""

    def __init__(self) -> None:
        # Lazy ROS imports so this module is importable without ROS installed.
        import rospy
        from sensor_msgs.msg import JointState

        self._JointState = JointState

        self.left_arm_pub = rospy.Publisher(
            "/master/joint_left", JointState, queue_size=10
        )
        self.right_arm_pub = rospy.Publisher(
            "/master/joint_right", JointState, queue_size=10
        )
        self.left_arm_current_pos: List[float] = []
        self.right_arm_current_pos: List[float] = []
        self.left_arm_state_lock = threading.Lock()
        self.right_arm_state_lock = threading.Lock()
        rospy.Subscriber("/puppet/joint_left", JointState, self.left_arm_callback)
        rospy.Subscriber("/puppet/joint_right", JointState, self.right_arm_callback)

        self.publish_rate = Config.ARM_PUBLISH_RATE
        # Per-step joint increment: base profile scaled by ARM_SPEED (originally
        # ([0.01]*6 + [0.2]) * 10).
        self.arm_steps_length = (
            np.array(Config.ARM_BASE_STEPS) * Config.ARM_SPEED
        ).tolist()
        self.JOINT_NAMES = [f"joint{i}" for i in range(7)]
        self.target_left_joints: List[float] = []
        self.target_right_joints: List[float] = []
        self.arm_control_thread: Optional[threading.Thread] = None
        self.run_thread = threading.Event()
        self.wait_for_initial_state()
        self.start_control_thread()

    def left_arm_callback(self, msg) -> None:
        with self.left_arm_state_lock:
            self.left_arm_current_pos = list(msg.position)

    def right_arm_callback(self, msg) -> None:
        with self.right_arm_state_lock:
            self.right_arm_current_pos = list(msg.position)

    def wait_for_initial_state(self) -> None:
        """Block until both arms report an initial joint state, then seed targets."""
        import rospy

        print_info("Waiting for initial arm states...")
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            with self.left_arm_state_lock, self.right_arm_state_lock:
                if self.left_arm_current_pos and self.right_arm_current_pos:
                    self.target_left_joints = self.left_arm_current_pos[:]
                    self.target_right_joints = self.right_arm_current_pos[:]
                    print_info("Initial arm states received!")
                    return
            rate.sleep()

    def _arm_control_loop(self) -> None:
        """Background loop: step each joint toward its target and publish."""
        import rospy
        from std_msgs.msg import Header

        rate = rospy.Rate(self.publish_rate)
        while self.run_thread.is_set() and not rospy.is_shutdown():
            with self.left_arm_state_lock, self.right_arm_state_lock:
                tgt_l = self.target_left_joints[:]
                tgt_r = self.target_right_joints[:]
                cur_l = self.left_arm_current_pos[:]
                cur_r = self.right_arm_current_pos[:]

            if not all([tgt_l, tgt_r, cur_l, cur_r]):
                rate.sleep()
                continue

            nxt_l, nxt_r = cur_l[:], cur_r[:]
            for i in range(7):
                for nxt, tgt, cur in [
                    (nxt_l, tgt_l, cur_l),
                    (nxt_r, tgt_r, cur_r),
                ]:
                    diff = tgt[i] - cur[i]
                    if abs(diff) > self.arm_steps_length[i]:
                        nxt[i] += self.arm_steps_length[i] * math.copysign(1, diff)
                    else:
                        nxt[i] = tgt[i]

            ts = rospy.Time.now()
            self.left_arm_pub.publish(
                self._JointState(
                    name=self.JOINT_NAMES, position=nxt_l, header=Header(stamp=ts)
                )
            )
            self.right_arm_pub.publish(
                self._JointState(
                    name=self.JOINT_NAMES, position=nxt_r, header=Header(stamp=ts)
                )
            )
            rate.sleep()

    def start_control_thread(self) -> None:
        if not self.arm_control_thread or not self.arm_control_thread.is_alive():
            self.run_thread.set()
            self.arm_control_thread = threading.Thread(
                target=self._arm_control_loop, daemon=True
            )
            self.arm_control_thread.start()

    def stop_control_thread(self) -> None:
        self.run_thread.clear()
        if self.arm_control_thread:
            self.arm_control_thread.join(timeout=1)

    def move_to_goal(
        self,
        left_target: List[float],
        right_target: List[float],
        tolerance: float = 0.05,
        timeout: float = 30,
    ) -> bool:
        """Set a new joint goal and block until reached or ``timeout`` elapses."""
        import rospy

        print_info("Setting new arm goal.")
        with self.left_arm_state_lock, self.right_arm_state_lock:
            self.target_left_joints = left_target
            self.target_right_joints = right_target

        start_time = rospy.Time.now()
        rate = rospy.Rate(20)
        while (
            not rospy.is_shutdown()
            and (rospy.Time.now() - start_time).to_sec() < timeout
        ):
            with self.left_arm_state_lock, self.right_arm_state_lock:
                cur_l = self.left_arm_current_pos[:]
                cur_r = self.right_arm_current_pos[:]
            if not all([cur_l, cur_r]):
                rate.sleep()
                continue
            if all(
                abs(left_target[i] - cur_l[i]) < tolerance for i in range(7)
            ) and all(
                abs(right_target[i] - cur_r[i]) < tolerance for i in range(7)
            ):
                return True
            rate.sleep()
        return False
