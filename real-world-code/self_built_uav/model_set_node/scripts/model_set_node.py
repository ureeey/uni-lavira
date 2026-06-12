#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
import numpy as np
import rospy
import copy
import tf.transformations as tft

from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from std_msgs.msg import UInt8


def norm3(dx, dy, dz):
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def wrap_deg180(a: float) -> float:
    """Wrap angle to [-180, 180)"""
    return (a + 180.0) % 360.0 - 180.0

def yaw_from_quat(q_msg) -> float:
    q = [q_msg.x, q_msg.y, q_msg.z, q_msg.w]
    _, _, yaw = tft.euler_from_quaternion(q)
    return yaw

def quat_from_yaw(yaw: float):
    q = tft.quaternion_from_euler(0.0, 0.0, yaw)
    class _Q: pass
    qq = _Q()
    qq.x, qq.y, qq.z, qq.w = q[0], q[1], q[2], q[3]
    return qq


class ModelSetNode:
    # -------- nav state enum (UInt8) --------
    STATE_IDLE     = 0
    STATE_TAKEOFF  = 1
    STATE_MOVING   = 2
    STATE_ARRIVED  = 3
    STATE_FAILSAFE = 4

    def __init__(self):
        self.last_yaw = 0.0
        self._takeoff_yaw_hold = None  # radians
        self._follow_yaw_hold  = None  # radians

        self.rate_hz = rospy.get_param("~rate_hz", 20.0)
        self.track_speed = rospy.get_param("~track_speed", 1.0)
        self.reach_tol = rospy.get_param("~reach_tol", 0.20)

        self.yaw_tol_deg = rospy.get_param("~yaw_tol_deg", 12.0)
        self.yaw_hold_sec = rospy.get_param("~yaw_hold_sec", 0.8)
        self._yaw_arrive_enter_time = None

        self.setpoint_frame_id = rospy.get_param(“~setpoint_frame_id”, “map”)

        self.arrive_tol = rospy.get_param(“~arrive_tol”, self.reach_tol)
        self.arrive_speed_tol = rospy.get_param(“~arrive_speed_tol”, 0.15)
        self.arrive_hold_sec = rospy.get_param(“~arrive_hold_sec”, 0.5)

        rospy.wait_for_service(“/mavros/cmd/arming”)
        rospy.wait_for_service(“/mavros/set_mode”)
        self.arming_srv = rospy.ServiceProxy(“/mavros/cmd/arming”, CommandBool)
        self.set_mode_srv = rospy.ServiceProxy(“/mavros/set_mode”, SetMode)

        self.lock = threading.Lock()

        self.current_state = State()

        self.have_local_pose = False
        self.local_pose = PoseStamped()

        self.have_lio = False

        self.have_local_odom = False
        self.vel_enu = np.zeros(3, dtype=np.float64)

        self.setpoint = PoseStamped()
        self.setpoint.pose.orientation.w = 1.0

        self.have_goal = False
        self.goal = PoseStamped()

        self.have_hold_setpoint = False
        self.hold_setpoint = PoseStamped()
        self.hold_setpoint.header.frame_id = self.setpoint_frame_id

        rospy.Subscriber("/mavros/state", State, self._state_callback, queue_size=10)
        rospy.Subscriber("unilavira/waypoint", PoseStamped, self._waypoint_callback, queue_size=10)

        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._local_pose_callback, queue_size=50)
        rospy.Subscriber("/Odometry", Odometry, self._lio_odom_callback, queue_size=200)
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self._local_odom_callback, queue_size=50)

        self.vision_pub = rospy.Publisher("/mavros/vision_pose/pose", PoseStamped, queue_size=200)
        self.setpoint_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=20)
        self.nav_state_pub = rospy.Publisher("/unilavira/nav_state", UInt8, queue_size=10, latch=True)
        self.nav_state = None
        self._set_nav_state(self.STATE_IDLE)
        self._arrive_enter_time = None


        rospy.loginfo("model_set_node: rate=%.1fHz speed=%.2fm/s align=%s",
                      self.rate_hz, self.track_speed, str(True))

    def _set_nav_state(self, s: int):
        """Publish nav_state if changed."""
        if getattr(self, "nav_state", None) == s:
            return
        self.nav_state = int(s)
        msg = UInt8()
        msg.data = self.nav_state
        self.nav_state_pub.publish(msg)
        rospy.loginfo("nav_state -> %d", self.nav_state)

    def _update_hold_setpoint_locked(self):
        “””Cache current setpoint as last valid hold point. Must be called under lock.”””
        self.hold_setpoint = copy.deepcopy(self.setpoint)
        self.hold_setpoint.header.frame_id = self.setpoint_frame_id
        self.have_hold_setpoint = True

    def _apply_hold_setpoint_locked(self):
        """Apply cached hold_setpoint to current setpoint to prevent drift. Must be called under lock."""
        if not self.have_hold_setpoint:
            return False
        self.setpoint.pose.position = self.hold_setpoint.pose.position
        self.setpoint.pose.orientation = self.hold_setpoint.pose.orientation
        return True

    def _state_callback(self, msg: State):
        self.current_state = msg

    def _local_pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.local_pose = msg
            self.have_local_pose = True

    def _local_odom_callback(self, msg: Odometry):
        with self.lock:
            self.vel_enu[0] = msg.twist.twist.linear.x
            self.vel_enu[1] = msg.twist.twist.linear.y
            self.vel_enu[2] = msg.twist.twist.linear.z
            self.have_local_odom = True

    def _waypoint_callback(self, wp: PoseStamped):
        with self.lock:
            self.goal = wp
            self.have_goal = True
            self._arrive_enter_time = None
            self._yaw_arrive_enter_time = None
            self.setpoint.pose.position = self.goal.pose.position
            self.setpoint.pose.orientation = self.goal.pose.orientation
            self._update_hold_setpoint_locked()
        self._set_nav_state(self.STATE_MOVING)

    def _lio_odom_callback(self, odom: Odometry):
        stamp = odom.header.stamp if odom.header.stamp != rospy.Time(0) else rospy.Time.now()

        vision_msg = PoseStamped()
        vision_msg.header.stamp = stamp
        vision_msg.header.frame_id = odom.header.frame_id if odom.header.frame_id else "lio_world"
        vision_msg.pose = odom.pose.pose
        self.have_lio = True
        self.vision_pub.publish(vision_msg)

    def _arm(self, arm: bool) -> bool:
        req = CommandBoolRequest(value=arm)
        resp = self.arming_srv(req)
        return bool(resp.success)

    def _set_mode(self, mode: str) -> bool:
        req = SetModeRequest(custom_mode=mode)
        resp = self.set_mode_srv(req)
        return bool(resp.mode_sent)

    def spin(self):
        rate = rospy.Rate(self.rate_hz)

        PRESTREAM_SEC = 2.0
        TAKEOFF_REL_ALT = 1.2
        TAKEOFF_TOL = 0.3

        while not rospy.is_shutdown() and not self.current_state.connected:
            print(" -Waiting for FCU connection...")
            rate.sleep()
        rospy.loginfo("[1] FCU connected.")

        while not rospy.is_shutdown() and not self.have_local_pose:
            rospy.loginfo_throttle(2.0, " -Waiting /mavros/local_position/pose ...")
            rate.sleep()
        rospy.loginfo("[2] PX4 local pose received.")

        with self.lock:
            self.setpoint.pose.position = self.local_pose.pose.position
            self.setpoint.pose.orientation = self.local_pose.pose.orientation
            self._update_hold_setpoint_locked()
            rospy.loginfo(f”[3] Initial setpoint set: {self.setpoint.pose.position}”)

        rospy.loginfo(“[4] Pre-streaming setpoint for %.1fs before OFFBOARD ...”, PRESTREAM_SEC)
        t0 = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < PRESTREAM_SEC:
            with self.lock:
                ok = self._apply_hold_setpoint_locked()
                if not ok:
                    self.setpoint.pose.position = self.local_pose.pose.position
                    self.setpoint.pose.orientation = self.local_pose.pose.orientation
                    self._update_hold_setpoint_locked()

                self.setpoint.header.stamp = rospy.Time.now()
                self.setpoint.header.frame_id = self.setpoint_frame_id
                self.setpoint_pub.publish(self.setpoint)
            rate.sleep()

        last_srv_call = rospy.Time(0)
        while not rospy.is_shutdown() and (self.current_state.mode != "OFFBOARD" or not self.current_state.armed):
            with self.lock:
                ok = self._apply_hold_setpoint_locked()
                if not ok:
                    self.setpoint.pose.position = self.local_pose.pose.position
                    self.setpoint.pose.orientation = self.local_pose.pose.orientation
                    self._update_hold_setpoint_locked()
                self.setpoint.header.stamp = rospy.Time.now()
                self.setpoint.header.frame_id = self.setpoint_frame_id
                self.setpoint_pub.publish(self.setpoint)

            if (rospy.Time.now() - last_srv_call).to_sec() >= 1.0:
                if self.current_state.mode != "OFFBOARD":
                    ok = self._set_mode("OFFBOARD")
                    rospy.loginfo("[5] Set OFFBOARD sent=%s current_mode=%s", str(ok), self.current_state.mode)
                if not self.current_state.armed:
                    if not self.have_local_pose:
                        rospy.logerr("Missing PX4 local pose. Refuse to arm.")
                    else:
                        ok = self._arm(True)
                        rospy.loginfo("[6] Arming sent=%s armed=%s", str(ok), str(self.current_state.armed))
                last_srv_call = rospy.Time.now()

            rate.sleep()

        rospy.loginfo("Switched to OFFBOARD and armed. Taking off...")
        self._set_nav_state(self.STATE_TAKEOFF)

        with self.lock:
            x0 = self.local_pose.pose.position.x
            y0 = self.local_pose.pose.position.y
            z0 = self.local_pose.pose.position.z
            self.setpoint.pose.orientation = self.local_pose.pose.orientation
            target_z = z0 + TAKEOFF_REL_ALT
            print(f"Takeoff target altitude: {target_z}")

            self._takeoff_yaw_hold = yaw_from_quat(self.local_pose.pose.orientation)
            qy = quat_from_yaw(self._takeoff_yaw_hold)
            self.setpoint.pose.position.x = x0
            self.setpoint.pose.position.y = y0
            self.setpoint.pose.position.z = target_z
            self.setpoint.pose.orientation.x = qy.x
            self.setpoint.pose.orientation.y = qy.y
            self.setpoint.pose.orientation.z = qy.z
            self.setpoint.pose.orientation.w = qy.w
            self._update_hold_setpoint_locked()
            print(f"Taking off to position: {self.setpoint.pose.position}")

        while not rospy.is_shutdown():
            with self.lock:
                if self.current_state.mode != "OFFBOARD" or (not self.current_state.armed):
                    rospy.logwarn_throttle(2.0, "WARNING: Unexpected state. Suspending control.")
                    self._set_nav_state(self.STATE_FAILSAFE)

                    ok = self._apply_hold_setpoint_locked()
                    if not ok:
                        self.setpoint.pose.position = self.local_pose.pose.position
                        self.setpoint.pose.orientation = self.local_pose.pose.orientation
                        self._update_hold_setpoint_locked()
                else:
                    self._apply_hold_setpoint_locked()
                    cz = self.local_pose.pose.position.z
                    if abs(cz - target_z) <= TAKEOFF_TOL:
                        rospy.loginfo("[7] Reached target altitude. |dz|=%.3f <= %.3f. Enter FOLLOW.",abs(cz - target_z), TAKEOFF_TOL)
                        self._set_nav_state(self.STATE_ARRIVED)
                        break
                self.setpoint.header.stamp = rospy.Time.now()
                self.setpoint.header.frame_id = self.setpoint_frame_id
                self.setpoint_pub.publish(self.setpoint)
            rate.sleep()

        rospy.loginfo("[8] Starting navigation...")
        self._follow_yaw_hold = yaw_from_quat(self.setpoint.pose.orientation)
        nav_step = 0
        while not rospy.is_shutdown():
            with self.lock:
                if self.current_state.mode != "OFFBOARD" or (not self.current_state.armed):
                    rospy.logwarn_throttle(2.0, "WARNING: Unexpected state. Suspending control.")
                    self._set_nav_state(self.STATE_FAILSAFE)
                    ok = self._apply_hold_setpoint_locked()
                    if not ok:
                        self.setpoint.pose.position = self.local_pose.pose.position
                        self.setpoint.pose.orientation = self.local_pose.pose.orientation
                        self._update_hold_setpoint_locked()
                else:
                    if self.have_goal:
                        self._apply_hold_setpoint_locked()

                        self.setpoint.pose.position = self.goal.pose.position
                        self.setpoint.pose.orientation = self.goal.pose.orientation

                        cx = self.local_pose.pose.position.x
                        cy = self.local_pose.pose.position.y
                        cz = self.local_pose.pose.position.z
                        gx = self.goal.pose.position.x
                        gy = self.goal.pose.position.y
                        gz = self.goal.pose.position.z

                        dist_to_goal = math.sqrt((cx - gx)**2 + (cy - gy)**2 + (cz - gz)**2)
                        is_pure_rotation = dist_to_goal < 0.15

                        qg = self.goal.pose.orientation
                        _, _, goal_yaw_rad = tft.euler_from_quaternion([qg.x, qg.y, qg.z, qg.w])
                        goal_yaw_deg = math.degrees(goal_yaw_rad)

                        qc = self.local_pose.pose.orientation
                        _, _, curr_yaw_rad = tft.euler_from_quaternion([qc.x, qc.y, qc.z, qc.w])
                        curr_yaw_deg = math.degrees(curr_yaw_rad)
                        yaw_err_deg = abs(wrap_deg180(goal_yaw_deg - curr_yaw_deg))

                        if is_pure_rotation:
                            elapsed = (rospy.Time.now() - self.goal.header.stamp).to_sec()
                            if elapsed < 1.6:
                                arrived_now = False
                            else:
                                arrived_now = (yaw_err_deg <= self.yaw_tol_deg)
                                if arrived_now and self._yaw_arrive_enter_time is None:
                                    self._yaw_arrive_enter_time = rospy.Time.now()
                                if arrived_now:
                                    hold = (rospy.Time.now() - self._yaw_arrive_enter_time).to_sec()
                                    arrived_now = hold >= self.yaw_hold_sec

                            rospy.loginfo_throttle(0.5, f"[YAW-ONLY] elapsed={elapsed:.1f}s yaw_err={yaw_err_deg:.1f}° tol={self.yaw_tol_deg}°")
                        else:
                            if self.have_local_odom:
                                speed = math.hypot(self.vel_enu[0], self.vel_enu[1])
                                arrived_now = (dist_to_goal <= self.arrive_tol) and (speed <= self.arrive_speed_tol)
                            else:
                                arrived_now = (dist_to_goal <= self.arrive_tol)
                            self._yaw_arrive_enter_time = None

                        if arrived_now:
                            if self._arrive_enter_time is None:
                                self._arrive_enter_time = rospy.Time.now()
                            hold = (rospy.Time.now() - self._arrive_enter_time).to_sec()
                            if hold >= self.arrive_hold_sec:
                                mode = "YAW-ONLY" if is_pure_rotation else "POSITION"
                                rospy.loginfo(f"ARRIVED ({mode}): dist={dist_to_goal:.3f}m yaw_err={yaw_err_deg:.1f}° hold={hold:.2f}s")
                                self._set_nav_state(self.STATE_ARRIVED)
                                self.have_goal = False
                                self._arrive_enter_time = None
                                self._yaw_arrive_enter_time = None
                        else:
                            self._arrive_enter_time = None
                            self._yaw_arrive_enter_time = None
                            self._set_nav_state(self.STATE_MOVING)

                    else:
                        self._arrive_enter_time = None
                        self._set_nav_state(self.STATE_ARRIVED)

                        ok = self._apply_hold_setpoint_locked()
                        if not ok:
                            self.setpoint.pose.position = self.local_pose.pose.position
                            qy = quat_from_yaw(self._follow_yaw_hold)
                            self.setpoint.pose.orientation.x = qy.x
                            self.setpoint.pose.orientation.y = qy.y
                            self.setpoint.pose.orientation.z = qy.z
                            self.setpoint.pose.orientation.w = qy.w

                self.setpoint.header.stamp = rospy.Time.now()
                self.setpoint.header.frame_id = self.setpoint_frame_id
                self.setpoint_pub.publish(self.setpoint)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("model_set_node", anonymous=False)
    node = ModelSetNode()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
