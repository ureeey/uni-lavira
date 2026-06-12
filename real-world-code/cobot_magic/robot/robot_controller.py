"""Cobot Magic hardware + motion controller.

Ports the hardware and motion layer of the original ``ImageNavObjectController``
(camera / odom I/O, dual-arm panorama scan, base rotation, depth lookup, and the
dynamic-replan navigation loop). The VLM / strategic-planning logic and the
per-cycle ObjectNav loop do NOT live here; those belong to the task layer.

Distinctive motion model
------------------------
Cobot Magic navigates with a *continuous dynamic-replan* control loop
(``navigate_dynamic``): a ~20 Hz tracker drives ``/cmd_vel`` while a background
thread re-runs the in-process iPlanner every ``Config.REPLAN_INTERVAL`` seconds.
This is deliberately different from the discrete rotate-then-move scheme used by
the quadruped / humanoid stacks and is preserved verbatim here.

Import safety
-------------
``rospy``, the ROS message types, ``cv2``, ``tf``, and the iPlanner agent are
imported lazily inside ``__init__`` / methods so ``import robot.robot_controller``
succeeds on a machine without ROS / torch / a GPU. ``utils`` and ``config`` are
imported at top level (they are ROS-free and torch-free).

Paths
-----
The iPlanner config / checkpoint paths and the output root come from ``Config``;
this module ships no absolute paths.
"""
from __future__ import annotations

import math
import os
import threading
import time
import traceback
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from config import Config
from utils import (
    TrajectoryFollower,
    pixel_to_robot_goal,
    print_info,
    print_success,
    print_warning,
    save_debug_image,
)
from robot.arm_controller import AutomatedController
from robot.iplanner_client import IPlannerClient


class RobotController:
    """Real-hardware controller for the Cobot Magic dual-arm mobile base."""

    def __init__(self) -> None:
        # Lazy ROS imports so this module is importable without ROS installed.
        import rospy
        from cv_bridge import CvBridge
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import (
            CameraInfo,
            Image as ROSImage,
            JointState,  # noqa: F401 (imported for ROS type registration parity)
        )
        from nav_msgs.msg import Odometry

        self._Twist = Twist

        self.bridge = CvBridge()

        # --- Robot state ---
        self.current_front_rgb: Optional[np.ndarray] = None
        self.current_front_depth: Optional[np.ndarray] = None
        self.current_left_rgb: Optional[np.ndarray] = None
        self.current_right_rgb: Optional[np.ndarray] = None
        self.image_lock = threading.Lock()
        self.odom_lock = threading.Lock()
        self.robot_pose: Optional[Tuple[float, float, float]] = None

        # --- Camera geometry (defaults from Config until camera_info arrives) ---
        self.rgb_width = Config.RGB_WIDTH
        self.rgb_height = Config.RGB_HEIGHT
        self.depth_width = Config.DEPTH_WIDTH
        self.depth_height = Config.DEPTH_HEIGHT
        self.depth_fx: Optional[float] = Config.DEPTH_FX
        self.depth_fy: Optional[float] = None
        self.depth_cx: Optional[float] = Config.DEPTH_CX
        self.depth_cy: Optional[float] = Config.DEPTH_CY

        # --- ROS pub / subs ---
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        all_topics = [t[0] for t in rospy.get_published_topics()]
        self.has_left_camera = "/camera_l/color/image_raw" in all_topics
        self.has_right_camera = "/camera_r/color/image_raw" in all_topics
        print_info(
            f"Camera Status: Left={self.has_left_camera}, "
            f"Right={self.has_right_camera}"
        )

        rospy.Subscriber(
            "/camera/color/image_raw", ROSImage, self.front_rgb_callback
        )
        rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            ROSImage,
            self.front_depth_callback,
        )
        rospy.Subscriber(
            "/camera/aligned_depth_to_color/camera_info",
            CameraInfo,
            self.camera_info_callback,
        )
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        if self.has_left_camera:
            rospy.Subscriber(
                "/camera_l/color/image_raw", ROSImage, self.left_rgb_callback
            )
        if self.has_right_camera:
            rospy.Subscriber(
                "/camera_r/color/image_raw", ROSImage, self.right_rgb_callback
            )

        rospy.sleep(2)

        # --- IPlanner (hardware/inference init) ---
        self.initialize_models()

        # --- Motion / observation history ---
        self.obs_horizon = 4
        self.obs_history: deque = deque(maxlen=self.obs_horizon)

        # --- Arm gimbal (panorama camera sweep) ---
        self.arm_controller = AutomatedController()
        self.TARGET_SEQUENCE = Config.TARGET_SEQUENCE
        self.ZERO_POSITION = Config.ZERO_POSITION

        # --- Planning state ---
        self.planning_lock = threading.Lock()
        self.next_path_global: Optional[np.ndarray] = None
        self.current_path_global: np.ndarray = np.zeros((0, 2))
        self.is_replanning = False
        self.global_goal: Optional[np.ndarray] = None
        self.current_cycle_dir: Optional[str] = None

        # --- Recording ---
        self.recording_thread: Optional[threading.Thread] = None
        self.is_recording = False

    # ------------------------------------------------------------------ #
    # Model / planner initialization
    # ------------------------------------------------------------------ #
    def initialize_models(self) -> None:
        """Initialize the in-process iPlanner agent (hardware/inference only).

        The VLM client is NOT initialized here; that is the task layer's job
        (via the ai_client package).
        """
        self.planner = IPlannerClient()

    # ------------------------------------------------------------------ #
    # IPlanner replan thread
    # ------------------------------------------------------------------ #
    def _replan_thread_func(
        self,
        obs_list: List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]],
        global_goal: np.ndarray,
        target_pixel_v: Optional[float] = None,
    ) -> None:
        """Background replan: depth + goal -> robot-frame traj -> global path.

        Mirrors the original ``_replan_thread_func``: the latest observation's
        depth and the goal (transformed into that observation's robot frame) are
        fed to the in-process iPlanner; the returned robot-frame trajectory is
        transformed back to the global frame and stored for the control loop.
        """
        path_global: Optional[np.ndarray] = None
        try:
            if not obs_list:
                return
            rgb, depth_raw, obs_pose = obs_list[-1]

            local_goal_x, local_goal_y = IPlannerClient.local_goal_from_global(
                (float(global_goal[0]), float(global_goal[1])), obs_pose
            )

            local_path = self.planner.plan(depth_raw, (local_goal_x, local_goal_y))
            if local_path:
                path_global = IPlannerClient.path_to_global(local_path, obs_pose)

                print_info(
                    f"IPlanner: {len(path_global)} waypoints. "
                    f"Goal=({global_goal[0]:.2f},{global_goal[1]:.2f})"
                )

                if self.current_cycle_dir:
                    save_debug_image(
                        rgb,
                        local_path,
                        [local_goal_x, local_goal_y],
                        target_pixel_v,
                        self.current_cycle_dir,
                        intrinsics=Config.NAVDP_INTRINSICS,
                    )
        except Exception as exc:  # noqa: BLE001 - replan must not crash nav
            from utils import print_error

            print_error(f"Replan error: {exc}\n{traceback.format_exc()}")
        finally:
            if path_global is not None and len(path_global) > 0:
                with self.planning_lock:
                    self.next_path_global = path_global
            self.is_replanning = False

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def start_recording(self, save_dir: str) -> None:
        if self.is_recording:
            return
        self.is_recording = True
        self.recording_thread = threading.Thread(
            target=self._record_images_thread, args=(save_dir,), daemon=True
        )
        self.recording_thread.start()

    def stop_recording(self) -> None:
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()

    def _record_images_thread(self, save_dir: str) -> None:
        import cv2
        import rospy

        front_view_dir = os.path.join(save_dir, "front_view")
        os.makedirs(front_view_dir, exist_ok=True)
        print_info(f"Recording to {front_view_dir} at 10 Hz")
        rate = rospy.Rate(10)
        while self.is_recording and not rospy.is_shutdown():
            try:
                with self.image_lock:
                    if self.current_front_rgb is not None:
                        img = self.current_front_rgb.copy()
                    else:
                        img = None
                if img is not None:
                    cv2.imwrite(
                        os.path.join(front_view_dir, f"img_{time.time():.3f}.jpg"),
                        img,
                    )
            except Exception as exc:  # noqa: BLE001 - recording is best-effort
                print_warning(f"Recording error: {exc}")
            rate.sleep()
        print_info("Recording stopped")

    # ------------------------------------------------------------------ #
    # Camera / odom callbacks
    # ------------------------------------------------------------------ #
    def front_rgb_callback(self, msg) -> None:
        with self.image_lock:
            self.current_front_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def front_depth_callback(self, msg) -> None:
        with self.image_lock:
            d = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.current_front_depth = (
                d.astype(np.float32) / 1000.0 if d.dtype != np.float32 else d
            )

    def left_rgb_callback(self, msg) -> None:
        with self.image_lock:
            self.current_left_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def right_rgb_callback(self, msg) -> None:
        with self.image_lock:
            self.current_right_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def camera_info_callback(self, msg) -> None:
        with self.image_lock:
            self.depth_fx, self.depth_fy = msg.K[0], msg.K[4]
            self.depth_cx, self.depth_cy = msg.K[2], msg.K[5]

    def odom_callback(self, msg) -> None:
        from tf.transformations import euler_from_quaternion

        with self.odom_lock:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            self.robot_pose = (p.x, p.y, yaw)

    # ------------------------------------------------------------------ #
    # Depth / geometry helpers
    # ------------------------------------------------------------------ #
    def pixel_to_robot_goal(
        self,
        u: float,
        v: float,
        depth_image: Optional[np.ndarray] = None,
        window_size: int = 5,
    ) -> Tuple[float, float]:
        """Project a pixel + depth into a robot-frame ``(Z, -x_cam)`` goal.

        Uses the live front depth frame and current depth intrinsics when
        ``depth_image`` is not supplied. Delegates to ``utils.pixel_to_robot_goal``.
        """
        if depth_image is None:
            with self.image_lock:
                if self.current_front_depth is None:
                    return 1.0, 0.0
                depth_image = self.current_front_depth
        return pixel_to_robot_goal(
            u, v, depth_image, self.depth_cx, self.depth_fx, window_size=window_size
        )

    def get_depth_value(
        self,
        u: float,
        v: float,
        depth_image: Optional[np.ndarray] = None,
        window_size: int = 5,
    ) -> float:
        """Return the median valid forward depth (metres) around pixel ``(u, v)``."""
        if depth_image is None:
            with self.image_lock:
                if self.current_front_depth is None:
                    return 1.0
                depth_image = self.current_front_depth
        h, w = depth_image.shape
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        patch = depth_image[
            max(0, v - window_size):min(h, v + window_size),
            max(0, u - window_size):min(w, u + window_size),
        ]
        valid = patch[patch > 0.1]
        return float(np.median(valid)) if len(valid) > 0 else 1.0

    def get_front_image_jpeg(self) -> Optional[bytes]:
        """Encode the latest front RGB frame as JPEG bytes (for the web feed).

        Returns ``None`` when no front frame is available yet or encoding fails.
        """
        import cv2

        with self.image_lock:
            img = self.current_front_rgb
            img = img.copy() if img is not None else None
        if img is None:
            return None
        try:
            success, buf = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            )
            if success:
                return buf.tobytes()
        except Exception as exc:  # noqa: BLE001 - encoding is best-effort
            print_warning(f"Failed to encode front image to JPEG: {exc}")
        return None

    # ------------------------------------------------------------------ #
    # Motion: rotation
    # ------------------------------------------------------------------ #
    def rotate_angle(self, angle_deg: float) -> None:
        """Rotate the base in place by ``angle_deg`` degrees.

        Applies ``Config.ROTATE_COMPENSATION`` overshoot correction and rotates
        at ``Config.ROTATE_SPEED`` deg/s, publishing ``/cmd_vel`` at 20 Hz.
        """
        import rospy

        if abs(angle_deg) < 1.0:
            return
        if angle_deg > 180:
            angle_deg -= 360
        if angle_deg < -180:
            angle_deg += 360

        compensation = Config.ROTATE_COMPENSATION
        target = angle_deg * compensation
        speed = Config.ROTATE_SPEED  # deg/s
        print_info(f"Rotating {angle_deg} deg (target {target:.1f} deg)")
        twist = self._Twist()
        twist.angular.z = (
            math.radians(speed) if target > 0 else -math.radians(speed)
        )
        duration = abs(target / speed)
        rate = rospy.Rate(20)
        t0 = rospy.Time.now().to_sec()
        while (
            rospy.Time.now().to_sec() - t0 < duration and not rospy.is_shutdown()
        ):
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        self.cmd_vel_pub.publish(self._Twist())

    # ------------------------------------------------------------------ #
    # Observation: arm-swept panorama
    # ------------------------------------------------------------------ #
    def collect_panoramic_images(
        self,
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]], threading.Thread]:
        """Sweep the arm gimbal and capture an 8-slot RGB panorama.

        Captures the front view, then moves the arms through
        ``Config.TARGET_SEQUENCE`` and captures left / right side views per the
        ``seq_map`` (left -> slots 1,2,3; right -> slots 7,6,5). Slot 0 is front;
        slot 4 (back) is never filled.

        Returns ``(pan_rgb, pan_depth, arm_reset_thread)`` where ``pan_rgb`` /
        ``pan_depth`` are 8-element lists (``None`` for unfilled slots) and
        ``arm_reset_thread`` is an unstarted daemon thread that returns the arms
        to the first sequence pose. The caller must call ``.start()`` on it (this
        parallel-reset pattern is preserved from the original).
        """
        import rospy

        print_info("Scanning panorama (RGB)...")
        pan_rgb: List[Optional[np.ndarray]] = [None] * 8
        pan_depth: List[Optional[np.ndarray]] = [None] * 8

        wait_start = time.time()
        while time.time() - wait_start < 5.0:
            with self.image_lock:
                front_ok = self.current_front_rgb is not None
                left_ok = (not self.has_left_camera) or self.current_left_rgb is not None
                right_ok = (
                    not self.has_right_camera
                ) or self.current_right_rgb is not None
                if front_ok and left_ok and right_ok:
                    break
            rospy.sleep(0.1)

        with self.image_lock:
            if self.current_front_rgb is not None:
                pan_rgb[0] = self.current_front_rgb.copy()
                pan_depth[0] = (
                    self.current_front_depth.copy()
                    if self.current_front_depth is not None
                    else None
                )

        seq_map = [{"l": 1, "r": 7}, {"l": 2, "r": 6}, {"l": 3, "r": 5}]
        for i, pos in enumerate(self.TARGET_SEQUENCE):
            if rospy.is_shutdown():
                break
            self.arm_controller.move_to_goal(pos["left"], pos["right"])
            rospy.sleep(2.0)
            with self.image_lock:
                l_idx, r_idx = seq_map[i]["l"], seq_map[i]["r"]
                if self.has_left_camera and self.current_left_rgb is not None:
                    pan_rgb[l_idx] = self.current_left_rgb.copy()
                if self.has_right_camera and self.current_right_rgb is not None:
                    pan_rgb[r_idx] = self.current_right_rgb.copy()

        home = self.TARGET_SEQUENCE[0]
        arm_reset_thread = threading.Thread(
            target=self.arm_controller.move_to_goal,
            args=(home["left"], home["right"]),
            daemon=True,
        )
        return pan_rgb, pan_depth, arm_reset_thread

    # ------------------------------------------------------------------ #
    # Motion: dynamic-replan navigation
    # ------------------------------------------------------------------ #
    def navigate_dynamic(
        self,
        target_u: float,
        target_v: float,
        cycle_dir: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Drive toward a pixel goal with continuous iPlanner replanning.

        Projects ``(target_u, target_v)`` into a global goal, then runs a ~20 Hz
        control loop: every ``Config.REPLAN_INTERVAL`` seconds a background
        thread replans via the in-process iPlanner; the latest path is tracked
        with a ``TrajectoryFollower``. Stops when within
        ``Config.ARRIVAL_THRESHOLD`` metres of the goal, on
        ``Config.NAV_TIMEOUT`` (overridable via ``timeout``), or on ROS shutdown.
        Applies a terminal slowdown inside 1.2 m. Returns ``True`` on arrival.
        """
        import rospy

        if timeout is None:
            timeout = Config.NAV_TIMEOUT

        print_info(f"Dynamic Navigation to pixel ({target_u}, {target_v})")

        wait_start = time.time()
        while self.robot_pose is None:
            if time.time() - wait_start > 5.0:
                print_warning("Odometry timeout — using (0,0,0)")
                with self.odom_lock:
                    self.robot_pose = (0.0, 0.0, 0.0)
                break
            rospy.sleep(0.1)

        with self.odom_lock:
            start_pose = self.robot_pose
        sx, sy, syaw = start_pose

        gx_local, gy_local = self.pixel_to_robot_goal(target_u, target_v)
        cos_s, sin_s = math.cos(syaw), math.sin(syaw)
        global_goal_x = sx + gx_local * cos_s - gy_local * sin_s
        global_goal_y = sy + gx_local * sin_s + gy_local * cos_s
        self.global_goal = np.array([global_goal_x, global_goal_y])

        self.current_path_global = np.array(
            [[sx, sy], [global_goal_x, global_goal_y]]
        )
        self.next_path_global = None
        self.is_replanning = False
        self.current_cycle_dir = cycle_dir

        tracker = TrajectoryFollower()
        rate = rospy.Rate(20)
        start_time = time.time()
        last_replan_time = 0.0
        replan_interval = Config.REPLAN_INTERVAL
        arrival_threshold = Config.ARRIVAL_THRESHOLD
        success = False

        while not rospy.is_shutdown():
            now = time.time()
            if now - start_time > timeout:
                print_warning(f"Navigation timeout ({timeout}s)")
                break

            with self.odom_lock:
                cx, cy, cyaw = self.robot_pose

            dist = math.hypot(self.global_goal[0] - cx, self.global_goal[1] - cy)
            if dist < arrival_threshold and dist > 0.01:
                print_success(f"Arrived (dist={dist:.2f}m)")
                success = True
                break

            with self.image_lock:
                cur_rgb = self.current_front_rgb
                cur_depth = self.current_front_depth

            if cur_rgb is not None and cur_depth is not None:
                self.obs_history.append((cur_rgb, cur_depth, (cx, cy, cyaw)))

            if (
                now - last_replan_time > replan_interval
                and not self.is_replanning
                and len(self.obs_history) >= 2
            ):
                obs_snap = list(self.obs_history)
                self.is_replanning = True
                last_replan_time = now
                t = threading.Thread(
                    target=self._replan_thread_func,
                    args=(obs_snap, self.global_goal.copy(), target_v),
                    daemon=True,
                )
                t.start()

            with self.planning_lock:
                if self.next_path_global is not None:
                    self.current_path_global = self.next_path_global
                    self.next_path_global = None

            cos_c, sin_c = math.cos(cyaw), math.sin(cyaw)
            local_path = []
            for pt in self.current_path_global:
                dx = pt[0] - cx
                dy = pt[1] - cy
                local_path.append(
                    [dx * cos_c + dy * sin_c, -dx * sin_c + dy * cos_c]
                )

            v, w = tracker.compute_velocity(np.array(local_path))
            if dist < 1.2:
                v = max(v * (dist / 1.2), 0.15)

            twist = self._Twist()
            twist.linear.x = v
            twist.angular.z = w
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        for _ in range(5):
            self.cmd_vel_pub.publish(self._Twist())
            rospy.sleep(0.05)
        return success

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        print_info("Shutting down.")
        self.cmd_vel_pub.publish(self._Twist())
        self.arm_controller.stop_control_thread()
