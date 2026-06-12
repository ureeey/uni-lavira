#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indoor Navigation with NavDP for Drone
========================================
Implements LaViRA-style navigation adapted for drone platform:
- 360-degree panorama capture with RealSense D435i
- Strategic decision with Gemini-2.5-pro
- Tactical bbox detection with Qwen2.5-VL-32B
- Trajectory planning with NavDP
- ENU coordinate system for waypoint publishing
"""

import os
import re
import io
import json
import math
import time
import base64
import threading
import yaml
from collections import deque
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import requests
from PIL import Image
from openai import OpenAI

import rospy
import tf.transformations as tft
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import UInt8
from cv_bridge import CvBridge

from vln_core.logger import logger


# ============================================================================
# Helper Functions
# ============================================================================

def wrap_deg180(a: float) -> float:
    """Wrap angle to [-180, 180) degrees"""
    return (a + 180.0) % 360.0 - 180.0


def quat_to_yaw_enu_deg(qx, qy, qz, qw) -> float:
    """Extract yaw from quaternion in ENU frame (degrees)"""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return wrap_deg180(math.degrees(yaw))


def numpy_to_base64(img_np: np.ndarray) -> str:
    """Convert numpy BGR image to base64 string"""
    if img_np is None:
        return ""
    _, buffer = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def img_to_base64(img_path: str) -> str:
    """Read image file and convert to base64"""
    if not os.path.exists(img_path):
        logger.error(f"Image file not found: {img_path}")
        return ""
    try:
        img = Image.open(img_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to convert image to base64: {e}")
        return ""


def safe_json_loads(text: str) -> Dict:
    """Safely parse JSON from LLM output with error handling"""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```', '', text)

    # Extract JSON object
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    logger.error(f"Failed to parse JSON from: {text[:200]}")
    return {}


# ============================================================================
# OpenAI Vision Client (Dual Model Support)
# ============================================================================

class OpenAIVisionClient:
    """OpenAI API client supporting primary and secondary models"""

    def __init__(self, api_key=None, base_url=None, model_name="gpt-4-vision-preview",
                 secondary_model_name=None, secondary_api_key=None, secondary_base_url=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        if secondary_model_name:
            self.secondary_client = OpenAI(
                api_key=secondary_api_key or api_key,
                base_url=secondary_base_url or base_url
            )
            self.secondary_model_name = secondary_model_name
        else:
            self.secondary_client = None
            self.secondary_model_name = None

        self.stats = {
            'primary': {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
            'secondary': {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        }

    def generate(self, messages, max_new_tokens=1024, temperature=0.7, use_secondary=False, **kwargs):
        """Generate response from API"""
        client = self.secondary_client if (use_secondary and self.secondary_client) else self.client
        model_name = self.secondary_model_name if (use_secondary and self.secondary_client) else self.model_name
        stats_key = 'secondary' if use_secondary else 'primary'

        logger.info(f"{'='*20} {model_name} start {'='*20}")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature
            )

            self.stats[stats_key]['calls'] += 1
            if hasattr(response, 'usage') and response.usage:
                self.stats[stats_key]['input_tokens'] += response.usage.prompt_tokens or 0
                self.stats[stats_key]['output_tokens'] += response.usage.completion_tokens or 0
                self.stats[stats_key]['total_tokens'] += response.usage.total_tokens or 0
                logger.info(f"API usage - Input: {response.usage.prompt_tokens}, "
                           f"Output: {response.usage.completion_tokens}")

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"API call error ({model_name}): {e}")
            logger.info("Retrying after 30s...")
            time.sleep(30)
            return self.generate(messages, max_new_tokens, temperature, use_secondary, **kwargs)

    def generate_with_primary(self, messages, max_new_tokens=1024, temperature=0.7, **kwargs):
        """Use primary model"""
        return self.generate(messages, max_new_tokens, temperature, use_secondary=False, **kwargs)

    def generate_with_secondary(self, messages, max_new_tokens=1024, temperature=0.7, **kwargs):
        """Use secondary model"""
        if not self.secondary_client:
            logger.warning("Secondary model not configured, using primary")
            return self.generate_with_primary(messages, max_new_tokens, temperature, **kwargs)
        return self.generate(messages, max_new_tokens, temperature, use_secondary=True, **kwargs)


# ============================================================================
# NavDP Client for Trajectory Planning
# ============================================================================

class NavDPClient:
    """Client for NavDP trajectory planning server"""

    def __init__(self, server_url="http://localhost:8888"):
        self.server_url = server_url
        self.session = requests.Session()
        self.initialized = False

    def reset(self, intrinsic, batch_size=1, stop_threshold=-3.0):
        """Initialize NavDP server with camera intrinsics"""
        url = f"{self.server_url}/navigator_reset"
        payload = {
            "intrinsic": intrinsic.tolist() if isinstance(intrinsic, np.ndarray) else intrinsic,
            "stop_threshold": stop_threshold,
            "batch_size": batch_size
        }
        try:
            resp = self.session.post(url, json=payload, timeout=40)
            if resp.status_code == 200:
                logger.info("[NavDP] Server connected successfully")
                self.initialized = True
                return resp.json().get("algo", "navdp")
            else:
                logger.error(f"[NavDP] Init failed: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"[NavDP] Cannot connect to server: {e}")
            return None

    def pointgoal_step(self, point_goals, rgb_images, depth_images, intrinsic=None):
        """
        Plan trajectory to point goal
        Args:
            point_goals: numpy array (N, 2) - goal positions in robot frame
            rgb_images: list of numpy arrays (H, W, 3)
            depth_images: list of numpy arrays (H, W)
            intrinsic: camera intrinsic matrix (3x3), optional
        Returns:
            trajectory: (N, 3) optimal trajectory points
            all_trajectories: (M, N, 3) all candidate trajectories
            all_values: (M,) trajectory scores
        """
        if not self.initialized:
            # Use provided intrinsic or default to identity
            init_intrinsic = intrinsic if intrinsic is not None else np.eye(3)
            if not self.reset(init_intrinsic):
                return None, None, None

        url = f"{self.server_url}/pointgoal_step"

        # Concatenate images
        concat_images = np.concatenate([img for img in rgb_images], axis=0)
        concat_depths = np.concatenate([img for img in depth_images], axis=0)

        # Encode RGB
        _, rgb_encoded = cv2.imencode('.jpg', concat_images)
        image_bytes = io.BytesIO()
        image_bytes.write(rgb_encoded)

        # Encode depth (meters -> 0.1mm units)
        depth_scaled = np.clip(concat_depths * 10000.0, 0, 65535.0).astype(np.uint16)
        _, depth_encoded = cv2.imencode('.png', depth_scaled)
        depth_bytes = io.BytesIO()
        depth_bytes.write(depth_encoded)

        files = {
            'image': ('image.jpg', image_bytes.getvalue(), 'image/jpeg'),
            'depth': ('depth.png', depth_bytes.getvalue(), 'image/png'),
        }
        data = {
            'goal_data': json.dumps({
                'goal_x': point_goals[:, 0].tolist(),
                'goal_y': point_goals[:, 1].tolist()
            }),
            'depth_time': time.time(),
            'rgb_time': time.time(),
        }

        try:
            resp = self.session.post(url, files=files, data=data, timeout=40)
            if resp.status_code == 200:
                result = resp.json()
                trajectory = np.array(result['trajectory'])
                all_trajectory = np.array(result['all_trajectory'])
                all_values = np.array(result['all_values'])
                return trajectory, all_trajectory, all_values
            else:
                logger.error(f"[NavDP] Planning failed: {resp.status_code} - {resp.text}")
                return None, None, None
        except Exception as e:
            logger.error(f"[NavDP] Network error: {e}")
            return None, None, None


# ============================================================================
# Indoor Navigation Evaluator
# ============================================================================

class IndoorNavEvaluator:
    """
    Indoor navigation evaluator for drone using:
    - RealSense D435i for RGB-D sensing
    - 360-degree panorama via drone rotation
    - Gemini-2.5-pro for strategic decisions
    - Qwen2.5-VL-32B for tactical bbox detection
    - NavDP for trajectory planning
    """

    # Nav state constants (sync with model_set_node)
    STATE_IDLE = 0
    STATE_TAKEOFF = 1
    STATE_MOVING = 2
    STATE_ARRIVED = 3
    STATE_FAILSAFE = 4

    def __init__(self):
        # Thread safety
        self.lock = threading.RLock()
        self.bridge = CvBridge()

        # State flags
        self.have_rgb = False
        self.have_depth = False
        self.have_local_pose = False
        self.have_nav_state = False

        # Current sensor data
        self.rgb_image = None
        self.depth_image = None
        self.position_enu = np.zeros(3, dtype=np.float64)
        self.yaw_enu_deg = 0.0

        # Camera intrinsics (RealSense D435i - from camera_info topic)
        # K: [377.8726806640625, 0.0, 321.0171813964844,
        #     0.0, 377.8726806640625, 236.71241760253906,
        #     0.0, 0.0, 1.0]
        self.fx = 377.8726806640625
        self.fy = 377.8726806640625
        self.cx = 321.0171813964844
        self.cy = 236.71241760253906
        self.camera_intrinsic = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ])

        # Image dimensions
        self.image_width = 640
        self.image_height = 480

        # Panorama storage (4 directions: front, right, back, left)
        self.panorama_rgb = [None, None, None, None]
        self.panorama_depth = [None, None, None, None]
        self.panorama_captured = False
        self.panorama_origin_pos = np.zeros(3, dtype=np.float64)
        self.panorama_origin_yaw_deg = 0.0

        # Navigation state
        self.nav_state = None
        self._arrival_seq = 0
        self._last_inferred_arrival_seq = -1
        self.nav_done = False

        # Task instruction
        self.instruction_yaml_path = rospy.get_param(
            "~instruction_yaml",
            "$(find vln_node)/config/instruction.yaml"
        )
        self.instruction_text = ""
        self._load_instruction()

        self.log_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../log")
        self.log_txt_path = os.path.join(self.log_root, "llm_reply.txt")
        self.log_bbox_dir = os.path.join(self.log_root, "bbox")
        self.log_traj_dir = os.path.join(self.log_root, "trajectory")
        self._init_log_files()

        # API clients
        self._init_api_clients()

        # NavDP client
        navdp_port = rospy.get_param("~navdp_port", 8888)
        self.navdp_client = NavDPClient(f"http://localhost:{navdp_port}")

        # ROS subscribers
        self.rgb_sub = rospy.Subscriber(
            "/camera/color/image_raw", ImageMsg,
            self.rgb_callback, queue_size=1
        )
        self.depth_sub = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", ImageMsg,
            self.depth_callback, queue_size=1
        )
        self.pose_sub = rospy.Subscriber(
            "/mavros/local_position/pose", PoseStamped,
            self.pose_callback, queue_size=10
        )
        self.nav_state_sub = rospy.Subscriber(
            "/unilavira/nav_state", UInt8,
            self.nav_state_callback, queue_size=10
        )

        # ROS publishers
        self.waypoint_pub = rospy.Publisher(
            '/unilavira/waypoint', PoseStamped,
            queue_size=10, latch=True
        )

        # Navigation history
        self.current_step = 0
        self.N = 10
        self.todo_list = ""
        self.history_info = []

        # RGB history (recorded at 1Hz during moving phase)
        self.rgb_history = []
        self.record_rgb_history = False
        self.last_rgb_record_time = 0.0

        logger.info("IndoorNavEvaluator initialized")

    def _load_instruction(self):
        """Load navigation instruction from YAML file"""
        try:
            with open(self.instruction_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.instruction_text = data.get('instruction', '')
                logger.info(f"Loaded instruction: {self.instruction_text}")
        except Exception as e:
            logger.error(f"Failed to load instruction: {e}")
            self.instruction_text = "Navigate to the target location"

    def _init_log_files(self):
        """Initialize log files and image save directories."""
        os.makedirs(self.log_root, exist_ok=True)
        os.makedirs(self.log_bbox_dir, exist_ok=True)
        os.makedirs(self.log_traj_dir, exist_ok=True)

        # Overwrite old log on each launch
        with open(self.log_txt_path, "w", encoding="utf-8") as f:
            f.write("instruction\n")
            f.write(f"{self.instruction_text}\n\n")


    def _append_text_log(self, text: str):
        """Append text to llm_reply.txt."""
        with open(self.log_txt_path, "a", encoding="utf-8") as f:
            f.write(text)


    def _append_step_header(self):
        """Write step header to log."""
        self._append_text_log(f"######## step-{self.current_step} ########\n")


    def _append_model_reply(self, tag: str, reply_text: str):
        """Write a model reply to log."""
        self._append_text_log(f"[{tag}]\n")
        self._append_text_log(f"{reply_text}\n\n")


    def _append_navdp_waypoint_log(self, waypoint_enu, total_points: int, target_idx: int):
        """Log the executed NavDP waypoint."""
        self._append_text_log("[navdp_executed_waypoint]\n")
        self._append_text_log(f"N = {self.N}\n")
        self._append_text_log(f"selected_index = {target_idx}\n")
        self._append_text_log(f"total_points = {total_points}\n")
        self._append_text_log(
            f"waypoint_enu = [{waypoint_enu[0]:.6f}, {waypoint_enu[1]:.6f}, {waypoint_enu[2]:.6f}]\n\n"
        )

    def _init_api_clients(self):
        """Initialize OpenAI API clients for Gemini and Qwen"""
        # Primary: Qwen2.5-VL-32B for tactical bbox detection
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
        model_name = "Qwen/Qwen2.5-VL-32B-Instruct"

        # Secondary: Gemini-2.5-pro for strategic decisions
        secondary_api_key = os.environ.get("OPENAI_API_KEY_SECONDARY", "")
        secondary_base_url = os.environ.get(
            "OPENAI_BASE_URL_SECONDARY",
            "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        secondary_model_name = "gemini-2.5-pro"

        self.model = OpenAIVisionClient(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            secondary_model_name=secondary_model_name,
            secondary_api_key=secondary_api_key,
            secondary_base_url=secondary_base_url
        )

    # ========================================================================
    # ROS Callbacks
    # ========================================================================

    def rgb_callback(self, msg: ImageMsg):
        """Callback for RealSense RGB image"""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.rgb_image = cv_img
                self.have_rgb = True

            # Append to rgb_history at 1Hz if recording is enabled
            self._maybe_append_rgb_history(cv_img)

        except Exception as e:
            logger.error(f"RGB callback error: {e}")

    def depth_callback(self, msg: ImageMsg):
        """Callback for RealSense aligned depth image"""
        try:
            # Depth is in mm (uint16), convert to meters
            depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            depth_m = depth_mm.astype(np.float32) / 1000.0
            with self.lock:
                self.depth_image = depth_m
                self.have_depth = True
        except Exception as e:
            logger.error(f"Depth callback error: {e}")

    def pose_callback(self, msg: PoseStamped):
        """Callback for drone local pose (ENU frame)"""
        with self.lock:
            self.position_enu[0] = msg.pose.position.x
            self.position_enu[1] = msg.pose.position.y
            self.position_enu[2] = msg.pose.position.z
            q = msg.pose.orientation
            self.yaw_enu_deg = quat_to_yaw_enu_deg(q.x, q.y, q.z, q.w)
            self.have_local_pose = True

    def nav_state_callback(self, msg: UInt8):
        """Callback for navigation state from model_set_node"""
        with self.lock:
            prev_state = self.nav_state
            self.nav_state = int(msg.data)
            self.have_nav_state = True

            # Detect transition to ARRIVED
            if prev_state != self.STATE_ARRIVED and self.nav_state == self.STATE_ARRIVED:
                self._arrival_seq += 1
                logger.info(f"Nav state -> ARRIVED (seq={self._arrival_seq})")
                # Stop RGB recording on arrival
                self._stop_rgb_history_recording()

    def _start_rgb_history_recording(self):
        """Start recording RGB history at 1Hz during moving phase."""
        with self.lock:
            self.record_rgb_history = True
            self.last_rgb_record_time = 0.0
        logger.info(f"[RGB_HISTORY] Start recording for step {self.current_step}")


    def _stop_rgb_history_recording(self):
        """Stop recording RGB history."""
        with self.lock:
            self.record_rgb_history = False
        logger.info(f"[RGB_HISTORY] Stop recording for step {self.current_step}, total={len(self.rgb_history)}")


    def _maybe_append_rgb_history(self, cv_img):
        """Append RGB image to history at 1Hz if recording is active."""
        now = time.time()
        with self.lock:
            if not self.record_rgb_history:
                return
            if now - self.last_rgb_record_time < 1.0:
                return

            # self.rgb_history.append(cv_img.copy())
            self.rgb_history.append(cv2.rotate(cv_img, cv2.ROTATE_180))
            self.last_rgb_record_time = now
            logger.info(f"[RGB_HISTORY] append one frame, total={len(self.rgb_history)}")

    # ========================================================================
    # Panorama Capture (360-degree rotation)
    # ========================================================================

    def capture_panorama(self):
        """
        Capture 360-degree panorama by rotating drone in place.
        Captures 4 directions: front (0°), right (90°), back (180°), left (270°)

        Returns:
            bool: True if panorama captured successfully
        """
        logger.info(f"[Step {self.current_step}] Starting 360° panorama capture")

        # Wait for initial sensor data
        if not self._wait_for_sensors(timeout=1):
            logger.error("Sensors not ready for panorama capture")
            return False

        # Capture sequence: front, right, back, left
        directions = [
            {"name": "front", "angle":   0, "idx": 0},
            {"name": "right", "angle": -90, "idx": 1},
            {"name": "back",  "angle": -180, "idx": 2},
            {"name": "left",  "angle": -270, "idx": 3},
        ]

        initial_yaw = None
        with self.lock:
            initial_yaw = self.yaw_enu_deg
            self.panorama_origin_yaw_deg = self.yaw_enu_deg
            self.panorama_origin_pos = self.position_enu.copy()

        for i, direction in enumerate(directions):
            # Calculate target yaw for this direction
            target_yaw = wrap_deg180(initial_yaw + direction["angle"])

            # Rotate to target yaw (publish rotation waypoint)
            if i > 0:  # Skip rotation for first direction (already facing it)
                self._rotate_to_yaw(target_yaw)
                time.sleep(1.0)  # Wait for stabilization

            # Capture current view
            with self.lock:
                if self.rgb_image is not None and self.depth_image is not None:
                    # self.panorama_rgb[direction["idx"]] = self.rgb_image.copy()
                    self.panorama_rgb[direction["idx"]] = cv2.rotate(self.rgb_image, cv2.ROTATE_180)
                    # self.panorama_depth[direction["idx"]] = self.depth_image.copy()
                    self.panorama_depth[direction["idx"]] = cv2.rotate(self.depth_image, cv2.ROTATE_180)
                    logger.info(f"Captured {direction['name']} view (yaw={target_yaw:.1f}°)")
                else:
                    logger.error(f"Failed to capture {direction['name']} view")
                    return False

        # Rotate back to initial yaw
        self._rotate_to_yaw(initial_yaw)
        time.sleep(0.5)

        self.panorama_captured = True
        logger.info(f"[Step {self.current_step}] Panorama capture complete")
        return True

    def _wait_for_sensors(self, timeout=5.0):
        """Wait for sensor data to be available"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if self.have_rgb and self.have_depth and self.have_local_pose:
                    return True
            time.sleep(0.1)
        return False

    # def _rotate_to_yaw(self, target_yaw_deg):
    #     """
    #     Rotate drone to target yaw by publishing a waypoint at current position
    #     with desired orientation.

    #     Args:
    #         target_yaw_deg: Target yaw in degrees (ENU frame)
    #     """
    #     with self.lock:
    #         current_pos = self.position_enu.copy()
    #         # Record current arrival_seq before publishing waypoint
    #         expected_arrival_seq = self._arrival_seq + 1

    #     # Create waypoint at current position with target yaw
    #     waypoint = PoseStamped()
    #     waypoint.header.stamp = rospy.Time.now()
    #     waypoint.header.frame_id = "map"
    #     waypoint.pose.position.x = current_pos[0]
    #     waypoint.pose.position.y = current_pos[1]
    #     waypoint.pose.position.z = current_pos[2]

    #     # Convert yaw to quaternion
    #     target_yaw_rad = math.radians(target_yaw_deg)
    #     q = tft.quaternion_from_euler(0.0, 0.0, target_yaw_rad)
    #     waypoint.pose.orientation.x = q[0]
    #     waypoint.pose.orientation.y = q[1]
    #     waypoint.pose.orientation.z = q[2]
    #     waypoint.pose.orientation.w = q[3]

    #     self.waypoint_pub.publish(waypoint)
    #     logger.info(f"Published rotation waypoint: yaw={target_yaw_deg:.1f}°, waiting for arrival_seq={expected_arrival_seq}")

    #     # Wait for rotation to complete (wait for NEW arrival)
    #     self._wait_for_new_arrival(expected_arrival_seq, timeout=10.0)

    def _rotate_to_yaw(self, target_yaw_deg):
        """Rotate in-place to target yaw."""
        with self.lock:
            current_pos = self.position_enu.copy()
            expected_arrival_seq = self._arrival_seq + 1
            current_yaw = self.yaw_enu_deg

        waypoint = PoseStamped()
        waypoint.header.stamp = rospy.Time.now()
        waypoint.header.frame_id = "map"
        waypoint.pose.position.x = current_pos[0]
        waypoint.pose.position.y = current_pos[1]
        waypoint.pose.position.z = current_pos[2]

        target_yaw_rad = math.radians(target_yaw_deg)
        q = tft.quaternion_from_euler(0.0, 0.0, target_yaw_rad)
        waypoint.pose.orientation.x = q[0]
        waypoint.pose.orientation.y = q[1]
        waypoint.pose.orientation.z = q[2]
        waypoint.pose.orientation.w = q[3]

        self.waypoint_pub.publish(waypoint)

        logger.info(f"[ROTATE] Published yaw-only waypoint → target={target_yaw_deg:.1f}° "
                   f"(current={current_yaw:.1f}°, delta={wrap_deg180(target_yaw_deg - current_yaw):+.1f}°), "
                   f"waiting for arrival_seq={expected_arrival_seq}")

        time.sleep(0.6)

        arrived = self._wait_for_new_arrival(expected_arrival_seq, timeout=5.0)

        if arrived:
            logger.info(f"[ROTATE SUCCESS] Reached yaw {target_yaw_deg:.1f}°")
            time.sleep(1)
        else:
            logger.warning(f"[ROTATE TIMEOUT] Failed to reach yaw {target_yaw_deg:.1f}°, continuing anyway")
            time.sleep(1)

        return arrived

    def _wait_for_new_arrival(self, expected_seq, timeout=10.0):
        """
        Wait for a NEW arrival event (arrival_seq reaches expected value).

        Args:
            expected_seq: Expected arrival_seq value
            timeout: Timeout in seconds

        Returns:
            bool: True if new arrival detected, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if self._arrival_seq >= expected_seq:
                    logger.info(f"Detected new arrival: arrival_seq={self._arrival_seq}")
                    return True
            time.sleep(0.1)
        logger.warning(f"Timeout waiting for new arrival (expected_seq={expected_seq}, current={self._arrival_seq})")
        return False

    def _wait_for_arrival(self, timeout=10.0):
        """
        Wait for drone to reach waypoint (nav_state becomes ARRIVED).
        DEPRECATED: Use _wait_for_new_arrival() instead to avoid false positives.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if self.nav_state == self.STATE_ARRIVED:
                    return True
            time.sleep(0.1)
        logger.warning("Timeout waiting for arrival")
        return False

    # ========================================================================
    # Coordinate Transformations
    # ========================================================================

    def unproject_pixel_to_3d(self, u, v, depth, direction_idx):
        """
        Unproject 2D pixel to 3D point in camera frame.

        Args:
            u, v: Pixel coordinates
            depth: Depth value in meters
            direction_idx: Index of panorama direction (0=front, 1=right, 2=back, 3=left)

        Returns:
            np.array: 3D point in camera frame [x, y, z]
        """
        if depth <= 0.01 or depth > 10.0:
            logger.warning(f"Invalid depth value: {depth}")
            return None

        # Camera frame: X right, Y down, Z forward
        x_cam = (u - self.cx) * depth / self.fx
        y_cam = (v - self.cy) * depth / self.fy
        z_cam = depth

        return np.array([x_cam, y_cam, z_cam])

    def camera_to_body_frame(self, point_cam, direction_idx):
        """
        Transform point from camera frame to drone body frame.

        Camera frame: X right, Y down, Z forward
        Body frame: X forward, Y left, Z up

        Args:
            point_cam: 3D point in camera frame [x, y, z]
            direction_idx: Panorama direction index

        Returns:
            np.array: 3D point in body frame [x, y, z]
        """
        # Camera to body transformation (RealSense D435i mounted forward-facing)
        # Camera Z -> Body X (forward)
        # Camera -X -> Body Y (left)
        # Camera -Y -> Body Z (up)

        x_body = point_cam[2]   # Z_cam -> X_body
        y_body = -point_cam[0]  # -X_cam -> Y_body
        z_body = -point_cam[1]  # -Y_cam -> Z_body

        # Apply rotation based on panorama direction
        # direction_idx: 0=front(0°), 1=right(90°), 2=back(180°), 3=left(270°)
        direction_angle_deg = {
            0: 0.0,     # front
            1: -90.0,   # right
            2: 180.0,   # back
            3: 90.0     # left
        }
        angle_rad = math.radians(direction_angle_deg[direction_idx])

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Rotate around Z axis (yaw rotation)
        x_rotated = x_body * cos_a - y_body * sin_a
        y_rotated = x_body * sin_a + y_body * cos_a
        z_rotated = z_body

        return np.array([x_rotated, y_rotated, z_rotated])

    def body_to_enu_frame(self, point_body, drone_pos_enu, drone_yaw_deg):
        """
        Transform point from drone body frame to ENU world frame.

        Args:
            point_body: 3D point in body frame [x, y, z]
            drone_pos_enu: Drone position in ENU [x, y, z]
            drone_yaw_deg: Drone yaw in degrees (ENU frame)

        Returns:
            np.array: 3D point in ENU frame [x, y, z]
        """
        yaw_rad = math.radians(drone_yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        # Rotate body frame point to ENU frame
        x_enu = point_body[0] * cos_yaw - point_body[1] * sin_yaw
        y_enu = point_body[0] * sin_yaw + point_body[1] * cos_yaw
        z_enu = point_body[2]

        # Translate to world position
        point_enu = np.array([
            drone_pos_enu[0] + x_enu,
            drone_pos_enu[1] + y_enu,
            drone_pos_enu[2] + z_enu
        ])

        return point_enu

    def bbox_to_goal_enu(self, bbox, direction_idx):
        """
        Convert bounding box to 3D goal position in ENU frame.

        Args:
            bbox: [x1, y1, x2, y2] bounding box in pixels
            direction_idx: Panorama direction index

        Returns:
            np.array: Goal position in ENU frame [x, y, z], or None if failed
        """
        if bbox is None or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = bbox

        # Target pixel: bottom center of bbox (25% up from bottom)
        box_h = y2 - y1
        cx = int((x1 + x2) / 2)
        cy = int(y2 - 0.25 * box_h)

        # Get depth at target pixel
        depth_img = self.panorama_depth[direction_idx]
        if depth_img is None:
            logger.error("Depth image not available")
            return None

        h, w = depth_img.shape
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            logger.error(f"Pixel out of bounds: ({cx}, {cy})")
            return None

        # Sample depth in small window for robustness
        k = 3
        cx_min = max(0, cx - k)
        cx_max = min(w, cx + k + 1)
        cy_min = max(0, cy - k)
        cy_max = min(h, cy + k + 1)

        depth_window = depth_img[cy_min:cy_max, cx_min:cx_max]
        valid_depths = depth_window[(depth_window > 0.1) & (depth_window < 10.0)]

        if len(valid_depths) == 0:
            logger.error("No valid depth values in bbox region")
            return None

        depth = float(np.percentile(valid_depths, 30))
        logger.info(f"Bbox depth: {depth:.2f}m at pixel ({cx}, {cy})")

        # Unproject to 3D
        point_cam = self.unproject_pixel_to_3d(cx, cy, depth, direction_idx)
        if point_cam is None:
            return None

        # Transform to body frame
        point_body = self.camera_to_body_frame(point_cam, direction_idx)

        # Transform to ENU frame
        # Use panorama capture origin pose, not current rotated pose
        with self.lock:
            drone_pos = self.panorama_origin_pos.copy()
            drone_yaw = self.panorama_origin_yaw_deg

        point_enu = self.body_to_enu_frame(point_body, drone_pos, drone_yaw)

        logger.info(f"Goal ENU: ({point_enu[0]:.2f}, {point_enu[1]:.2f}, {point_enu[2]:.2f})")
        return point_enu


    # ========================================================================
    # LLM Prompts and Decision Making
    # ========================================================================

    def _generate_initial_todo(self):
        """Generate initial TODO list using Gemini-2.5-pro"""
        logger.info("Generating initial TODO list...")

        content = [
            {
                "type": "text",
                "text": f'Instruction: "{self.instruction_text}"\n\n'
                        "The images provided are 4-directional views from the starting position."
            }
        ]

        # Add panorama images
        for i, direction in enumerate(["front", "right", "back", "left"]):
            img_base64 = numpy_to_base64(self.panorama_rgb[i])
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })
            content.append({"type": "text", "text": f"Image {i+1}: {direction.upper()} view"})

        prompt = """
Your task is to create a dynamic checklist (TODO list) to complete the instruction based on the visual context.

Requirements:
- Break down the instruction into logical, sequential steps
- Use the visual information to identify landmarks or initial direction if possible
- Format as a Markdown checklist:
  - [ ] Step 1 description
  - [ ] Step 2 description

Response format:
Return ONLY the markdown checklist string. Do not use JSON.
"""
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        todo = self.model.generate_with_secondary(
            messages=messages,
            max_new_tokens=8192,
            temperature=0.1
        )

        self._append_model_reply("_generate_initial_todo", todo)

        logger.info(f"Initial TODO List:\n{todo}")
        return todo

    def _decide_navigation_direction(self):
        """
        Strategic decision: which direction to go (Gemini-2.5-pro)

        Returns:
            dict: {
                "turn_direction": "front"/"right"/"left"/"back",
                "stop": bool,
                "reasoning": str,
                "updated_todo_list": str
            }
        """
        logger.info(f"[Step {self.current_step}] Strategic decision: choosing direction")
        logger.info(f"[Step {self.current_step}] rgb_history size = {len(self.rgb_history)}")

        # Prepare history info
        history_text = "No visual history available yet." if len(self.history_info) == 0 else \
                       f"Previous {len(self.history_info)} steps completed."

        content = [
            {
                "type": "text",
                "text": f'Navigation Task: "{self.instruction_text}"\n\n'
                        f"Current Step: {self.current_step}"
            }
        ]

        # Add history summary
        content.append({"type": "text", "text": f"History: {history_text}"})


        # Add rgb history images collected during previous moving phases
        if len(self.rgb_history) > 0:
            content.append({"type": "text", "text": f"RGB History from previous moving phases: total {len(self.rgb_history)} images"})
            for i, hist_img in enumerate(self.rgb_history):
                hist_base64 = numpy_to_base64(hist_img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{hist_base64}"}
                })
                content.append({
                    "type": "text",
                    "text": f"History RGB {i+1}"
                })
        else:
            content.append({"type": "text", "text": "RGB History: no previous moving RGB images available yet."})


        # Add current panorama views
        content.append({"type": "text", "text": "Current Panorama Views:"})
        for i, direction in enumerate(["front", "right", "back", "left"]):
            img_base64 = numpy_to_base64(self.panorama_rgb[i])
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })
            content.append({
                "type": "text",
                "text": f"Image {i+1}: {direction.upper()} view (Step {self.current_step})"
            })

        allowed_dirs = '"front", "left", "right", "back"' if self.current_step == 1 else '"front", "left", "right"'

        prompt = f"""
**ROLE**: You are an intelligent drone navigator using a checklist to guide your actions.
**MISSION**: "{self.instruction_text}"

**Current TODO List**:
{self.todo_list}

You are also given RGB history images collected during previous moving phases. Use them together with the textual history to infer what the drone has recently passed by and what direction is most consistent.

**Task**:
1. **Update the TODO list**:
   - Check if the current step is completed based on the visual views
   - If completed, mark it as [x] and append "Result: ..."
   - If the plan is stuck, add new items or modify steps

2. **Decide the next action**:
   - Based on the *first incomplete* TODO item
   - Choose strictly from: {allowed_dirs}

3. **Stop Decision**:
   - Set "stop": true if you are sure you have reached the final goal

**JSON RESPONSE FORMAT**:
{{
    "progress_analysis": "Assessment of current progress...",
    "updated_todo_list": "The full updated Markdown checklist string (with [x] and [ ])",
    "reasoning": "Why update the list this way and why choose this direction...",
    "turn_direction": "front" or "right" or "left" or "back",
    "stop": true or false,
    "expected_landmark": "What to look for next"
}}
"""
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        result_text = self.model.generate_with_secondary(
            messages=messages,
            max_new_tokens=8192,
            temperature=0
        )

        self._append_model_reply("_decide_navigation_direction", result_text)

        return safe_json_loads(result_text)

    def _query_tactical_bbox(self, direction_idx, strategic_goal, strategic_stop):
        """
        Tactical decision: detect target bbox (Qwen2.5-VL-32B)

        Args:
            direction_idx: Index of chosen direction (0=front, 1=right, 2=back, 3=left)
            strategic_goal: Strategic reasoning from Gemini
            strategic_stop: Whether strategic layer suggests stopping

        Returns:
            dict: {
                "action": "NAVIGATE"/"STOP",
                "bbox_2d": [x1, y1, x2, y2],
                "target": str,
                "visual_check": str
            }
        """
        logger.info(f"[Step {self.current_step}] Tactical bbox detection")

        img_np = self.panorama_rgb[direction_idx]
        if img_np is None:
            return {}

        img_base64 = numpy_to_base64(img_np)

        prompt = f"""
**ROLE**: You are a drone navigator's TACTICAL EYES.
**MISSION**: "{self.instruction_text}"
**CURRENT STRATEGY**: "{strategic_goal}"
**STRATEGIC STOP SIGNAL**: {strategic_stop}

**INPUT**: You are looking at the CURRENT VIEW after turning.

**TASK**:
1. **Verification**: Do you see the object/area mentioned in "CURRENT STRATEGY"?
2. **Targeting**: Draw a Bounding Box (bbox_2d) around the best navigation target to move forward.
   - If the target object is visible, box it
   - If not, box the landmark mentioned in CURRENT STRATEGY

3. **Action Decision (NAVIGATE vs STOP)**:
   - **NAVIGATE**: If the target is far away or not centered
   - **STOP**: ONLY if the target is clearly visible, centered, and **occupies more than 20% of the image height**
   - **SPECIAL CASE**: If STRATEGIC STOP SIGNAL is True, verify if we are indeed at the goal. If yes, output STOP.

**JSON FORMAT**:
{{
    "visual_check": "I see [Object] which aligns with strategy...",
    "action": "NAVIGATE" or "STOP",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "Name of the object in the bbox",
    "stop_reasoning": "Only fill this if stopping."
}}
"""

        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            },
            {"type": "text", "text": prompt}
        ]

        messages = [{"role": "user", "content": content}]
        result_text = self.model.generate_with_primary(
            messages=messages,
            max_new_tokens=4096,
            temperature=0
        )

        self._append_model_reply("_query_tactical_bbox", result_text)

        parsed = safe_json_loads(result_text)

        # Save visualization with bbox
        self._save_bbox_visualization(img_np, parsed, direction_idx)

        return parsed

    def _save_bbox_visualization(self, img, bbox_data, direction_idx):
        """Draw and save bbox visualization"""
        bbox = bbox_data.get("bbox_2d", [])
        if not bbox or len(bbox) != 4:
            return

        vis_img = img.copy()
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = vis_img.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        action = bbox_data.get("action", "NAVIGATE")
        color = (0, 255, 0) if action == "NAVIGATE" else (0, 0, 255)

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            vis_img, f"{action}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )

        # Save to /tmp for debugging
        save_path = f"/tmp/bbox_step{self.current_step}.jpg"
        cv2.imwrite(save_path, vis_img)

        save_path_log = os.path.join(self.log_bbox_dir, f"bbox_step{self.current_step}.jpg")
        cv2.imwrite(save_path_log, vis_img)

        logger.info(f"Saved bbox visualization: {save_path}")
        logger.info(f"Saved bbox visualization (log): {save_path_log}")

    def _save_trajectory_visualization(self, img, trajectory_robot, direction_idx, num_waypoints=6):
        """
        Draw NavDP trajectory on RGB image and save.

        Args:
            img: RGB image (numpy array)
            trajectory_robot: List of 3D points in robot frame [(x, y, z), ...]
            direction_idx: Panorama direction index (0=front, 1=right, 2=back, 3=left)
            num_waypoints: Number of waypoints to execute (green), rest are red
        """
        if trajectory_robot is None or len(trajectory_robot) == 0:
            logger.warning("Empty trajectory, skipping visualization")
            return

        vis_img = img.copy()
        h, w = vis_img.shape[:2]

        # Project trajectory points to 2D pixels
        pixel_points = []
        for point_robot in trajectory_robot:
            pixel = self._project_robot_to_pixel(point_robot, direction_idx)
            if pixel is not None:
                pixel_points.append(pixel)

        if len(pixel_points) == 0:
            logger.warning("No valid trajectory points to visualize")
            return

        # Draw trajectory points
        for i, (u, v) in enumerate(pixel_points):
            # Green for first num_waypoints, red for rest
            if i < num_waypoints:
                color = (0, 255, 0)  # Green (BGR)
            else:
                color = (0, 0, 255)  # Red (BGR)

            # Draw circle for each point
            cv2.circle(vis_img, (int(u), int(v)), 5, color, -1)

            # Draw line connecting points
            if i > 0:
                prev_u, prev_v = pixel_points[i-1]
                # Use color based on current point
                cv2.line(vis_img, (int(prev_u), int(prev_v)), (int(u), int(v)), color, 2)

        # Add legend
        cv2.putText(vis_img, "Green: Execute", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_img, "Red: Future", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Save to /tmp
        save_path = f"/tmp/waypoint_step{self.current_step}.jpg"
        cv2.imwrite(save_path, vis_img)

        save_path_log = os.path.join(self.log_traj_dir, f"trajectory_step{self.current_step}.jpg")
        cv2.imwrite(save_path_log, vis_img)

        logger.info(f"Saved trajectory visualization: {save_path}")
        logger.info(f"Saved trajectory visualization (log): {save_path_log}")

    def _project_robot_to_pixel(self, point_robot, direction_idx):
        """
        Project 3D point in robot frame to 2D pixel coordinates.

        Args:
            point_robot: 3D point in robot frame [x, y, z]
            direction_idx: Panorama direction index (0=front, 1=right, 2=back, 3=left)

        Returns:
            (u, v): Pixel coordinates, or None if point is behind camera
        """
        # Reverse the rotation applied during panorama capture
        direction_angle_deg = {
            0: 0.0,     # front
            1: 90.0,    # right view -> inverse rotation
            2: 180.0,   # back view
            3: -90.0    # left view -> inverse rotation
        }
        angle_rad = math.radians(direction_angle_deg[direction_idx])

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Rotate point back to camera's local frame
        x_local = point_robot[0] * cos_a - point_robot[1] * sin_a
        y_local = point_robot[0] * sin_a + point_robot[1] * cos_a
        z_local = point_robot[2]

        # Transform from body frame to camera frame
        # Body frame: X forward, Y left, Z up
        # Camera frame: X right, Y down, Z forward
        # Inverse transformation:
        # X_cam = -Y_body
        # Y_cam = -Z_body
        # Z_cam = X_body
        x_cam = -y_local
        y_cam = -z_local
        z_cam = x_local

        # Check if point is in front of camera
        if z_cam <= 0.1:
            return None

        # Project to pixel coordinates using camera intrinsics
        u = self.fx * x_cam / z_cam + self.cx
        v = self.fy * y_cam / z_cam + self.cy

        # Check if pixel is within image bounds
        if u < 0 or u >= self.image_width or v < 0 or v >= self.image_height:
            return None

        return (u, v)


    # ========================================================================
    # NavDP Trajectory Planning and Execution
    # ========================================================================

    def _plan_trajectory_navdp(self, goal_enu, direction_idx):
        """
        Plan trajectory using NavDP from current position to goal.

        Args:
            goal_enu: Goal position in ENU frame [x, y, z]
            direction_idx: Direction index for RGB/depth images

        Returns:
            trajectory_enu: List of waypoints in ENU frame, or None if failed
        """
        logger.info(f"[Step {self.current_step}] Planning trajectory with NavDP")

        # Get current drone state
        with self.lock:
            drone_pos = self.position_enu.copy()
            drone_yaw = self.yaw_enu_deg

        # Convert goal from ENU to robot frame (for NavDP input)
        goal_robot = self._enu_to_robot_frame(goal_enu, drone_pos, drone_yaw)
        goal_2d = np.array([[goal_robot[0], goal_robot[1]]])  # NavDP expects (N, 2)

        logger.info(f"Goal robot frame: ({goal_robot[0]:.2f}, {goal_robot[1]:.2f})")

        # Get RGB and depth for current direction
        rgb = self.panorama_rgb[direction_idx]
        depth = self.panorama_depth[direction_idx]

        if rgb is None or depth is None:
            logger.error("RGB or depth not available for planning")
            return None

        # Call NavDP
        try:
            trajectory, all_trajectories, all_values = self.navdp_client.pointgoal_step(
                point_goals=goal_2d,
                rgb_images=[rgb],
                depth_images=[depth],
                intrinsic=self.camera_intrinsic
            )

            if trajectory is None or len(trajectory) == 0:
                logger.error("NavDP returned empty trajectory")
                return None

            # trajectory shape: (batch, N, 3) -> take first batch
            traj_robot = trajectory[0]  # (N, 3)
            logger.info(f"NavDP returned trajectory with {len(traj_robot)} points")

            # Visualize trajectory on RGB image
            self._save_trajectory_visualization(rgb, traj_robot, direction_idx, num_waypoints=6)

            # Convert trajectory from robot frame to ENU frame
            trajectory_enu = []
            for point_robot in traj_robot:
                point_enu = self.body_to_enu_frame(point_robot, drone_pos, drone_yaw)
                trajectory_enu.append(point_enu)

            return trajectory_enu

        except Exception as e:
            logger.error(f"NavDP planning failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _enu_to_robot_frame(self, point_enu, drone_pos_enu, drone_yaw_deg):
        """
        Transform point from ENU world frame to robot body frame.

        Args:
            point_enu: 3D point in ENU frame [x, y, z]
            drone_pos_enu: Drone position in ENU [x, y, z]
            drone_yaw_deg: Drone yaw in degrees (ENU frame)

        Returns:
            np.array: 3D point in robot frame [x, y, z]
        """
        # Translate to drone-relative position
        dx = point_enu[0] - drone_pos_enu[0]
        dy = point_enu[1] - drone_pos_enu[1]
        dz = point_enu[2] - drone_pos_enu[2]

        # Rotate to body frame
        yaw_rad = math.radians(drone_yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        x_robot = dx * cos_yaw + dy * sin_yaw
        y_robot = -dx * sin_yaw + dy * cos_yaw
        z_robot = dz

        return np.array([x_robot, y_robot, z_robot])

    def _execute_trajectory(self, trajectory_enu):
        """
        Execute only the N-th waypoint from NavDP trajectory.
        Handles timeout recovery to prevent state machine from getting stuck.
        """
        if trajectory_enu is None or len(trajectory_enu) == 0:
            logger.error("Empty trajectory, cannot execute")
            return False

        total_points = len(trajectory_enu)
        target_idx = self.N - 1
        if target_idx >= total_points:
            logger.warning(f"Trajectory only has {total_points} points, fallback to last")
            target_idx = total_points - 1

        waypoint_enu = trajectory_enu[target_idx]
        self._append_navdp_waypoint_log(waypoint_enu, total_points, target_idx)

        logger.info(
            f"[Step {self.current_step}] Executing only waypoint N={self.N}/{total_points}: "
            f"({waypoint_enu[0]:.2f}, {waypoint_enu[1]:.2f}, {waypoint_enu[2]:.2f})"
        )

        with self.lock:
            expected_arrival_seq = self._arrival_seq + 1

        # Publish
        self._publish_waypoint_enu(waypoint_enu)

        start_wait = time.time()
        while time.time() - start_wait < 2.0:
            with self.lock:
                if self.nav_state == self.STATE_MOVING:
                    self._start_rgb_history_recording()
                    break
            time.sleep(0.05)

        logger.info(f"Waiting for arrival (expected_seq={expected_arrival_seq}, timeout=30s)...")

        start_time = time.time()
        arrived = False
        while time.time() - start_time < 10.0:
            with self.lock:
                if self._arrival_seq >= expected_arrival_seq and self.nav_state == self.STATE_ARRIVED:
                    logger.info(f"[SUCCESS] Waypoint N={self.N} reached! seq={self._arrival_seq}")
                    arrived = True
                    break
            time.sleep(0.05)

        if arrived:
            logger.info(f"[Step {self.current_step}] Waypoint reached, continue to next step")
            time.sleep(0.6)
            return True
        else:
            logger.warning(f"[TIMEOUT] Waypoint N={self.N} not reached (seq stuck at {self._arrival_seq})")
            self._stop_rgb_history_recording()
            with self.lock:
                self._arrival_seq = expected_arrival_seq
                self.nav_state = self.STATE_ARRIVED
                self._last_inferred_arrival_seq = expected_arrival_seq - 1

            with self.lock:
                current_pos = self.position_enu.copy()
                current_yaw = self.yaw_enu_deg

            hold_wp = PoseStamped()
            hold_wp.header.stamp = rospy.Time.now()
            hold_wp.header.frame_id = "map"
            hold_wp.pose.position.x = current_pos[0]
            hold_wp.pose.position.y = current_pos[1]
            hold_wp.pose.position.z = current_pos[2]
            q = tft.quaternion_from_euler(0, 0, math.radians(current_yaw))
            hold_wp.pose.orientation.x = q[0]
            hold_wp.pose.orientation.y = q[1]
            hold_wp.pose.orientation.z = q[2]
            hold_wp.pose.orientation.w = q[3]
            self.waypoint_pub.publish(hold_wp)

            logger.info("[RECOVERY] Forced hold waypoint + ARRIVED state to unblock next step")
            time.sleep(1.2)
            return False

    def _publish_waypoint_enu(self, position_enu):
        """
        Publish waypoint in ENU frame.

        Args:
            position_enu: [x, y, z] position in ENU frame
        """
        waypoint = PoseStamped()
        waypoint.header.stamp = rospy.Time.now()
        waypoint.header.frame_id = "map"
        waypoint.pose.position.x = position_enu[0]
        waypoint.pose.position.y = position_enu[1]
        waypoint.pose.position.z = 1.2
        # waypoint.pose.position.z = position_enu[2]

        # Keep current yaw (or point towards waypoint)
        with self.lock:
            current_pos = self.position_enu.copy()
            current_yaw = self.yaw_enu_deg

        # Calculate yaw to point towards waypoint
        dx = position_enu[0] - current_pos[0]
        dy = position_enu[1] - current_pos[1]

        if dx*dx + dy*dy > 0.1:  # Only update yaw if moving significantly
            target_yaw_rad = math.atan2(dy, dx)
        else:
            target_yaw_rad = math.radians(current_yaw)

        q = tft.quaternion_from_euler(0.0, 0.0, target_yaw_rad)
        waypoint.pose.orientation.x = q[0]
        waypoint.pose.orientation.y = q[1]
        waypoint.pose.orientation.z = q[2]
        waypoint.pose.orientation.w = q[3]

        self.waypoint_pub.publish(waypoint)


    # ========================================================================
    # Main Navigation Loop
    # ========================================================================

    def run(self):
        """
        Main navigation loop - called repeatedly from ROS node.
        Implements step-by-step navigation synchronized with model_set_node.
        """
        # Check if we should run inference (only when ARRIVED and new arrival)
        with self.lock:
            if not self.have_nav_state:
                return  # Wait for nav_state

            if self.nav_state != self.STATE_ARRIVED:
                return  # Wait for ARRIVED state

            current_arrival_seq = self._arrival_seq

            # Check if we already processed this arrival
            if current_arrival_seq == self._last_inferred_arrival_seq:
                return  # Already processed this arrival

            # Mark this arrival as being processed
            self._last_inferred_arrival_seq = current_arrival_seq

        # Increment step counter
        self.current_step += 1
        self._append_step_header()

        logger.info(f"\n{'='*60}")
        logger.info(f"[STEP {self.current_step}] Starting navigation step")
        logger.info(f"{'='*60}\n")

        try:
            # Step 1: Capture 360-degree panorama
            if not self.capture_panorama():
                logger.error("Failed to capture panorama, aborting step")
                return

            # Step 2: Generate initial TODO list (first step only)
            if self.current_step == 1:
                self.todo_list = self._generate_initial_todo()

            # Step 3: Strategic decision (Gemini-2.5-pro)
            decision = self._decide_navigation_direction()

            # Update TODO list
            if decision.get("updated_todo_list"):
                self.todo_list = decision["updated_todo_list"]
                logger.info(f"Updated TODO:\n{self.todo_list}")

            # Parse direction and stop signal
            turn_direction = decision.get("turn_direction", "front").lower()
            strategic_stop = bool(decision.get("stop", False))
            strategic_reasoning = decision.get("reasoning", "")

            logger.info(f"Strategic decision: {turn_direction}, stop={strategic_stop}")
            logger.info(f"Reasoning: {strategic_reasoning}")

            direction_map = {"front": 0, "right": 1, "back": 2, "left": 3}
            yaw_offset_map = {
                "front": 0.0,
                "right": -90.0,
                "back": 180.0,
                "left": 90.0
            }

            if turn_direction not in direction_map:
                logger.warning(f"Invalid direction '{turn_direction}', defaulting to 'front'")
                turn_direction = "front"

            direction_idx = direction_map[turn_direction]

            # Step 4: Tactical bbox detection first (on panorama image of chosen direction)
            bbox_decision = self._query_tactical_bbox(
                direction_idx,
                strategic_reasoning,
                strategic_stop
            )

            action = bbox_decision.get("action", "NAVIGATE")
            bbox = bbox_decision.get("bbox_2d")

            logger.info(f"Tactical decision: action={action}, bbox={bbox}")

            # Step 5: Rotate to chosen direction
            if turn_direction != "front":
                logger.info(f"Rotating to {turn_direction} direction")

                yaw_offset_map = {
                    "front": 0.0,
                    "right": -90.0,
                    "back": 180.0,
                    "left": 90.0
                }

                with self.lock:
                    target_yaw = wrap_deg180(self.panorama_origin_yaw_deg + yaw_offset_map[turn_direction])

                self._rotate_to_yaw(target_yaw)
                time.sleep(0.3)

            # Step 6: Check stop conditions
            should_stop = False

            if strategic_stop and action == "STOP":
                logger.info("Both strategic and tactical layers agree to STOP")
                should_stop = True
            elif strategic_stop and (bbox is None or len(bbox) != 4):
                logger.info("Strategic STOP with no valid bbox - stopping")
                should_stop = True
            elif action == "STOP" and (bbox is None or len(bbox) != 4):
                logger.info("Tactical STOP with no valid bbox - stopping")
                should_stop = True

            if should_stop:
                logger.info(f"\n{'='*60}")
                logger.info("NAVIGATION COMPLETE - Goal reached!")
                logger.info(f"{'='*60}\n")
                self.nav_done = True
                return

            # Step 7: Convert bbox to 3D goal in ENU frame
            if bbox is None or len(bbox) != 4:
                logger.warning("No valid bbox, using default forward goal")
                # Default: move 2m forward
                with self.lock:
                    drone_pos = self.position_enu.copy()
                    drone_yaw = self.yaw_enu_deg

                yaw_rad = math.radians(drone_yaw)
                goal_enu = np.array([
                    drone_pos[0] + 2.0 * math.cos(yaw_rad),
                    drone_pos[1] + 2.0 * math.sin(yaw_rad),
                    drone_pos[2]
                ])
            else:
                goal_enu = self.bbox_to_goal_enu(bbox, direction_idx)
                if goal_enu is None:
                    logger.error("Failed to convert bbox to 3D goal, aborting step")
                    return

            logger.info(f"Goal ENU: ({goal_enu[0]:.2f}, {goal_enu[1]:.2f}, {goal_enu[2]:.2f})")

            # Step 8: Plan trajectory with NavDP
            trajectory_enu = self._plan_trajectory_navdp(goal_enu, direction_idx)

            if trajectory_enu is None:
                logger.error("NavDP planning failed, aborting step")
                return

            # # Step 9: Execute trajectory (select 6 waypoints)
            # self._execute_trajectory(trajectory_enu, num_waypoints=6)

            # Step 9: Execute only the N-th waypoint from trajectory
            self._execute_trajectory(trajectory_enu)

            # Record history
            self.history_info.append({
                "step": self.current_step,
                "direction": turn_direction,
                "goal_enu": goal_enu.tolist(),
                "reasoning": strategic_reasoning
            })

            logger.info(f"\n{'='*60}")
            logger.info(f"[STEP {self.current_step}] Complete")
            logger.info(f"{'='*60}\n")

        except Exception as e:
            logger.error(f"Error in navigation step {self.current_step}: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# Usage:
#   1. Start NavDP server:
#      cd <workspace>/src/NavDP/baselines/navdp
#      python navdp_server.py --port 8888 --checkpoint <path_to_checkpoint>
#
#   2. Start model_set_node:
#      roslaunch vln_node model_set.launch
#
#   3. Start indoor navigation:
#      roslaunch vln_node indoor_eval.launch
#
# Environment variables required:
#   export OPENAI_API_KEY="<your_api_key>"
#   export OPENAI_BASE_URL="<your_api_base_url>"
#   export OPENAI_API_KEY_SECONDARY="<your_secondary_api_key>"
#   export OPENAI_BASE_URL_SECONDARY="<your_secondary_api_base_url>"
#
# ============================================================================

