"""iPlanner remote client for the Unitree G1.

Communicates with the iPlanner Flask server to obtain trajectory plans. The
client sends an RGB image, a depth image, and a goal point, and receives a
planned trajectory.

Import safety
-------------
OpenCV (``cv2``) is imported lazily inside :meth:`get_plan` so this module can
be imported on a machine without OpenCV installed.

Paths and network endpoints
---------------------------
The server URL defaults to ``Config.IPLANNER_URL`` so it is environment driven
and the module ships no absolute paths.
"""
from __future__ import annotations

import json
from typing import Optional, Sequence, Tuple

import numpy as np
import requests

from config import Config


class IPlannerRemoteClient:
    """Client for the iPlanner trajectory-planning server."""

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url or Config.IPLANNER_URL
        self.session = requests.Session()
        self.initialized = False

    def reset(self, intrinsic: Optional[Sequence[Sequence[float]]] = None) -> bool:
        """Initialise the iPlanner server with camera intrinsics.

        Args:
            intrinsic: 3x3 camera intrinsic matrix as a nested list.
                Default: ``[[384.0, 0.0, 320.0], [0.0, 384.0, 240.0],
                [0.0, 0.0, 1.0]]``.

        Returns:
            ``True`` if the server connection succeeded, ``False`` otherwise.
        """
        url = f"{self.server_url}/navigator_reset"

        # Default intrinsic if not provided.
        if intrinsic is None:
            intrinsic = [
                [384.0, 0.0, 320.0],
                [0.0, 384.0, 240.0],
                [0.0, 0.0, 1.0],
            ]

        payload = {
            "intrinsic": intrinsic,
            "stop_threshold": 0.1,
            "batch_size": 1,
        }
        try:
            resp = self.session.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                print("[iPlanner] Server connected successfully")
                self.initialized = True
                return True
            else:
                print(f"[iPlanner] Init failed: {resp.text}")
                return False
        except Exception as e:
            print(f"[iPlanner] Cannot connect to server: {e}")
            return False

    def get_plan(
        self, rgb_img, depth_img_mm, goal_local
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Send RGB-D data and goal to the iPlanner server, get a trajectory.

        Args:
            rgb_img: Numpy array ``(H, W, 3)`` in BGR format.
            depth_img_mm: Numpy array ``(H, W)`` uint16, in millimetres.
            goal_local: ``(x, y)`` goal in the robot frame (metres).

        Returns:
            ``(trajectory_points, fear_value)`` on success, otherwise
            ``(None, None)``. ``trajectory_points`` is a numpy array of shape
            ``(N, 3)`` in the robot frame.
        """
        import cv2  # Lazy import: keep this module importable without OpenCV.

        if not self.initialized:
            if not self.reset():
                return None, None

        url = f"{self.server_url}/pointgoal_step"

        # 1. Prepare goal data.
        goal_payload = {
            "goal_x": [float(goal_local[0])],
            "goal_y": [float(goal_local[1])],
        }

        # 2. Encode RGB image.
        _, rgb_encoded = cv2.imencode(".png", rgb_img)

        # 3. Depth processing: mm -> 0.1mm units for the iPlanner protocol.
        depth_scaled = (depth_img_mm.astype(np.uint32) * 10).astype(np.uint16)
        _, depth_encoded = cv2.imencode(".png", depth_scaled)

        # 4. Build request.
        files = {
            "image": ("rgb.png", rgb_encoded.tobytes(), "image/png"),
            "depth": ("depth.png", depth_encoded.tobytes(), "image/png"),
        }
        data = {"goal_data": json.dumps(goal_payload)}

        try:
            resp = self.session.post(url, files=files, data=data, timeout=5)
            if resp.status_code == 200:
                result = resp.json()
                traj = result["trajectory"][0]  # [N, 3]
                fear = result["all_values"][0][0]  # scalar
                return np.array(traj), fear
            else:
                print(
                    f"[iPlanner] Plan request failed: "
                    f"{resp.status_code} - {resp.text}"
                )
                return None, None
        except Exception as e:
            print(f"[iPlanner] Network error: {e}")
            return None, None
