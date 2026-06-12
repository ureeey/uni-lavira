"""HTTP client for the iPlanner remote service.

Thin ``requests``-based wrapper around the iPlanner Flask server. The server
URL defaults to ``Config.IPLANNER_URL`` so it is environment driven and ships
no absolute paths.

OpenCV (``cv2``) is imported lazily inside :meth:`get_plan` so this module can
be imported on a machine without OpenCV installed.
"""
import json
from typing import Optional, Tuple

import numpy as np
import requests

from config import Config


class IPlannerRemoteClient:
    """Thin requests-based wrapper around the iPlanner Flask server."""

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url or Config.IPLANNER_URL
        self.session = requests.Session()
        self.initialized = False

    def reset(self) -> bool:
        """Initialise / reset the planner model on the server."""
        url = f"{self.server_url}/navigator_reset"
        payload = {
            "intrinsic": [[384.0, 0.0, 320.0],
                          [0.0, 384.0, 240.0],
                          [0.0, 0.0, 1.0]],
            "stop_threshold": 0.1,
            "batch_size": 1,
        }
        try:
            # Short timeout for the reset / health check.
            resp = self.session.post(url, json=payload, timeout=2)
            if resp.status_code == 200:
                print(f"iPlanner server connected at {self.server_url}")
                self.initialized = True
                return True
            print(f"iPlanner init failed: {resp.text}")
            return False
        except Exception as e:
            # Suppress the full traceback for a refused connection; just warn.
            if "Connection refused" in str(e):
                print(
                    f"Cannot connect to iPlanner server at {self.server_url}. "
                    "Make sure the server is running."
                )
            else:
                print(f"Cannot connect to iPlanner server: {e}")
            return False

    def get_plan(self, rgb_img, depth_img_mm, goal_local) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Send sensor data and obtain a planned path.

        Args:
            rgb_img: ``(H, W, 3)`` BGR or RGB array.
            depth_img_mm: ``(H, W)`` uint16 depth image in millimetres.
            goal_local: ``(x, y)`` goal point in the robot frame.

        Returns:
            ``(trajectory_points, fear_value)`` on success, otherwise
            ``(None, None)``.
        """
        import cv2  # Lazy import: keep this module importable without OpenCV.

        if not self.initialized:
            if not self.reset():
                return None, None

        url = f"{self.server_url}/pointgoal_step"

        # 1. Prepare the goal payload.
        goal_payload = {
            "goal_x": [float(goal_local[0])],
            "goal_y": [float(goal_local[1])],
        }

        # 2. Encode the RGB image.
        _, rgb_encoded = cv2.imencode(".png", rgb_img)

        # 3. Depth handling: scale millimetres to 0.1-millimetre units as the
        #    server expects, then store as uint16.
        depth_scaled = (depth_img_mm.astype(np.uint32) * 10).astype(np.uint16)
        _, depth_encoded = cv2.imencode(".png", depth_scaled)

        # 4. Build the multipart request.
        files = {
            "image": ("rgb.png", rgb_encoded.tobytes(), "image/png"),
            "depth": ("depth.png", depth_encoded.tobytes(), "image/png"),
        }
        data = {"goal_data": json.dumps(goal_payload)}

        try:
            resp = self.session.post(url, files=files, data=data, timeout=2)
            if resp.status_code == 200:
                result = resp.json()
                traj = result["trajectory"][0]  # [N, 3]
                fear = result["all_values"][0][0]  # scalar
                return np.array(traj), fear
            print(f"iPlanner request failed: {resp.status_code} - {resp.text}")
            return None, None
        except Exception as e:
            print(f"iPlanner network error: {e}")
            return None, None
