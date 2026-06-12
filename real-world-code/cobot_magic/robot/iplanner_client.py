"""In-process iPlanner client for the Cobot Magic dynamic-replan navigator.

Wraps the vendored :class:`iplanner.iplanner_agent.IPlannerAgent` and reproduces
the depth -> trajectory inference originally inlined in the development script's
``_replan_thread_func``. The point-goal is expressed in the robot frame; the
returned trajectory is in the same robot frame (callers handle the global
transform when needed).

Import safety
-------------
``torch`` and ``IPlannerAgent`` are imported lazily inside ``__init__`` so this
module can be imported on a machine without torch / the iPlanner package / a
GPU. ``import robot.iplanner_client`` therefore succeeds on plain Python.

The iPlanner package modules (``planner_net``, ``traj_opt``) are top-level
imports inside the vendored agent, so the ``iplanner/`` directory is added to
``sys.path`` before importing the agent.
"""
from __future__ import annotations

import math
import os
import sys
import traceback
from typing import List, Optional, Tuple

import numpy as np

from config import Config
from utils import print_error, print_info, print_success

_HERE = os.path.dirname(os.path.abspath(__file__))
# The repository root (one level above ``robot/``).
_REPO_ROOT = os.path.dirname(_HERE)
# Vendored iPlanner package directory; its internal modules import each other
# by bare name (``import traj_opt`` / ``from planner_net import PlannerNet``).
_IPLANNER_DIR = os.path.join(_REPO_ROOT, "iplanner")


def _resolve_path(path: str) -> str:
    """Resolve a possibly repo-relative path against the repository root."""
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


class IPlannerClient:
    """In-process wrapper around :class:`IPlannerAgent`.

    Constructs the neural planner from ``Config`` (intrinsics, checkpoint, YAML
    config, device) and exposes :meth:`plan`, which mirrors the original
    ``_replan_thread_func`` inference: resize depth to ``224x224``, build the
    depth / goal tensors, run ``step_pointgoal``, and return the first batch's
    2D trajectory in the robot frame.
    """

    # Depth resize target used by the original replan inference.
    TARGET_SIZE: Tuple[int, int] = (224, 224)

    def __init__(self) -> None:
        # Lazy heavy imports so the module stays importable without torch / GPU.
        import torch  # noqa: WPS433 (runtime import is intentional)

        if _IPLANNER_DIR not in sys.path:
            sys.path.append(_IPLANNER_DIR)
        try:
            from iplanner.iplanner_agent import IPlannerAgent  # type: ignore
        except ImportError:
            from iplanner_agent import IPlannerAgent  # type: ignore

        self._torch = torch
        self.device = Config.IPLANNER_DEVICE
        self.intrinsics = Config.NAVDP_INTRINSICS

        config_path = _resolve_path(Config.IPLANNER_CONFIG_PATH)
        checkpoint_path = _resolve_path(Config.IPLANNER_CHECKPOINT)

        print_info(f"Initializing IPlanner Agent (device={self.device})...")
        try:
            self.agent = IPlannerAgent(
                image_intrinsic=torch.tensor(self.intrinsics),
                model_path=checkpoint_path,
                model_config_path=config_path,
                device=self.device,
            )
            print_success("IPlanner Initialized!")
        except Exception as exc:  # noqa: BLE001 - surface init failure to caller
            print_error(f"IPlanner Failed: {exc}")
            raise

    def plan(
        self,
        depth_image: np.ndarray,
        local_goal_xy: Tuple[float, float],
    ) -> Optional[List[List[float]]]:
        """Plan a robot-frame 2D trajectory toward ``local_goal_xy``.

        Reproduces the original ``_replan_thread_func`` inference exactly:

        1. Resize ``depth_image`` to ``224x224`` with nearest-neighbour.
        2. Build the ``[1, H, W, 1]`` float32 depth tensor on the planner device.
        3. Build the ``[[gx, gy, 0.0]]`` float32 goal tensor.
        4. Run ``step_pointgoal`` under ``torch.no_grad``.
        5. Return the first batch's trajectory as a list of ``[x, y]`` points in
           the robot frame (the ``z`` column, if present, is dropped).

        Args:
            depth_image: Front-camera depth image (metres), ``H x W`` float32.
            local_goal_xy: Goal ``(x, y)`` in the robot frame (x forward,
                y left).

        Returns:
            A list of ``[x, y]`` robot-frame waypoints, or ``None`` when
            inference yields no trajectory or fails.
        """
        import cv2  # noqa: WPS433 (runtime import is intentional)

        torch = self._torch
        try:
            local_goal_x, local_goal_y = float(local_goal_xy[0]), float(local_goal_xy[1])

            depth_resized = cv2.resize(
                depth_image, self.TARGET_SIZE, interpolation=cv2.INTER_NEAREST
            )
            depth_input = depth_resized.astype(np.float32)[np.newaxis, :, :, np.newaxis]
            depth_tensor = torch.as_tensor(depth_input, device=self.agent.device)

            goal_input = np.array(
                [[local_goal_x, local_goal_y, 0.0]], dtype=np.float32
            )
            goal_tensor = torch.as_tensor(goal_input, device=self.agent.device)

            with torch.no_grad():
                _, trajectory, _ = self.agent.step_pointgoal(depth_tensor, goal_tensor)

            traj_list = trajectory.cpu().numpy().tolist()
            if not traj_list:
                return None

            raw_path = np.array(traj_list[0])
            if raw_path.ndim == 2 and raw_path.shape[1] == 3:
                raw_path = raw_path[:, :2]
            return raw_path.tolist()
        except Exception as exc:  # noqa: BLE001 - replan must never crash the loop
            print_error(f"IPlanner replan error: {exc}\n{traceback.format_exc()}")
            return None

    @staticmethod
    def local_goal_from_global(
        global_goal: Tuple[float, float],
        obs_pose: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        """Transform a global ``(gx, gy)`` goal into the robot frame at ``obs_pose``.

        ``obs_pose`` is ``(ox, oy, oyaw)``. Returns the goal expressed in the
        robot frame (x forward, y left), matching the original replan transform.
        """
        ox, oy, oyaw = obs_pose
        cos_o, sin_o = math.cos(oyaw), math.sin(oyaw)
        dx = global_goal[0] - ox
        dy = global_goal[1] - oy
        local_goal_x = dx * cos_o + dy * sin_o
        local_goal_y = -dx * sin_o + dy * cos_o
        return local_goal_x, local_goal_y

    @staticmethod
    def path_to_global(
        local_path: List[List[float]],
        obs_pose: Tuple[float, float, float],
    ) -> np.ndarray:
        """Transform a robot-frame 2D path into the global frame at ``obs_pose``.

        ``obs_pose`` is ``(ox, oy, oyaw)``. Returns an ``N x 2`` array of global
        ``[gx, gy]`` points, matching the original replan transform.
        """
        ox, oy, oyaw = obs_pose
        cos_o, sin_o = math.cos(oyaw), math.sin(oyaw)
        path_global_list = []
        for pt in local_path:
            lx, ly = pt[0], pt[1]
            gx = ox + lx * cos_o - ly * sin_o
            gy = oy + lx * sin_o + ly * cos_o
            path_global_list.append([gx, gy])
        return np.array(path_global_list)
