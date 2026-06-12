"""
VLN task for the Unitree G1 humanoid.

Ported verbatim (in behaviour) from the monolithic source ``tasks/vln.py``.
The inline OpenAI client construction and the inline strategic / tactical VLM
methods (``_generate_initial_todo``, ``_decide_navigation``,
``_query_tactical_eyes``, ``_draw_bbox``) have been relocated into
``robot.navigation_api.LaViRANavigationAPI`` and are now invoked through
``self.nav_api.*``.

The per-cycle ``run()`` loop is unchanged: 4-direction panorama -> strategic LA
direction (with ``stop`` boolean) -> rotate (incl. ``behind`` -> 180-deg turn)
-> tactical VA bbox/STOP -> bbox bottom-center -> depth -> 3D goal -> iPlanner
-> ``execute_trajectory`` (with the ``SAFE_DISTANCE`` safety-buffer truncation)
-> strategic-stop final-approach handling.  All G1 constants and sleeps are
preserved.

The single approved deviation lives in ``LaViRANavigationAPI.vision_action``
(bbox ``[0, 1000]`` -> pixel de-normalisation); the bbox bottom-center goal
point below is unchanged from the G1 source.
"""
import os
import time
from typing import Dict, List

import numpy as np

from ai_client import LaViRAVisionClient
from config import Config
from robot.navigation_api import LaViRANavigationAPI
from tasks import register_task
from utils import (
    img_to_base64,
    print_action,
    print_error,
    print_info,
    print_success,
    print_warning,
)


@register_task("vln")
class VLNTask:
    """Vision-Language Navigation Task for the G1 humanoid."""

    def __init__(self, robot, instruction: str):
        self.robot = robot
        self.instruction = instruction

        # VLM client + navigation API (LA = strategic/secondary, VA =
        # tactical/primary).  Replaces the source's two inline OpenAI clients.
        self.client = LaViRAVisionClient()
        self.nav_api = LaViRANavigationAPI(self.client)

        self.current_step = 1
        self.todo_list = ""
        self.visited_targets = []
        self.global_target = instruction
        self.is_task_running = True

    def stop_task(self):
        """Stop the current task execution loop (used by the web demo)."""
        print_info("Stopping VLN Task...")
        self.is_task_running = False

    def run(self):
        """Main VLN navigation loop."""
        print_info(f"Starting VLN Task with instruction: {self.instruction}")
        self.is_task_running = True

        while self.robot.running and self.is_task_running:
            # =================================================================
            # 1. Capture 360-degree Panorama
            # =================================================================
            # If some cameras fail, we still proceed with what we have.
            self.robot.capture_all_directions(self.current_step)

            panorama_frames = self._get_panorama_frames()

            if not panorama_frames:
                print_error("No panorama images available. Cannot proceed.")
                break

            # =================================================================
            # 2. Generate Initial TODO (First step only)
            # =================================================================
            if self.current_step == 1:
                self.todo_list = self.nav_api.generate_initial_todo(
                    self.instruction, panorama_frames
                )

            # =================================================================
            # 3. Strategic Decision (LA model)
            # =================================================================
            decision = self.nav_api.language_action(
                instruction=self.instruction,
                global_target=self.global_target,
                todo_list=self.todo_list,
                iplanner_history=self.robot.iplanner_history,
                panorama_frames=panorama_frames,
                current_step=self.current_step,
            )

            # Update Todo List
            if decision.get("updated_todo_list"):
                self.todo_list = decision["updated_todo_list"]
                print_info(f"Updated TODO List:\n{self.todo_list}")

            # Parse direction and stop signal
            turn_direction = decision.get("turn_direction", "front").lower()

            raw_stop = decision.get("stop", False)
            if isinstance(raw_stop, str):
                strategic_stop = raw_stop.lower() == "true"
            else:
                strategic_stop = bool(raw_stop)

            # Validate direction
            allowed_directions = ["front", "right", "left", "behind"]
            if turn_direction not in allowed_directions:
                print_warning(
                    f"Unknown direction '{turn_direction}', defaulting to 'front'"
                )
                turn_direction = "front"

            strategic_reasoning = decision.get("reasoning", "")
            strategic_goal = f"Go {turn_direction}. {strategic_reasoning}"

            # =================================================================
            # 4. Rotate G1 Robot
            # =================================================================
            if turn_direction == "left":
                self.robot.rotate_left(90)
            elif turn_direction == "right":
                self.robot.rotate_right(90)
            elif turn_direction == "behind":
                self.robot.rotate_right(180)

            # =================================================================
            # 5. Tactical Eyes (VA model) - Bbox Detection
            # =================================================================
            if not self.robot.capture_current_image():
                print_warning("Failed to capture current view after rotation")
                self.current_step += 1
                continue

            # Force strategic_stop=False to ensure we get a target BBox for the
            # final approach.
            tactical_img = self.robot.direction_images[
                self.robot.current_direction
            ]["rgb"]
            bbox_decision = self.nav_api.vision_action(
                img_np=tactical_img,
                instruction=self.instruction,
                global_target=self.global_target,
                strategic_goal=strategic_goal,
                strategic_stop=False,
                current_step=self.current_step,
            )

            action = bbox_decision.get("action", "NAVIGATE")
            bbox = bbox_decision.get("bbox_2d")

            # =================================================================
            # Stop Logic
            # =================================================================
            should_stop_now = False

            if strategic_stop:
                if bbox and len(bbox) == 4:
                    print_info(
                        "Strategic Stop requested. Executing FINAL approach to target BBox."
                    )
                    action = "NAVIGATE"
                else:
                    print_success(
                        "Strategic Stop requested and no target BBox. Stopping immediately."
                    )
                    should_stop_now = True
            elif action == "STOP":
                if bbox and len(bbox) == 4:
                    print_info(
                        "Tactical Eyes triggered STOP with visible target. Executing FINAL approach."
                    )
                    action = "NAVIGATE"
                    strategic_stop = True
                else:
                    print_success("Tactical Eyes triggered STOP. Stopping immediately.")
                    should_stop_now = True

            if should_stop_now:
                break

            # =================================================================
            # 6. Compute 3D Goal from BBox + Depth
            # =================================================================
            target_pixel = None
            goal_2d = (1.5, 0.0)  # Default: 1.5m forward

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                # Target Pixel: Bottom Center of the bbox (G1 source behaviour).
                # Instead of center or 75% down, we select the bottom edge center.
                box_h = y2 - y1
                cx = int((x1 + x2) / 2)

                # Ensure cy does not exceed image height.
                depth_img = self.robot.direction_images[
                    self.robot.current_direction
                ]["depth"]
                img_h = depth_img.shape[0] if depth_img is not None else 480
                cy = min(int(y2), img_h - 1)

                target_pixel = (cx, cy)

                depth_val = self.robot.get_depth_value(
                    cx, cy, self.robot.current_direction
                )

                if depth_val is not None:
                    cam_data = self.robot.camera_data[self.robot.current_camera]
                    fx = cam_data.get("fx")
                    cx_int = cam_data.get("cx")

                    if fx and cx_int:
                        # Unproject: Camera Frame -> Robot Frame.
                        # Camera: Z forward, X right, Y down.
                        # Robot:  X forward, Y left.
                        z_3d = depth_val
                        x_3d = (cx - cx_int) * z_3d / fx

                        goal_x = z_3d  # Robot X = Camera Z
                        goal_y = -x_3d  # Robot Y = -Camera X

                        goal_2d = (goal_x, goal_y)
                        print_info(
                            f"Goal from BBox: ({goal_x:.2f}, {goal_y:.2f})m "
                            f"(Depth: {depth_val:.2f}m)"
                        )
                    else:
                        print_warning("Missing camera intrinsics for unprojection")
                else:
                    print_warning("Could not get depth for bbox target")

            # =================================================================
            # 7. iPlanner: Plan Trajectory
            # =================================================================
            rgb = self.robot.direction_images[self.robot.current_direction]["rgb"]
            depth = self.robot.direction_images[self.robot.current_direction]["depth"]

            print_action(f"Requesting plan from iPlanner to {goal_2d}...")
            traj, fear = self.robot.planner_client.get_plan(rgb, depth, goal_2d)

            # Visualization
            if traj is not None:
                timestamp = time.strftime("%H%M%S")
                save_name = f"step{self.current_step}_plan_{timestamp}.jpg"
                target_3d_viz = (goal_2d[0], goal_2d[1], 0.0) if goal_2d else None
                self.robot.project_trajectory_to_image(
                    traj,
                    rgb,
                    save_name=save_name,
                    target_pixel=target_pixel,
                    target_3d=target_3d_viz,
                )

            # =================================================================
            # 8. Execute Trajectory with Safety Buffer
            # =================================================================
            if traj is not None and len(traj) > 0:
                try:
                    path_arr = np.array(traj)
                    diffs = path_arr[1:] - path_arr[:-1]
                    dists = np.linalg.norm(diffs[:, :2], axis=1)
                    total_dist = np.sum(dists)

                    SAFE_DISTANCE = Config.SAFE_DISTANCE
                    target_dist = total_dist - SAFE_DISTANCE

                    if target_dist <= 0:
                        print_warning(
                            f"Target too close ({total_dist:.2f}m < {SAFE_DISTANCE}m). "
                            "Stopping here."
                        )
                    else:
                        current_dist = 0
                        cutoff_idx = len(path_arr) - 1

                        for i, d in enumerate(dists):
                            current_dist += d
                            if current_dist >= target_dist:
                                cutoff_idx = i + 1
                                break

                        truncated_traj = traj[: cutoff_idx + 1]

                        if len(truncated_traj) > 0:
                            final_pt = truncated_traj[-1]
                            new_goal = (final_pt[0], final_pt[1])
                            print_warning(
                                f"Truncating trajectory: stop {SAFE_DISTANCE}m early "
                                f"({target_dist:.2f}m / {total_dist:.2f}m). "
                                f"New Goal: {new_goal}"
                            )
                            self.robot.execute_trajectory(
                                truncated_traj, goal_local=new_goal
                            )
                        else:
                            self.robot.execute_trajectory(traj, goal_local=goal_2d)

                except Exception as e:
                    print_error(f"Error truncating trajectory: {e}")
                    self.robot.execute_trajectory(traj, goal_local=goal_2d)
            else:
                print_warning("Planner failed to find a path.")

            if strategic_stop:
                print_success("Mission Complete (Strategic Stop after final approach).")
                break

            self.current_step += 1

        print_success("VLN Task Completed or Stopped.")

    # =========================================================================
    # Panorama Frame Preparation
    # =========================================================================

    def _get_panorama_frames(self) -> List[Dict]:
        """Load panorama images for the current step (4-view: F/R/B/L)."""
        frames = []
        view_defs = [
            {
                "angle": 360,
                "label": f"Image 1: The current FORWARD view (Step {self.current_step}).",
                "file": "view_0.png",
            },
            {
                "angle": 90,
                "label": f"Image 2: The view after turning 90 deg to the RIGHT (Step {self.current_step}).",
                "file": "view_90.png",
            },
            {
                "angle": 180,
                "label": f"Image 3: The view directly BEHIND (180 deg turn) (Step {self.current_step}).",
                "file": "view_180.png",
            },
            {
                "angle": 270,
                "label": f"Image 4: The view after turning 90 deg to the LEFT (Step {self.current_step}).",
                "file": "view_270.png",
            },
        ]

        step_dir = os.path.join(Config.PANORAMA_DIR, f"step{self.current_step}")
        for view in view_defs:
            img_path = os.path.join(step_dir, view["file"])
            if os.path.exists(img_path):
                frames.append(
                    {
                        "angle": view["angle"],
                        "label": view["label"],
                        "img_base64": img_to_base64(img_path),
                    }
                )
            else:
                print_warning(
                    f"Panorama view missing: {view['file']}, using placeholder."
                )

        return frames
