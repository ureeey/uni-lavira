"""
VLN task for the Unitree Go1.

Ported verbatim (in behaviour) from the monolithic source ``tasks/vln.py``.
The inline OpenAI client construction and the inline strategic / tactical VLM
methods (``_decide_navigation``, ``_query_tactical_eyes``, ``_draw_bbox``,
``_generate_initial_todo``) have been relocated into
``robot.navigation_api.LaViRANavigationAPI`` and are now invoked through
``self.nav_api.*``.  The per-cycle ``run()`` loop, its geometry, the
``SAFE_DISTANCE = 1.5`` trajectory truncation, and the strategic-stop
``time.sleep(3.0)`` are unchanged.
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
    def __init__(self, robot, instruction: str):
        self.robot = robot
        self.instruction = instruction
        self.client = LaViRAVisionClient()
        self.nav_api = LaViRANavigationAPI(self.client)

        self.current_step = 1
        self.todo_list = ""
        self.visited_targets = []
        self.global_target = instruction
        self.is_task_running = True

    def stop_task(self):
        """Stop the current task execution loop."""
        print_info("Stopping VLN Task...")
        self.is_task_running = False

    def run(self):
        print_info(f"Starting VLN Task with instruction: {self.instruction}")
        self.is_task_running = True

        while self.robot.running and self.is_task_running:
            # 1. Capture Panorama
            if not self.robot.capture_all_directions(self.current_step):
                print_error("Failed to capture panorama images.")
                break

            panorama_frames = self._get_panorama_frames()

            # 2. Generate Initial TODO (Skipped - Merged into Navigation)
            if self.current_step == 1 and not self.todo_list:
                # self.todo_list = self.nav_api.generate_initial_todo(
                #     self.instruction, panorama_frames)
                print_info("Skipping separate Initial TODO generation. Will generate during first navigation step.")
                self.todo_list = "No TODO list yet. Please generate one."

            # 3. Decide Navigation (Strategic Brain)
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

            # Check for STOP
            action_decision = decision.get("action", "NAVIGATE")
            strategic_stop = (action_decision == "STOP")

            turn_direction = decision.get("turn_direction", "front").lower()

            # Validate direction
            allowed_directions = ['front', 'right', 'left']
            if turn_direction not in allowed_directions:
                print_warning(f"Unknown direction '{turn_direction}', defaulting to 'front'")
                turn_direction = 'front'

            strategic_reasoning = decision.get("reasoning", "")

            # Extract expected_landmark for visual targeting
            expected_landmark = decision.get("expected_landmark", "")

            # Use expected_landmark if available, otherwise fallback to instruction (handled in prompts.py if empty string passed)
            # We avoid using "Go {direction}" string as target to prevent confusion.
            visual_target = expected_landmark if expected_landmark and len(expected_landmark) > 2 else ""

            # Keep strategic_goal for logging or other uses if needed, but not for visual targeting prompt
            strategic_goal = f"Go {turn_direction}. {strategic_reasoning}"

            progress_analysis = decision.get("progress_analysis", "")

            # 4. Rotate Robot
            if turn_direction == 'left':
                self.robot.rotate_left(90)
            elif turn_direction == 'right':
                self.robot.rotate_right(90)

            # Update internal direction state (handled by robot controller mostly, but good to know)

            # 5. Tactical Eyes (Look at new view)
            # Capture current view after rotation
            if not self.robot.capture_current_image():
                print_warning("Failed to capture current view after rotation")
                continue

            # Force strategic_stop=False here to ensure Primary Model gives us a target BBox for final approach
            # Pass visual_target instead of strategic_goal string
            #
            # The source _query_tactical_eyes re-captured a fresh, stabilized
            # image before sending; preserve that behaviour here so the VA call
            # sees the identical frame.
            print_action("Capturing fresh image for Tactical Eyes (with stabilization)...")
            time.sleep(0.5)  # Stabilization delay
            self.robot.capture_current_image()
            tactical_img = self.robot.direction_images[self.robot.current_direction]['rgb']

            bbox_decision = self.nav_api.vision_action(
                img_np=tactical_img,
                instruction=self.instruction,
                global_target=self.global_target,
                strategic_goal=visual_target,
                strategic_stop=False,
                current_step=self.current_step,
                progress_analysis=progress_analysis,
            )

            action = bbox_decision.get("action", "NAVIGATE")
            bbox = bbox_decision.get("bbox_2d")

            # Logic Update:
            # If Strategic Brain (Secondary) says STOP, we don't stop immediately.
            # We check if Tactical Eyes (Primary) provided a bbox.
            # If yes, we move to that bbox (Final Approach) and THEN stop.

            should_stop_now = False

            if strategic_stop:
                if bbox and len(bbox) == 4:
                    print_info("Strategic Stop requested. Executing FINAL approach to target BBox before stopping.")
                    # Force action to NAVIGATE for this last step so we enter the execution block
                    action = "NAVIGATE"
                else:
                    print_success("Strategic Stop requested and no target BBox provided. Stopping immediately.")
                    should_stop_now = True
            elif action == "STOP":
                if bbox and len(bbox) == 4:
                    print_info("Tactical Eyes triggered STOP with visible target. Executing FINAL approach.")
                    action = "NAVIGATE"
                    # We mark strategic_stop=True here so that we break after this move
                    strategic_stop = True
                else:
                    print_success("Tactical Eyes triggered STOP (No target bbox). Stopping immediately.")
                    should_stop_now = True

            if should_stop_now:
                break
            target_pixel = None
            goal_2d = (1.5, 0.0)  # Default goal: 1.5m forward

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                # Target Pixel: Bottom Center (Sample 25% up from bottom)
                box_h = y2 - y1
                cx = int((x1 + x2) / 2)
                cy = int(y2 - 0.25 * box_h)
                target_pixel = (cx, cy)

                depth_val = self.robot.get_depth_value(cx, cy, self.robot.current_direction)

                if depth_val is not None:
                    # Unproject
                    cam_data = self.robot.camera_data[self.robot.current_camera]
                    fx = cam_data.get('fx')
                    cx_int = cam_data.get('cx')

                    if fx and cx_int:
                        # Z = depth
                        # X = (u - cx) * Z / fx
                        z_3d = depth_val
                        x_3d = (cx - cx_int) * z_3d / fx

                        # Convert Camera Frame to Robot Frame (Planar)
                        # Camera Frame: Z forward, X right, Y down
                        # Robot Frame: X forward, Y left
                        # Robot X = Camera Z
                        # Robot Y = -Camera X

                        goal_x = z_3d
                        goal_y = -x_3d

                        raw_forward = goal_x
                        raw_lateral = goal_y

                        goal_2d = (goal_x, goal_y)
                        print_info(f"Calculated Goal from BBox: {goal_2d} (Depth: {depth_val:.2f}m)")
                    else:
                        print_warning("Missing camera intrinsics for unprojection")
                else:
                    print_warning("Could not get depth for bbox target")

            # We need to get a plan from iPlanner
            rgb = self.robot.direction_images[self.robot.current_direction]['rgb']
            depth = self.robot.direction_images[self.robot.current_direction]['depth']

            # Let's get a plan!
            print_action(f"Requesting plan from iPlanner to {goal_2d}...")
            traj, fear = self.robot.planner_client.get_plan(rgb, depth, goal_2d)

            # Visualization
            if traj is not None:
                timestamp = time.strftime("%H%M%S")
                save_name = f"step{self.current_step}_plan_{timestamp}.jpg"
                # Pass target_3d for visualization text
                target_3d_viz = (goal_2d[0], goal_2d[1], 0.0)
                self.robot.project_trajectory_to_image(traj, rgb, save_name=save_name, target_pixel=target_pixel, target_3d=target_3d_viz)

            if traj is not None and len(traj) > 0:
                try:
                    path_arr = np.array(traj)
                    # Calculate distances between consecutive points
                    diffs = path_arr[1:] - path_arr[:-1]
                    dists = np.linalg.norm(diffs[:, :2], axis=1)
                    total_dist = np.sum(dists)

                    SAFE_DISTANCE = 1.5
                    target_dist = total_dist - SAFE_DISTANCE

                    if target_dist <= 0:
                        print_warning(f"Target is too close ({total_dist:.2f}m < {SAFE_DISTANCE}m). Stopping here.")
                        # We might want to execute a tiny movement or just stop?
                        # Let's not move if it's too close to avoid hitting it.
                        pass
                    else:
                        current_dist = 0
                        cutoff_idx = len(path_arr) - 1

                        for i, d in enumerate(dists):
                            current_dist += d
                            if current_dist >= target_dist:
                                cutoff_idx = i + 1
                                break

                        truncated_traj = traj[:cutoff_idx + 1]

                        if len(truncated_traj) > 0:
                            final_pt = truncated_traj[-1]
                            new_goal = (final_pt[0], final_pt[1])
                            print_warning(f"Truncating trajectory to stop {SAFE_DISTANCE}m early ({target_dist:.2f}m / {total_dist:.2f}m). New Goal: {new_goal}")
                            self.robot.execute_trajectory(truncated_traj, goal_local=new_goal)
                        else:
                            # Should not happen if target_dist > 0, but safety check
                            self.robot.execute_trajectory(traj, goal_local=goal_2d)

                except Exception as e:
                    print_error(f"Error truncating trajectory: {e}")
                    self.robot.execute_trajectory(traj, goal_local=goal_2d)
            else:
                print_warning("Planner failed to find a path. Moving forward blindly a bit?")
                # Fallback: simple forward command?
                # self.robot.move_forward(0.5)
                pass

            if strategic_stop:
                print_success("Mission Complete (Strategic Stop after final approach).")
                # Wait for robot to actually stop and settle
                time.sleep(3.0)
                break

            self.current_step += 1

        print_success("VLN Task Completed or Stopped.")
        # Ensure saver catches the final moments
        time.sleep(2.0)

    def _get_panorama_frames(self) -> List[Dict]:
        frames = []
        view_defs = [
            {"angle": 360, "label": f"Image 1: The current FORWARD view (Step {self.current_step}).", "file": "view_0.png"},
            {"angle": 90, "label": f"Image 2: The view after turning 90° to the RIGHT (Step {self.current_step}).", "file": "view_90.png"},
            {"angle": 270, "label": f"Image 4: The view after turning 90° to the LEFT (Step {self.current_step}).", "file": "view_270.png"},
        ]

        step_dir = os.path.join(Config.PANORAMA_DIR, f"step{self.current_step}")
        for view in view_defs:
            img_path = os.path.join(step_dir, view["file"])
            frames.append({
                "angle": view["angle"],
                "label": view["label"],
                "img_base64": img_to_base64(img_path),
            })

        return frames
