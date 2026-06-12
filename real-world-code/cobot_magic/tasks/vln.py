"""
Vision-Language Navigation (VLN) task for Cobot Magic.

Text-instruction navigation on the same primitives as ObjectNav (panorama scan
-> strategic direction decision -> base rotation -> tactical bbox detection ->
dynamic-replan navigation -> TODO update). The strategic / tactical reasoning
is driven purely by the natural-language instruction.

Shares the hardware / motion primitives on ``RobotController`` and the VLM
orchestration on ``LaViRANavigationAPI``.
"""
from __future__ import annotations

import datetime
import os
import time
from typing import Any, Dict, List

from ai_client import LaViRAVisionClient
from config import Config
from robot.navigation_api import LaViRANavigationAPI
from tasks import register_task
from utils import (
    image_to_base64,
    is_all_tasks_completed,
    mark_first_incomplete_task_completed,
    print_info,
    print_success,
)

# 8-slot panorama view labels (front=0; back=4 is never captured).
VIEW_LABELS = [
    "FRONT (0°)", "LEFT-FRONT (45°)", "LEFT (90°)", "LEFT-BACK (135°)",
    "BACK (180°) - NO IMAGE", "RIGHT-BACK (-135°)", "RIGHT (-90°)", "RIGHT-FRONT (-45°)",
]

# Strategic turn direction -> base rotation angle (degrees).
DIRECTION_ANGLES = {
    "front": 0, "left_front": 45, "left": 90, "left_back": 135,
    "right_back": -135, "right": -90, "right_front": -45,
}


@register_task("vln")
class VLNTask:
    """Text-instruction vision-language navigation."""

    def __init__(self, robot, instruction: str, max_cycles: int = 10) -> None:
        self.robot = robot
        self.instruction = instruction or "navigate forward"
        self.max_cycles = max_cycles

        self.client = LaViRAVisionClient()
        self.nav_api = LaViRANavigationAPI(self.client)

        self.visited_targets: List[Dict[str, Any]] = []
        self.current_todo_list: str = ""
        self.is_task_running = True

    def stop_task(self) -> None:
        """Stop the navigation loop."""
        print_info("Stopping VLN Task...")
        self.is_task_running = False

    # ------------------------------------------------------------------ #
    # Driver loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Run text-instruction navigation across up to ``max_cycles`` cycles."""
        self.is_task_running = True
        print_info(f"Starting VLN Task: '{self.instruction}'")

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        session_root = Config.SESSION_DIR or os.path.join(Config.OUTPUT_ROOT, timestamp)
        os.makedirs(session_root, exist_ok=True)

        try:
            for i in range(self.max_cycles):
                if not self.is_task_running:
                    break
                done = self._run_cycle(session_root, cycle_index=i)
                if done or not self.is_task_running:
                    break
                time.sleep(1.0)
        finally:
            print_success("VLN Task Completed or Stopped.")

    # ------------------------------------------------------------------ #
    # Single navigation cycle
    # ------------------------------------------------------------------ #
    def _run_cycle(self, session_root: str, cycle_index: int = 0) -> bool:
        import cv2

        cycle_dir = os.path.join(session_root, f"cycle_{cycle_index:02d}")
        os.makedirs(cycle_dir, exist_ok=True)

        print_info("=" * 50)
        print_info(f"Cycle {cycle_index} | Instruction: '{self.instruction}'")

        # Step 1: Panorama (8 slots; front=0, back=4=None). Start the arm-reset
        # thread immediately so the gimbal returns home in parallel.
        pan_rgb, pan_depth, arm_reset_thread = self.robot.collect_panoramic_images()
        arm_reset_thread.start()

        panorama_dir = os.path.join(cycle_dir, "panorama")
        os.makedirs(panorama_dir, exist_ok=True)
        for i, img in enumerate(pan_rgb):
            if img is not None:
                cv2.imwrite(os.path.join(panorama_dir, f"pano_{i}.jpg"), img)

        panorama_frames: List[Dict[str, Any]] = []
        for rgb, label in zip(pan_rgb, VIEW_LABELS):
            if rgb is not None:
                panorama_frames.append({
                    "img_base64": image_to_base64(rgb),
                    "label": label,
                })

        # Step 2: Strategic decision (LA model).
        strategy = self.nav_api.decide_direction(
            panorama_frames, self.instruction, self.visited_targets,
            self.current_todo_list, cycle_dir=cycle_dir)

        self.current_todo_list = strategy.get("updated_todo_list", self.current_todo_list)
        if strategy.get("action") == "STOP":
            print_success("Task complete!")
            if arm_reset_thread.is_alive():
                arm_reset_thread.join()
            return True

        # Step 3: Rotate toward the chosen direction.
        direction = strategy.get("turn_direction", "front")
        landmark = strategy.get("expected_landmark", "target ahead")
        self.visited_targets.append({
            "step": len(self.visited_targets) + 1,
            "direction": direction,
            "target": landmark,
            "description": f"Turned {direction} toward {landmark}",
        })

        if direction in DIRECTION_ANGLES:
            self.robot.rotate_angle(DIRECTION_ANGLES[direction])
        time.sleep(2.0)

        if arm_reset_thread.is_alive():
            arm_reset_thread.join()

        # Step 4: Capture the post-rotation front view + detect the bbox.
        curr_rgb = None
        for _ in range(10):
            with self.robot.image_lock:
                if self.robot.current_front_rgb is not None:
                    curr_rgb = self.robot.current_front_rgb.copy()
            if curr_rgb is not None:
                break
            time.sleep(0.2)

        if curr_rgb is None:
            return False
        cv2.imwrite(os.path.join(cycle_dir, "raw_rgb.jpg"), curr_rgb)

        progress_analysis = strategy.get("progress_analysis", "")
        bbox_result = self.nav_api.query_llm_bbox(
            curr_rgb, self.instruction, progress_analysis,
            self.visited_targets, landmark,
            self.robot.rgb_width, self.robot.rgb_height,
            cycle_dir=cycle_dir)

        nav_u, nav_v = self.robot.rgb_width / 2, self.robot.rgb_height / 2
        if bbox_result:
            bbox = bbox_result.get("bbox_2d", [])
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                nav_u = (x1 + x2) / 2
                nav_v = y2
                debug = curr_rgb.copy()
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
                bbox_dir = os.path.join(cycle_dir, "bbox")
                os.makedirs(bbox_dir, exist_ok=True)
                cv2.imwrite(os.path.join(bbox_dir, "bbox.jpg"), debug)

        # Step 5: Navigate (continuous dynamic-replan).
        arrived = self.robot.navigate_dynamic(nav_u, nav_v, cycle_dir=cycle_dir)
        if arrived:
            self.current_todo_list = mark_first_incomplete_task_completed(
                self.current_todo_list)
        if is_all_tasks_completed(self.current_todo_list):
            return True
        return False
