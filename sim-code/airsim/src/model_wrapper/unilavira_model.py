from openai import OpenAI
import os
from PIL import Image
import numpy as np
import io
import base64

from utils.logger import logger
from src.model_wrapper.base_model import BaseModelWrapper
from src.model_wrapper.utils.travel_util import *
from src.vlnce_src.dino_monitor_online import DinoMonitor

from vlnce_baselines.utils.map_utils import *

from scipy.spatial.transform import Rotation as R

import re
import json
import math
from collections import defaultdict


class OpenAIVisionClient:
    
    def __init__(self, api_key=None, base_url=None, model_name="gpt-4-vision-preview", 
                 secondary_model_name=None, secondary_api_key=None, secondary_base_url=None):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
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
            'primary': {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            },
            'secondary': {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        }
    
    def generate(self, messages, images=None, max_new_tokens=1024, temperature=0.7, use_secondary=False, **kwargs):
        import time

        t = time.time()
        
        if use_secondary and self.secondary_client:
            client = self.secondary_client
            model_name = self.secondary_model_name
            stats_key = 'secondary'
        else:
            client = self.client
            model_name = self.model_name
            stats_key = 'primary'

        print(f"{'='*20} {model_name} start {'='*20}")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            
            
            self.stats[stats_key]['calls'] += 1
            if hasattr(response, 'usage') and response.usage:
                print(f"API Call usage - {response.usage}")
                self.stats[stats_key]['input_tokens'] += response.usage.prompt_tokens or 0
                self.stats[stats_key]['output_tokens'] += response.usage.completion_tokens or 0
                self.stats[stats_key]['total_tokens'] += response.usage.total_tokens or 0
                
                print(f"{stats_key.upper()} model usage - Input: {response.usage.prompt_tokens}, "
                           f"Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
            
            print(f'Generating uses {time.time()-t} seconds.')
            import sys
            sys.stdout.flush()
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"API Call error with {'secondary' if use_secondary else 'primary'} model ({model_name}): {e}")
            print('Forcing retry..')
            import time
            time.sleep(30)
            return self.generate(messages, images, max_new_tokens, temperature, use_secondary, **kwargs)
            # return "Error: Failed to get response from API"

class ZeroShotVlnEvaluatorMP(BaseModelWrapper):
    # def __init__(self, config, r2r, segment_module=None, mapping_module=None) -> None:
    def __init__(self, model_args=None, data_args=None, data_habitat=None) -> None:
        super().__init__()

        
        self.dino_moinitor = None

        self.view_definitions = [
            {"idx": 1, "name": "front", "angle_deg": 0,    "desc": "FORWARD view (body +X)."},
            {"idx": 2, "name": "left",  "angle_deg": -90,  "desc": "LEFT view (yaw -90° in body frame, body -Y)."},
            {"idx": 3, "name": "right", "angle_deg": 90,   "desc": "RIGHT view (yaw +90° in body frame, body +Y)."},
            {"idx": 4, "name": "rear",  "angle_deg": 180,  "desc": "REAR view (yaw 180° in body frame, body -X)."},
            {"idx": 5, "name": "down",  "angle_deg": None, "desc": "DOWNWARD view (camera points along body +Z / global +Z Down)."},
        ]

        
        api_key = os.environ.get('VA_API_KEY', '')
        base_url = os.environ.get('VA_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        model_name = os.environ.get('VA_MODEL_NAME', 'qwen3.5-27b')
        secondary_api_key = os.environ.get('LA_API_KEY', '')
        secondary_base_url = os.environ.get('LA_BASE_URL', 'https://yunwu.ai/v1')
        secondary_model_name = os.environ.get('LA_MODEL_NAME', 'gemini-3.5-flash')
        self.model = OpenAIVisionClient(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            secondary_model_name=secondary_model_name,
            secondary_api_key=secondary_api_key,
            secondary_base_url=secondary_base_url
        )

        
        self.history_waypoint = []
        self.history_image = []
        self.just_backtracked = False  
        self.backtrack_failed_context = None  

        
        
        self._ep_state = defaultdict(lambda: {
            "current_step": 0,
            "instruction": None,
            "todo_list": "",                
            "llm_replies": [],              
        })

    def eval(self):
        pass

    def generate_todo_list(self, instruction, rgb_list):
        print("Generating initial TODO list...")

        content = [
            {
                "type": "text",
                "text": f'Instruction: "{instruction}"\n\n'
                        "The images provided are multi-directional views from the starting position."
            }
        ]

        view_names = ["front", "left", "right", "rear", "down"]
        for i, rgb_dict in enumerate(rgb_list):
            rgb_base64 = self.img_to_base64(rgb_dict["rgb"])
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{rgb_base64}"}
            })
            content.append({"type": "text", "text": f"Image {i+1}: {view_names[i].upper()} view"})

        prompt = """Your task is to create a dynamic checklist (TODO list) to complete the instruction based on the visual context.

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
        todo = self.model.generate(
            messages=messages,
            max_new_tokens=8192,
            temperature=0.1,
            use_secondary=True
        )

        print(f"Initial TODO List:\n{todo}")
        return todo

    def query_llm(self, instruction, position_init, orientation_init, position, orientation, distance_depth, todo_list=""):  

        content = [{
  "type": "text",
  "text": """
You are analyzing 5 views to help a drone navigate in a global NED (North-East-Down) coordinate system.
The expected navigation strategy is: first fly 10-16 meters away from the obstacle below, then maintain this distance and move as straight as possible toward the target direction . 
During the flight, avoid obstacles and maintain safety clearance by adjusting heading and altitude as needed

Coordinate Frames:
    Body frame (attached to the drone): +X = forward (nose), +Y = right, +Z = down. Right-hand rule. Using Body frame for: camera/view directions.
    Global frame: NED. +X = North, +Y = East, +Z = Down. Use GLOBAL NED for: initial position, current position, initial yaw, current yaw, and the output 'direction'.
    yaw = 0° points to +X (North) and positive yaw turns from +X toward +Y (turning RIGHT)
"""
}]

        for i, rgb_dict in enumerate(self.rgb_list):
            v = self.view_definitions[i]
            view_name = v["name"]
            angle_deg = v["angle_deg"]
            angle_str = "N/A (downward)" if angle_deg is None else f"{angle_deg}° (relative to body +X)"

            # RGB
            rgb_img = rgb_dict["rgb"]
            rgb_base64 = self.img_to_base64(rgb_img)

            # obstacle distance (meters) for this view
            obstacle_distance = float(distance_depth[i])

            content.append({
                "type": "text",
                "text": (
                    f"Image {v['idx']}A (RGB) — {v['desc']}\n"
                    f"Relative view yaw: {angle_str}.\n"
                    f"Obstacle distance in the {view_name} direction: approximately {obstacle_distance:.1f} meters.This is VERY important"
                )
            })
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{rgb_base64}"}})

        todo_section = f"""
Current TODO List:
{todo_list}

TODO Update Instructions:
    - Check the current views and progress, then update the TODO list accordingly
    - Mark completed steps with [x] and add brief result notes
    - Keep uncompleted steps as [ ]
    - Add new steps ONLY when you see the target.
""" if todo_list else ""

        if self.history_image:
            content.append({"type": "text", "text": f"\nHistory observations ({len(self.history_image)} images, oldest→newest):"})
            for hi, h_img in enumerate(self.history_image):
                h_base64 = self.img_to_base64(h_img)
                content.append({"type": "text", "text": f"History image {hi+1} (step {hi+1}):"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{h_base64}"}})

        
        backtrack_notice = ""
        if self.just_backtracked and self.backtrack_failed_context:
            failed_wps = self.backtrack_failed_context.get("waypoints", [])
            failed_imgs = self.backtrack_failed_context.get("images", [])
            backtrack_notice = f"""
IMPORTANT: You just executed a BACKTRACK in the previous step. You have returned to an earlier position.
The following path was abandoned (failed waypoints): {failed_wps}
Now re-evaluate the situation from the current views. Do NOT repeat the same failed direction. Choose a different approach.
"""
            
            if failed_imgs:
                content.append({"type": "text", "text": f"Failed path images ({len(failed_imgs)} images, these are views from the abandoned path):"})
                for fi, f_img in enumerate(failed_imgs):
                    f_base64 = self.img_to_base64(f_img)
                    content.append({"type": "text", "text": f"Failed path image {fi+1}:"})
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{f_base64}"}})
            print(f"[BACKTRACK] Inserted a backtrack hint into the query_llm prompt with {len(failed_imgs)} failed-path images")

        prompt = f"""Navigation task:{instruction}
Note: In the instruction, "you" refers to the drone at the INITIAL state (position = {position_init}, yaw = {orientation_init}°). Any relative angle (yaw_rel) is defined with respect to this INITIAL pose, not the current pose.

Current situation (GLOBAL NED):
    Initial position: {position_init}
    Initial yaw (at episode start): {orientation_init}°
    Current position: {position}
    Current yaw: {orientation}°

History waypoints (GLOBAL NED, oldest→newest): {self.history_waypoint}
{todo_section}{backtrack_notice}
Tasks:
    1) Target visibility: Do you see the target object described in the Instruction in ANY RGB view?
    2) Movement direction: Choose the best forward moving direction as an ABSOLUTE yaw angle in GLOBAL NED degrees.
        yaw_goal_global = wrap_to[-180,180]( Initial yaw + yaw_rel )
        direction MUST be yaw_goal_global.
    3) Action: Decide whether to adjust altitude (ascend/descend), move horizontally, or backtrack to a previous position.
        Space ABOVE is always safe (no ceiling)
        Keep a safety margin: if the nearest obstacle distance is ≥ 10 m, treat it as safe to proceed.
        If you are stuck, going in circles, or heading in a clearly wrong direction, choose "backtrack" to return to an earlier position and try a different route.
    4) TODO update: Update the TODO list based on current observations and progress.

Output:
Return ONLY a valid JSON object. No markdown, no code fences, no extra keys, no explanations.
    {{
        "target": <True/False>,
        "direction": <float>,
        "decision": <"ascend", "descend", "move", or "backtrack">,
        "updated_todo_list": <"full updated markdown checklist string, or empty string if no todo list">,
        "reasoning":<why you make the "decision">
    }}

Guidelines:
    In most cases, you will NOT see the target object; therefore, you should navigate primarily using position_init and the derived global goal heading yaw_goal_global
    If the obstacle distances in all surrounding directions are large (i.e., the environment is clear), you MUST gradually descend to a lower altitude—even if you may need to climb again in a later step.
    You MUST pay attention to the obstacle distances when making navigation decisions.
    Unless the obstacle is very tall, an altitude of 10-16 meters is usually sufficient.
    The TODO list is for progress tracking only; never let it override the navigation direction derived from your current observations and goal heading.
    Use obstacle distances as the primary safety constraint, and use images only for landmark recognition and target identification.
    Choose "backtrack" ONLY when you believe you are stuck or have taken a clearly wrong path. Do not backtrack casually.
"""

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=8192,
            temperature=0.7,
            use_secondary=True
        )

        print("LLM Output:")
        print(output_text)
        return output_text

    def query_image(self, rgb_direction, depth_direction, distance_depth, current_action, idx) -> str:

        
        if len(self.history_image) >= 10:
            self.history_image.pop(0)
        self.history_image.append(rgb_direction)
        print(f"[history_image] image count:{len(self.history_image)}")

        body = {
            "front": float(distance_depth[0]),
            "left":  float(distance_depth[1]),
            "right": float(distance_depth[2]),
            "rear":  float(distance_depth[3]),
            "down":  float(distance_depth[4]),
        }

        if idx == 0 or idx == 4:
            rel = {"forward": body["front"], "left": body["left"],  "right": body["right"], "back": body["rear"],  "down": body["down"]}
        elif idx == 1:
            rel = {"forward": body["left"],  "left": body["rear"],  "right": body["front"], "back": body["right"], "down": body["down"]}
        elif idx == 2:
            rel = {"forward": body["right"], "left": body["front"], "right": body["rear"],  "back": body["left"],  "down": body["down"]}
        elif idx == 3:
            rel = {"forward": body["rear"],  "left": body["right"], "right": body["left"],  "back": body["front"], "down": body["down"]}
        else:
            logger.warning(f"Unexpected view index: {idx}")
            rel = None
        print("QUERY_IMAGE ",rel["down"])
        v = self.view_definitions[idx]
        view_name = v["name"]
        angle_deg = v["angle_deg"]
        angle_str = "N/A (downward)" if angle_deg is None else f"{angle_deg}° (relative to body +X)"

        rgb_base64 = self.img_to_base64(rgb_direction)

        depth_base64 = self.img_to_base64(depth_direction)

        content = [{
        "type": "text",
        "text": """
You are a UAV planner. 
You will receive an RGB–Depth pair for the relevant view, obstacle distances in five BODY directions, and the current action type.
Your job is to output either a forward travel distance (if action is "move") or a vertical altitude change 
Note:for altitude, negative = ascend, positive = descend.
"""
        }]

        content.append({
            "type": "text",
            "text": (
                f"Image {v['idx']}A (RGB) — {v['desc']}\n"
                f"Relative view yaw: {angle_str}.\n"
                f"Pairing: This RGB corresponds to the Depth in Image {v['idx']}B for the SAME view."
            )
        })
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{rgb_base64}"}})

        content.append({
            "type": "text",
            "text": (
                f"Image {v['idx']}B (Depth) — SAME view as Image {v['idx']}A ({view_name}).\n"
                "Depth is grayscale: black=near(0m), white=far(100m), linear mapping.\n"
                "Use it to infer free space / nearest obstacle along this view direction.\n"
            )
        })
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{depth_base64}"}})
        
        prompt = f"""
Current situation:
    action:You need to {current_action}.
    Obstacle distances (m): front={rel['forward']:.1f}, left={rel['left']:.1f}, right={rel['right']:.1f}, rear={rel['back']:.1f}, down={rel['down']:.1f}.

Tasks:
    If action is "move": output a forward travel distance; distance is always positive.
    If action is "descend": output a vertical distance in meters; the value must be positive. Larger positive values mean a greater descent.
    If action is "ascend": output a vertical distance in meters; the value must be negative. More negative values mean a greater ascent.

Output:
Return ONLY a valid JSON object. No markdown, no code fences, no extra keys, no explanations.
    {{
        "distance": <float(negative=ascend, positive=descend)>,
        "reason": <why you choose this distance>
    }}
Guidelines:
    You MUST choose distance from {-5, -10, -15, 5, 10, 15}.
    If the forward path is clear (forward obstacle distance is large), prefer a longer move (15 m over 10 m over 5 m).
    Keep a safety margin: if the nearest obstacle distance is ≥ 10 m, treat it as safe to proceed.
    Space ABOVE is always safe (no ceiling).
    Unless the obstacle is very tall, keeping 10-16 m clearance from obstacles below is usually sufficient.
    When action is "move", if obstacle distance - distance < 0, considering a shorter distance.
"""
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=4096,
            temperature=0.7,
            use_secondary=False
        )

        print("LLM Output:")
        print(output_text)
        return output_text

    
    def img_to_base64(self, img) -> str:
        
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64

    def _get_instruction_from_episode(self, ep_frames, fallback_text=None):
        try:
            if len(ep_frames) > 0:
                f0 = ep_frames[0]
                if 'instruction' in f0:
                    ins = f0['instruction']
                    if isinstance(ins, dict) and 'text' in ins:
                        return ins['text']
                    if isinstance(ins, str):
                        return ins
            if len(ep_frames) > 0:
                fl = ep_frames[-1]
                if 'instruction' in fl:
                    ins = fl['instruction']
                    if isinstance(ins, dict) and 'text' in ins:
                        return ins['text']
                    if isinstance(ins, str):
                        return ins
        except Exception:
            pass
        return fallback_text or "Find the described target."

    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        bs = len(episodes)
        inputs = []
        for env_idx in range(bs):
            ep = episodes[env_idx]
            last_obs = ep[-1]
            
            rgb_list, depth_list = self.get_panorama([last_obs], step=len(ep))
            
            st = self._ep_state[env_idx]

            
            if len(ep) == 1:
                st["current_step"] = 0
                st["instruction"] = None
                st["todo_list"] = ""
                st["llm_replies"] = []
                
                self.history_image = []
                self.history_waypoint = []
                self.just_backtracked = False
                self.backtrack_failed_context = None

            st["current_step"] = len(ep)
            if st["instruction"] is None:
                st["instruction"] = self._get_instruction_from_episode(ep, fallback_text=None)
            inputs.append({
                "last_obs": last_obs,
                "panorama": (rgb_list, depth_list),
                "target_pos": target_positions[env_idx]
            })
        rot_to_targets = [None] * bs
        return inputs, rot_to_targets
    
    def get_reply_from_llm(self, content, yaw_now):
        try:
            
            content_cleaned = re.sub(r'```json\s*', '', content)
            content_cleaned = re.sub(r'```\s*', '', content_cleaned)

            
            json_match = re.search(
                r'\{\s*"[^"]+"\s*:\s*[^{}]+(?:,\s*"[^"]+"\s*:\s*[^{}]+)*\s*\}',
                content_cleaned, re.DOTALL
            )

            if not json_match:
                logger.error(f"Failed to extract JSON from LLM output: {content}")
                
                return self.rgb_list[0]["rgb"], self.depth_list[0]["depth"], "move", 0, 0, ""

            
            json_str = json_match.group().strip()
            response_json = json.loads(json_str)

            
            target_visible = bool(response_json.get("target", False))
            direction = float(response_json.get("direction", 0.0))   # GLOBAL desired heading
            current_action = response_json.get("decision", "move")
            updated_todo_list = response_json.get("updated_todo_list", "")

            
            if current_action == "backtrack":
                print("[BACKTRACK] LLM decision: backtrack, skipping direction/view computation")
                return None, None, "backtrack", 0, 0, updated_todo_list

            
            idx = self.from_direction_calculate_idx(direction_global_deg=direction, yaw_now_deg=float(yaw_now))

            
            
            def wrap180(a: float) -> float:
                return (a + 180.0) % 360.0 - 180.0
            yaw_body = wrap180(direction - float(yaw_now))

            print("LLM response parsing result:")
            print(f"  - target visible: {target_visible}")
            print(f"  - global direction: {direction:.2f}°")
            print(f"  - current yaw: {float(yaw_now):.2f}°")
            print(f"  - body-relative yaw (yaw_body = direction - yaw_now): {yaw_body:.2f}°")
            print(f"  - decision: {current_action}")
            print(f"  - view index: {idx} ({self.view_definitions[idx]['name']})")
            # if updated_todo_list:
                

            
            rgb_direction = self.rgb_list[idx]["rgb"]
            depth_direction = self.depth_list[idx]["depth"]

            return rgb_direction, depth_direction, current_action, direction, idx, updated_todo_list

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"LLM output: {content}")
            return self.rgb_list[0]["rgb"], self.depth_list[0]["depth"], "move", 0, 0, ""
        except Exception as e:
            logger.error(f"Error while parsing LLM response: {e}")
            logger.error(f"LLM output: {content}")
            return self.rgb_list[0]["rgb"], self.depth_list[0]["depth"], "move", 0, 0, ""

    def parse_distance_from_llm(self, content, default_distance=5.0):
        try:
            
            content_cleaned = re.sub(r'```json\s*', '', content)
            content_cleaned = re.sub(r'```\s*', '', content_cleaned)

            
            json_match = re.search(r'\{\s*"[^"]+"\s*:\s*[^{}]+(?:,\s*"[^"]+"\s*:\s*[^{}]+)*\s*\}', content_cleaned, re.DOTALL)

            if not json_match:
                logger.error(f"Failed to extract JSON from LLM output: {content}")
                logger.warning(f"using default distance: {default_distance}m")
                return default_distance

            
            json_str = json_match.group().strip()
            response_json = json.loads(json_str)

            
            distance = float(response_json.get("distance", default_distance))

            
            if distance < -100 or distance > 100:
                logger.warning(f"invalid distance value: {distance}m, using default value: {default_distance}m")
                distance = default_distance

            
            print(f"forward distance: {distance}m")

            return distance

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"LLM output: {content}")
            logger.warning(f"using default distance: {default_distance}m")
            return default_distance
        except (ValueError, TypeError) as e:
            logger.error(f"failed to cast distance value: {e}")
            logger.error(f"LLM output: {content}")
            logger.warning(f"using default distance: {default_distance}m")
            return default_distance
        except Exception as e:
            logger.error(f"error while parsing distance: {e}")
            logger.error(f"LLM output: {content}")
            logger.warning(f"using default distance: {default_distance}m")
            return default_distance

    def from_direction_calculate_idx(self, direction_global_deg: float, yaw_now_deg: float) -> int:
        # wrap angle to [-180, 180)
        def wrap180(a: float) -> float:
            a = (a + 180.0) % 360.0 - 180.0
            return a

        yaw_body = wrap180(direction_global_deg - yaw_now_deg)

        # Sector decision (front/left/right/rear)
        if -45.0 <= yaw_body < 45.0:
            return 0  # front
        elif -135.0 <= yaw_body < -45.0:
            return 1  # left
        elif 45.0 <= yaw_body < 135.0:
            return 2  # right
        else:
            return 3  # rear


    def run(self, inputs, episodes, rot_to_targets=None):
        bs = len(episodes)
        all_waypoints = []
        print("----------------Starting Uni-LaViRA navigation----------------")
        
        for env_idx in range(bs):
            ep = episodes[env_idx]

            
            first_obs = ep[0]
            position_init = first_obs['sensors']['state']['position']  # [x, y, z]
            orientation_init = first_obs['sensors']['state']['orientation']  
            yaw_init = self.get_yaw_from_quaternion(orientation_init)
            print(f"Episode {env_idx} - instruction: {self._ep_state[env_idx]['instruction']}")
            print(f"Episode {env_idx} - initial position: {position_init}, initial yaw: {yaw_init}")


            
            last_obs = inputs[env_idx]['last_obs']
            position = last_obs['sensors']['state']['position']  
            orientation = last_obs['sensors']['state']['orientation']  
            yaw_now = self.get_yaw_from_quaternion(orientation)
            print(f"Episode {env_idx} - current position: {position}, current yaw: {yaw_now}")
            
            self.rgb_list, self.depth_list = inputs[env_idx]['panorama']

            
            distance_depth = self.get_distance_frome_depth()  
            print(f"obstacle distances by direction (front/left/right/rear/down): {distance_depth}")

            
            instruction = self._ep_state[env_idx]['instruction']
            st = self._ep_state[env_idx]

            
            if st["current_step"] == 1:
                st["todo_list"] = self.generate_todo_list(instruction, self.rgb_list)

            todo_list = st["todo_list"]

            
            content = self.query_llm(instruction, position_init, yaw_init, position, yaw_now, distance_depth, todo_list)

            
            rgb_direction, depth_direction, current_action, direction, idx, updated_todo_list = self.get_reply_from_llm(content, yaw_now)

            
            if updated_todo_list:
                st["todo_list"] = updated_todo_list

            
            if current_action == "backtrack":
                wps = self._generate_backtrack_waypoints()
                self.just_backtracked = True

                
                step = self._ep_state[env_idx]['current_step']
                step_text = f"\n########################step_{step}########################\n"
                step_text += f"#####query_llm###########\n{content}\n"
                step_text += f"#########BACKTRACK (skipped query_image)############\n"
                step_text += f"backtrack waypoints: {wps}\n"
                self._ep_state[env_idx]['llm_replies'].append(step_text)

            else:
                self.just_backtracked = False

                
                query_image_reply = self.query_image(rgb_direction, depth_direction, distance_depth, current_action, idx)
                distance = self.parse_distance_from_llm(query_image_reply)

                
                step = self._ep_state[env_idx]['current_step']
                step_text = f"\n########################step_{step}########################\n"
                
                step_text += f"#####query_llm###########\n{content}\n"
                step_text += f"#########query_image############\n{query_image_reply}\n"
                self._ep_state[env_idx]['llm_replies'].append(step_text)
                
                wps = self._generate_single_waypoint_list(direction, distance, position, current_action)

            
            all_waypoints.append(wps)

            print(f"Episode {env_idx} - generated {len(wps)} waypoints")

        print(f"Generated waypoints for {bs} episodes")
        return all_waypoints

    def pop_llm_replies(self, env_idx):
        replies = self._ep_state[env_idx].get('llm_replies', [])
        self._ep_state[env_idx]['llm_replies'] = []
        return replies

    def _generate_backtrack_waypoints(self):
        if len(self.history_waypoint) < 2:
            print("[BACKTRACK] Not enough history waypoints (<2); cannot backtrack, holding position")
            fallback = self.history_waypoint[-1] if self.history_waypoint else [0, 0, 0]
            return [fallback] * 5

        
        candidates = self.history_waypoint[:-1]  
        backtrack_wps = list(reversed(candidates[-5:]))  

        
        remove_count = len(backtrack_wps) + 1  
        keep_count = len(self.history_waypoint) - remove_count

        
        failed_waypoints = self.history_waypoint[keep_count:]
        failed_images = self.history_image[keep_count:] if keep_count < len(self.history_image) else []

        self.backtrack_failed_context = {
            "waypoints": failed_waypoints,
            "images": failed_images,
        }

        
        self.history_waypoint = self.history_waypoint[:keep_count]
        self.history_image = self.history_image[:min(keep_count, len(self.history_image))]

        print(f"[BACKTRACK] backtrack {len(backtrack_wps)}  historical decision points:")
        for i, wp in enumerate(backtrack_wps):
            print(f"  [{i+1}] {wp}")
        print(f"[BACKTRACK] history_waypoint remaining after backtrack {len(self.history_waypoint)} points")
        print(f"[BACKTRACK] history_image remaining after backtrack {len(self.history_image)} ")
        print(f"[BACKTRACK] failed-path storage: {len(failed_waypoints)} waypoints, {len(failed_images)} images")

        return backtrack_wps

    def get_distance_frome_depth(self):
        distance_depth = []

        for depth_dict in self.depth_list:
            depth_img = depth_dict["depth"]  

            
            h, w = depth_img.shape[:2]

            
            roi_ratio = 0.20
            center_h, center_w = h // 2, w // 2
            roi_h = max(1, int(h * roi_ratio))
            roi_w = max(1, int(w * roi_ratio))

            top = max(0, center_h - roi_h // 2)
            bottom = min(h, center_h + roi_h // 2)
            left = max(0, center_w - roi_w // 2)
            right = min(w, center_w + roi_w // 2)
    
            roi = depth_img[top:bottom, left:right].astype(np.float32).reshape(-1)


            
            closest_ratio = 0.05
            k = max(1, int(roi.size * closest_ratio))
            closest_vals = np.partition(roi, k - 1)[:k]
            closest_u8 = float(np.mean(closest_vals))

            
            closest_meters = (closest_u8 / 255.0) * 100.0
            distance_depth.append(closest_meters)

        return distance_depth

    def get_yaw_from_quaternion(self, orientation):
        # print(f"[DEBUG get_yaw_from_quaternion] Input quaternion [qx,qy,qz,qw]: {orientation}")
        
        r = R.from_quat(orientation)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        return yaw

    def _generate_single_waypoint_list(self, direction, distance, position, current_action):
        cur_xyz = position  # [x, y, z]
        rad = math.radians(direction)
        if current_action == "move":
            
            dx = distance * math.cos(rad)
            dy = distance * math.sin(rad)

            target_x = position[0] + dx
            target_y = position[1] + dy
            tgt_xy = [target_x, target_y]

            
            wps = self._waypoints_towards(cur_xyz, tgt_xy, num=5, max_step=10.0)

            print("[Waypoint generation] horizontal movement")
            print(f"[Waypoint generation] current position(x,y,z): {cur_xyz}")
            print(f"[Waypoint generation] target position(x,y): {tgt_xy}")
            print(f"[Waypoint generation] direction: {direction}°, distance: {distance}m")
            print(f"[Waypoint generation] generated waypoints: {wps}")
        
        else:
            
            target_z = position[2] + distance

            
            eps = 2
            dx = eps * math.cos(rad)
            dy = eps * math.sin(rad)

            wps = []
            for k in range(1, 6):  
                z_step = (target_z - position[2]) * k / 5.0
                wps.append([position[0] + dx, position[1] + dy, position[2] + z_step])

            print("[Waypoint generation] vertical obstacle avoidance")
            print(f"[Waypoint generation] current position(x,y,z): {cur_xyz}")
            print(f"[Waypoint generation] target altitude z: {target_z}")
            print(f"[Waypoint generation] direction: {direction}°, vertical distance: {distance}m")
            print(f"[Waypoint generation] generated waypoints: {wps}")

        
        self.history_waypoint.append(wps[-1])
        print(f"history waypoint count: {len(self.history_waypoint)}")

        return wps
        
    def get_panorama(self, obs, step: int):
        try:
            views_rgb = obs[0]['rgb']
            views_dep = obs[0]['depth']
        except Exception as e:
            raise KeyError(f"obs is missing 'rgb' or 'depth' keys: {e}")

        if len(views_rgb) < 5 or len(views_dep) < 5:
            raise ValueError(f"At least five camera streams are required(front/left/right/rear/down), but got rgb={len(views_rgb)} streams, depth={len(views_dep)} streams")

        
        rgb_front, rgb_left, rgb_right, rgb_back, rgb_down = views_rgb[0], views_rgb[1], views_rgb[2], views_rgb[3], views_rgb[4]
        dep_front, dep_left, dep_right, dep_back, dep_down = views_dep[0], views_dep[1], views_dep[2], views_dep[3], views_dep[4]

        
        if any(x is None for x in [rgb_front, rgb_left, rgb_right, rgb_back, rgb_down]):
            raise ValueError("RGB camera views are incomplete: found None")
        if any(x is None for x in [dep_front, dep_left, dep_right, dep_back, dep_down]):
            raise ValueError("Depth camera views are incomplete: found None")

        rgb_list = [
            {'rgb': rgb_front.copy(), 'angle':   0},
            {'rgb': rgb_left.copy(),  'angle': -90},
            {'rgb': rgb_right.copy(), 'angle':  90},
            {'rgb': rgb_back.copy(),  'angle': 180},
            {'rgb': rgb_down.copy(),  'angle': None},  # Downward camera treated as -90 degrees
        ]
        depth_list = [
            {'depth': dep_front.copy(), 'angle':   0},
            {'depth': dep_left.copy(),  'angle': -90},
            {'depth': dep_right.copy(), 'angle':  90},
            {'depth': dep_back.copy(),  'angle': 180},
            {'depth': dep_down.copy(),  'angle': None},  # Downward camera treated as -90 degrees
        ]

        return rgb_list, depth_list
    
    def _waypoints_towards(self, cur_xyz, tgt_xy, num=5, max_step=10.0):
        cx, cy, cz = cur_xyz      
        tx, ty = tgt_xy           

        
        dir_x, dir_y = tx - cx, ty - cy
        dist = math.hypot(dir_x, dir_y)

        if dist < 1e-3:
            return [[cx, cy, cz] for _ in range(num)]

        step = min(max_step, dist) / num
        ux, uy = dir_x / dist, dir_y / dist

        wps = []
        for k in range(1, num + 1):
            
            wps.append([cx + ux * step * k, cy + uy * step * k, cz])
        return wps


    def predict_done(self, episodes, object_infos):
        prediction_dones = []
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            prediction_done = self.dino_moinitor.get_dino_results(episodes[i], object_infos[i])
            prediction_dones.append(prediction_done)
        return prediction_dones
