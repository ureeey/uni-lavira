import os
import pdb
import queue
import pickle
import copy
import gzip
import json
import time
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from fastdtw import fastdtw
from typing import List, Any, Dict, Optional, Callable
from collections import defaultdict
from skimage.morphology import binary_closing, disk, remove_small_objects

import torch
from torch import Tensor
from torchvision import transforms
from transformers import GenerationConfig

from qwen_vl_utils import process_vision_info

from habitat import Config, logger
from habitat_extensions.measures import NDTW
from habitat.core.simulator import Observations
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.environments import get_env_class
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.map.value_map import ValueMap
from vlnce_baselines.map.history_map import HistoryMap
from vlnce_baselines.map.direction_map import DirectionMap
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.Policy import FusionMapPolicy
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import gather_list_and_concat, get_device
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.common.constraints import ConstraintsMonitor
from vlnce_baselines.utils.constant import base_classes, map_channels

from pyinstrument import Profiler
import warnings
warnings.filterwarnings('ignore')


@baseline_registry.register_trainer(name="ZS-Evaluator-mp-ft")
class ZeroShotVlnEvaluatorMPForFinetuning(BaseTrainer):
    def __init__(self, config: Config, segment_module=None, mapping_module=None) -> None:
        super().__init__()
        
        self.device = get_device(config.TORCH_GPU_ID)
        torch.cuda.set_device(self.device)
        self.config = config
        self.map_args = config.MAP
        self.visualize = config.MAP.VISUALIZE
        self.resolution = config.MAP.MAP_RESOLUTION
        self.keyboard_control = config.KEYBOARD_CONTROL
        self.width = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        self.height = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
        self.max_step = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.map_shape = (config.MAP.MAP_SIZE_CM // self.resolution,
                          config.MAP.MAP_SIZE_CM // self.resolution)
        
        self.trans = transforms.Compose([transforms.ToPILImage(), 
                                         transforms.Resize(
                                             (self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH), 
                                             interpolation=Image.NEAREST)
                                        ])
        
        self.classes = []
        self.current_episode_id = None
        self.current_detections = None
        self.map_channels = map_channels
        self.floor = np.zeros(self.map_shape)
        self.one_step_floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversible = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)
        self.min_constraint_steps = config.EVAL.MIN_CONSTRAINT_STEPS
        self.max_constraint_steps = config.EVAL.MAX_CONSTRAINT_STEPS
        
        # New attributes for LLM integration and trajectory tracking
        self.llm_model = None
        self.rgb_history = []
        self.trajectory_rewards = []
        self.trajectory_history = []
        self.dist_reward = []
        self.save_rgb_history = True
        self.episode_rgb_dir = None
        self.success = 0
        self.tot = 0
        
        # New attributes for dynamic constraint querying
        self.completed_constraints = []  # List of completed constraints
        self.use_dynamic_constraints = False  # Flag to enable dynamic mode
        self.current_constraint_step = 0  # Track current constraint step
    
    def _set_eval_config(self) -> None:
        print("set eval configs")
        self.config.defrost()
        self.config.MAP.DEVICE = self.config.TORCH_GPU_ID
        self.config.MAP.HFOV = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        self.config.MAP.AGENT_HEIGHT = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
        self.config.MAP.NUM_ENVIRONMENTS = self.config.NUM_ENVIRONMENTS
        self.config.MAP.RESULTS_DIR = self.config.RESULTS_DIR
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.config.freeze()
        
    def _init_envs(self) -> None:
        print("start to initialize environments")

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
        )
        print(f"local rank: {self.local_rank}, num of episodes: {self.envs.number_of_episodes}")
        self.detected_classes = OrderedSet()
        print("initializing environments finished!")
        
    def _calculate_metric(self, infos: List):
        curr_eps = self.envs.current_episodes()
        info = infos[0]
        ep_id = curr_eps[0].episode_id
        gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
        pred_path = np.array(info['position']['position'])
        distances = np.array(info['position']['distance'])
        gt_length = distances[0]
        dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
        metric = {}
        metric['steps_taken'] = info['steps_taken']
        metric['distance_to_goal'] = distances[-1]
        metric['success'] = 1. if distances[-1] <= 3. else 0.
        metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
        metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
        # metric['collisions'] = info['collisions']['count'] / len(pred_path)
        metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
        metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
        metric['sdtw'] = metric['ndtw'] * metric['success']
        self.state_eps[ep_id] = metric
        self.success += metric['success']
        self.tot += 1
        print(self.state_eps[ep_id])
        
    def _initialize_policy(self) -> None:
        print("start to initialize policy")
        # print(type(self.device))
        self.segment_module = GroundedSAM(self.config, self.device)
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        
        self.value_map_module = ValueMap(self.config, self.mapping_module.map_shape, self.device)
        self.history_module = HistoryMap(self.config, self.mapping_module.map_shape)
        self.direction_module = DirectionMap(self.config, self.mapping_module.map_shape)
        self.policy = FusionMapPolicy(self.config, self.mapping_module.map_shape[0])
        self.policy.reset()
        
        self.constraints_monitor = ConstraintsMonitor(self.config, self.device)
        
    def _concat_obs(self, obs: Observations) -> np.ndarray:
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1) # (h, w, c)->(c, h, w)
        
        return state
    
    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        state = state.transpose(1, 2, 0)
        rgb = state[:, :, :3].astype(np.uint8) #[3, h, w]
        rgb = rgb[:,:,::-1] # RGB to BGR
        depth = state[:, :, 3:4] #[1, h, w]
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        env_frame_width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        
        sem_seg_pred = self._get_sem_pred(rgb) #[num_detected_classes, h, w]
        depth = self._preprocess_depth(depth, min_depth, max_depth) #[1, h, w]
        
        """
        ds: Downscaling factor
        args.env_frame_width = 640, args.frame_width = 160
        """
        ds = env_frame_width // self.map_args.FRAME_WIDTH # ds = 4
        if ds != 1:
            rgb = np.asarray(self.trans(rgb.astype(np.uint8))) # resize
            depth = depth[ds // 2::ds, ds // 2::ds] # down scaling start from 2, step=4
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2) # recover depth.shape to (height, width, 1)
        state = np.concatenate((rgb, depth, sem_seg_pred),axis=2).transpose(2, 0, 1) # (4+num_detected_classes, h, w)
        
        return state
        
    def _get_sem_pred(self, rgb: np.ndarray) -> np.ndarray:
        """
        mask.shape=[num_detected_classes, h, w]
        labels looks like: ["kitchen counter 0.69", "floor 0.37"]
        """
        masks, labels, annotated_images, self.current_detections = \
            self.segment_module.segment(rgb, classes=self.classes)
        self.mapping_module.rgb_vis = annotated_images
        assert len(masks) == len(labels), f"The number of masks not equal to the number of labels!"
        print("current step detected classes: ", labels)
        class_names = self._process_labels(labels)
        masks = self._process_masks(masks, class_names)
        
        return masks.transpose(1, 2, 0)
    
    def _process_labels(self, labels: List[str]) -> List:
        class_names = []
        for label in labels:
            class_name = " ".join(label.split(' ')[:-1])
            class_names.append(class_name)
            self.detected_classes.add(class_name)
        
        return class_names
        
    def _process_masks(self, masks: np.ndarray, labels: List[str]):
        """Since we are now handling the open-vocabulary semantic mapping problem,
        we need to maintain a mask tensor with dynamic channels. The idea is to combine
        all same class tensors into one tensor, then let the "detected_classes" to 
        record all classes without duplication. Finally we can use each class's index
        in the detected_classes to determine as it's channel in the mask tensor.
        The organization of mask is similar to chaplot's Sem_Exp, please refer to this link:
        https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/agents/utils/semantic_prediction.py#L41
        
        Args:
            masks (np.ndarray): shape:(c,h,w), each instance(even the same class) has one channel
            labels (List[str]): masks' corresponding labels. len(masks) = len(labels)

        Returns:
            final_masks (np.ndarray): each mask will find their channel in self.detected_classes.
            len(final_masks) = len(self.detected_classes)
        """
        if masks.shape != (0,):
            same_label_indexs = defaultdict(list)
            for idx, item in enumerate(labels):
                same_label_indexs[item].append(idx) #dict {class name: [idx]}
            combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
            for i, indexs in enumerate(same_label_indexs.values()):
                combined_mask[i] = np.sum(masks[indexs, ...], axis=0)
            
            idx = [self.detected_classes.index(label) for label in same_label_indexs.keys()]
            
            """
            max_idx = max(idx) + 1, attention: remember to add one becaure index start from 0
            init final masks as [max_idx + 1, h, w]; add not_a_category channel at last
            """
            final_masks = np.zeros((len(self.detected_classes), *masks.shape[1:]))
            final_masks[idx, ...] = combined_mask
        else:
            final_masks = np.zeros((len(self.detected_classes), self.height, self.width))
        
        return final_masks
    
    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        # Preprocesses a depth map by handling missing values, removing outliers, and scaling the depth values.
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99 # turn too far pixels to invalid
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0 # then turn all invalid pixels to vision_range(100)
        depth = min_depth * 100.0 + depth * max_depth * 100.0
        
        return depth
    
    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        concated_obs = self._concat_obs(obs)
        state = self._preprocess_state(concated_obs)
        
        return state # state.shape=(c,h,w)
    
    def _batch_obs(self, n_obs: List[Observations]) -> Tensor:
        n_states = [self._preprocess_obs(obs) for obs in n_obs]
        max_channels = max([len(state) for state in n_states])
        batch = np.stack([np.pad(state, 
                [(0, max_channels - state.shape[0]), 
                 (0, 0), 
                 (0, 0)], 
                mode='constant') 
         for state in n_states], axis=0)
        
        return torch.from_numpy(batch).to(self.device)
    
    def _random_policy(self):
        action = np.random.choice([
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ])
        
        return {"action": action}

    def _process_classes(self, base_class: List, target_class: List) -> List:
        for item in target_class:
            if item in base_class:
                base_class.remove(item)
        base_class.extend(target_class)
        
        return base_class
    
    def _check_destination(self, current_idx: int, sub_constraints: dict, llm_destination: str, decisions: dict) -> str:
        for idx in range(current_idx, len(sub_constraints)):
                constraints = sub_constraints[str(idx)]
                landmarks = decisions[str(idx)]["landmarks"]
                for constraint in constraints:
                    if constraint[0] == "direction constraint":
                        continue
                    else:
                        landmark = constraint[1]
                        for item in landmarks:
                            print(landmark, item)
                            if landmark in item:
                                choice = item[1]
                            else:
                                continue
                            print(choice, choice != "move away")
                            if choice != "move away":
                                return constraint[1]
                            else:
                                break
        else:
            return llm_destination
    
    def _query_llm_for_constraints(self, instruction: str, current_rgb: np.ndarray) -> Dict:
        """
        Query the LLM with instruction, current observation and RGB history to get constraints
        
        Args:
            instruction: Navigation instruction
            current_rgb: Current RGB observation
            
        Returns:
            Dictionary containing constraints in the same format as original llm_reply
        """
        if self.llm_model is None:
            raise ValueError("LLM model not set. Call set_llm_model() first.")
        
        # Prepare input for LLM
        rgb_inputs = self.rgb_history + [current_rgb]
        
        prompt = f"""
        [TASK DESCRIPTION]
Parse a navigation instruction delimited by triple quotes and your task is to perform the following actions:
1. Extract Destination: Understand the entire instruction and summarize a description of the destination. The description should be asentence containing landmark and room type. The description of the destination should not accurately describe the orientation and order.Here are examples about destination: "second room on the left" -> "room"(neglect order and direction); "between the bottom of the firststair and the console table in the entry way" -> "console table near entry way"(simplify description); "in front of the railing about halfwaybetween the two upstairs rooms" -> "railing near two upstair rooms"
2. Split instructions: Split the instruction into a series of sub-instructions according to the execution steps. Each sub-instruction containone landmark.
3. Infer agent's state constraints: Infer the state constraints that the agent should satisfy for each sub-instruction. There're thee constrainttypes: location constraints, direction constraints and object constraints. You need to select an appropriate constraint type and give thecorresponding constraint object. Direction constraint object has two types: left, right. Constraints can format as a tuple: (constraint type,constraint object)4. Make a decision: Analyze the landmarks, actions, and directions in each sub-instruction to determine how the agent should act. For alandmark, the agent has three options: approach, move away, or approach and then move away. For direction, the agent has three options:turn left, turn right, or go forward
[OUTPUT DEFINITION]
Provide your answer in JSON format with the following details:
1. use the following keys: destination, sub-instructions, state-constraints, decisions
2. the value of destination is a string
3. the value of sub-instructions is a list of all sub-instructions
4. the value of state-constraints is a JSON. The key is index start from zero and the value is a list of all constraints, each constraint is a tuple
5. the value of decisions is a nested JSON. The first level JSON's key is index start from zero and it’s value is second level JSONS withkeys: landmarks, directions. The value of landmarks is a list of tuples, each tuple contains (landmark, action). The value of directions is alist of direction choice for each sub-instruction.
[FEW-SHOT PROMPT]
An Example:
User: "Walk into the living room and keep walking straight past the living room. Then walk into the entrance under the balcony. Wait inthe entrance to the other room."
You: {{"destination": "entrance to the other room under the balcony", "sub-instructions": ["Walk into the living room", "keep walkingstraight past the living room", "walk into the entrance under the balcony", "wait in the entrance to the other room"], "state-constraints":{{"0": [["location constraint", "living room"]], "1": [["location constraint", "living room"]], "2": [["location constraint", "balcony"],["object constraint", "entrance"]], "3": [["location constraint", "other room"], ["object constraint", "entrance"]]}}, "decisions": {{"0":{{"landmarks": [["living room", "approach"]], "directions": ["forward"]}}, "1": {{"landmarks": [["living room", "move away"]],"directions": ["forward"]}}, "2": {{"landmarks": [["balcony", "approach"], ["entrance", "approach"]], "directions": ["forward"]}}, "3":{{"landmarks": [["other room", "approach"], ["entrance", "approach"]], "directions": ["forward"]}}}}}}
[KEY CONTENT REMINDER]
ATTENTION:
1. constraint type: location constraint is for room type, object constraint is for object type, directions constraint. Don't confuse object constraint with location constraint!
2. landmark choice: approach, move away, approach then move away
3. direction choice: left, right, forward4. The landmark and constraint object should not accurately describe the orientation and order. 
Here are examples about landmark:"second step from the top" -> "step"(neglect order and position relation); "room directly ahead" -> "room"; "right bedroom door" ->"bedroom door"
Instruction : {instruction}
"""
        
        # Prepare the conversation format that the model expects
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        
        # Add visual context from RGB observations
        for i, rgb_img in enumerate(rgb_inputs):  # Use all RGB observations for context
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(rgb_img.astype(np.uint8))
            conversation[0]["content"].append({
                "type": "image", 
                "image": pil_image
            })
        
        # Process the conversation with the model's processor
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        
        # Process images and text together
        image_inputs, video_inputs, video_kwargs = process_vision_info(conversation, return_video_kwargs=True)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generation_config = GenerationConfig(
                max_new_tokens=10240,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            
            generated_ids = self.llm_model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode the response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(response_text)

        # print('YEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEES' * 50)
        
        # Parse JSON response
        try:
            # Extract JSON from response text (in case there's additional text)
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                llm_response = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response text: {response_text}")
            # Fallback to empty constraint structure
            llm_response = {
                "destination": "unknown destination",
                "sub-instructions": [instruction],
                "state-constraints": {"0": []},
                "decisions": {"0": {"landmarks": [], "directions": ["forward"]}}
            }
        
        return llm_response
    
    def _query_llm_for_next_constraint(self, instruction: str, current_rgb: np.ndarray, completed_constraints: List[Dict], current_idx: int) -> Dict:
        """
        Query the LLM with instruction, current observation and RGB history to get the next constraint
        
        Args:
            instruction: Navigation instruction
            current_rgb: Current RGB observation
            completed_constraints: List of already completed constraints
            current_idx: Current constraint index that was just completed
            
        Returns:
            Dictionary containing the next constraint information
        """
        if self.llm_model is None:
            raise ValueError("LLM model not set. Call set_llm_model() first.")
        
        # Prepare input for LLM
        rgb_inputs = self.rgb_history + [current_rgb]
        
        # Create a summary of completed constraints
        completed_summary = ""
        for i, constraint in enumerate(completed_constraints):
            completed_summary += f"Step {i}: Completed constraint - {constraint}\n"
        
        prompt = f"""[TASK DESCRIPTION]
Based on the navigation instruction and visual observations, you need to determine the next navigation constraint for the agent.

[CONTEXT]
You will be given the following information:
- The original high-level instruction for the entire task.
- A summary of constraints that have already been completed.
- The current step number.
- A description of the agent's current visual observations.

[TASK]
Analyze the current visual observations and the original instruction to determine what the agent should do next.

IMPORTANT: 
- If the agent has completed the navigation task and appears to be at or very close to the final destination based on the visual evidence, set "task_completed" to true and provide empty constraints.
- If there are more steps needed, provide the next constraint for the immediate next action.
- Critically consider the visual evidence. Do the current observations match the goal of the current sub-instruction or the final destination?

[OUTPUT DEFINITION]
Provide your answer in JSON format with the following details:
1. use the following keys: task_completed, next_constraint, next_destination, landmarks, directions
2. task_completed: boolean, indicating if the navigation task is fully completed.
3. next_constraint: a list of constraint tuples. Each constraint is a list `[constraint_type, constraint_object]`. This list should be empty if the task is completed.
4. next_destination: a **noun or noun phrase** describing the immediate target landmark or location for the next step (e.g., "kitchen", "stairs", "second door"). It should be "final destination reached" if the task is completed.
5. landmarks: a list of tuples, each tuple contains `[landmark, action]` where action is "approach", "move away", or "approach then move away".
6. directions: a list of direction choices: "left", "right", or "forward".

[CONSTRAINT TYPES]
- location constraint: for room types (e.g., "living room", "kitchen").
- object constraint: for object types (e.g., "table", "door", "stairs").
- direction constraint: for directional movement ("left", "right").

[FEW-SHOT EXAMPLES]

---
**Example 1: Continuing Navigation (Directional Turn)**

[USER]
### Original instruction:
"Go up the stairs, turn left, and wait in front of the second door on your right."

### Completed constraints so far:
Step 1: Approached and went up the stairs.

### Current step:
2

### Visual Observations:
The agent is at the top of the stairs, on a landing. Ahead is a hallway. There are doors visible to the left and right.

[YOU]
{{
  "task_completed": false,
  "next_constraint": [
    ["direction constraint", "left"]
  ],
  "next_destination": "hallway landing",
  "landmarks": [],
  "directions": [
    "left"
  ]
}}
---
**Example 2: Reaching the Final Destination**

[USER]
### Original instruction:
"Go up the stairs, turn left, and wait in front of the second door on your right."

### Completed constraints so far:
Step 1: Approached and went up the stairs.
Step 2: Turned left.
Step 3: Moved forward down the hall past the first door.

### Current step:
4

### Visual Observations:
The agent is in a hallway, positioned directly in front of a wooden door. Based on the previous steps, this is the second door on the right. The agent is very close to it.

[YOU]
{{
  "task_completed": true,
  "next_constraint": [],
  "next_destination": "final destination reached",
  "landmarks": [],
  "directions": []
}}
---
**Example 3: Navigating with Objects**

[USER]
### Original instruction:
"Go through the bathroom into the kitchenette, go past the counter, out the open door, head down the hallway and stop right past the open book."

### Completed constraints so far:
Step 1: Went through the bathroom.
Step 2: Entered the kitchenette.

### Current step:
3

### Visual Observations:
The agent is now inside a kitchenette area. A counter is visible directly ahead.

[YOU]
{{
  "task_completed": false,
  "next_constraint": [
    ["object constraint", "counter"]
  ],
  "next_destination": "counter",
  "landmarks": [
    ["counter", "move away"]
  ],
  "directions": [
    "forward"
  ]
}}
---

[CONTEXT]
Original instruction: {instruction}

Completed constraints so far:
{completed_summary}

Current step: {current_idx + 1}

[TASK]
Analyze the current visual observations and the original instruction to determine what the agent should do next.

[OUTPUT DEFINITION]
Provide your answer in JSON format."""
        
        # Prepare the conversation format that the model expects
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        
        # Add visual context from RGB observations
        for i, rgb_img in enumerate(rgb_inputs):  # Use all RGB observations for context
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(rgb_img.astype(np.uint8))
            conversation[0]["content"].append({
                "type": "image", 
                "image": pil_image
            })
        
        # Process the conversation with the model's processor
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        
        # Process images and text together
        image_inputs, video_inputs, video_kwargs = process_vision_info(conversation, return_video_kwargs=True)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generation_config = GenerationConfig(
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            
            generated_ids = self.llm_model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode the response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print("LLM Response for Next Constraint:")
        print(response_text)

        # Parse JSON response
        try:
            # Extract JSON from response text (in case there's additional text)
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                llm_response = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing LLM response for next constraint: {e}")
            print(f"Response text: {response_text}")
            # Fallback to empty constraint structure
            llm_response = {
                "task_completed": False,
                "next_constraint": [],
                "next_destination": "unknown destination",
                "landmarks": [],
                "directions": ["forward"]
            }

        return llm_response
    
    def _save_rgb_to_file(self, rgb: np.ndarray, step: int):
        """Save RGB observation to file if enabled"""
        if self.save_rgb_history and self.episode_rgb_dir is not None:
            rgb_path = os.path.join(self.episode_rgb_dir, f"step_{step:04d}.png")
            Image.fromarray(rgb).save(rgb_path)
    
    def _process_llm_reply(self, obs: Observations, online_mode: bool = True):
        def _get_first_destination(sub_constraints: dict, llm_destination: str) -> str:
            for constraints in sub_constraints.values():
                for constraint in constraints:
                    if constraint[0] != "direction constraint":
                        return constraint[1]
            else:
                return llm_destination
        
        if online_mode and self.llm_model is not None:
            self.instruction = obs['instruction']['text']
            current_rgb = obs['rgb']
            
            # Save current RGB to history
            self.rgb_history.append(current_rgb)
            
            if self.use_dynamic_constraints:
                # In dynamic mode, only get the first constraint
                self.llm_reply = self._query_llm_for_next_constraint(
                    self.instruction, current_rgb, self.completed_constraints, 0
                )
                # Convert the dynamic response to standard format
                self._convert_dynamic_to_standard_format()
            else:
                # Original mode: get all constraints at once
                self.llm_reply = self._query_llm_for_constraints(self.instruction, current_rgb)
        else:
            # Offline mode: use pre-computed llm_reply from observation
            self.llm_reply = obs['llm_reply']
            self.instruction = obs['instruction']['text']
            
            # Still save RGB for potential debugging/analysis
            if 'rgb' in obs:
                self.rgb_history.append(obs['rgb'])
        
        if not self.use_dynamic_constraints:
            # Original logic for static mode
            self.sub_instructions = self.llm_reply['sub-instructions']
            self.sub_constraints = self.llm_reply['state-constraints']
            self.decisions = self.llm_reply['decisions']
            self.destination = _get_first_destination(self.sub_constraints, self.llm_reply['destination'])
            print("!!!!!!!!!!!!!!! first destination: ", self.destination)
            # self.destination = self.sub_instructions[0]
            self.last_destination = self.destination
            first_landmarks = self.decisions['0']['landmarks']
            self.destination_class = [item[0] for item in first_landmarks]
            self.classes = self._process_classes(self.base_classes, self.destination_class)
            self.constraints_check = [False] * len(self.sub_constraints)
        else:
            # Dynamic mode: handle current constraint only
            self._setup_current_constraint()
    
    def _convert_dynamic_to_standard_format(self):
        """Convert dynamic LLM response to standard format for compatibility"""
        # Create a simplified structure for the current constraint
        self.current_dynamic_constraint = {
            "constraint": self.llm_reply.get('next_constraint', []),
            "destination": self.llm_reply.get('next_destination', 'unknown'),
            "landmarks": self.llm_reply.get('landmarks', []),
            "directions": self.llm_reply.get('directions', ['forward'])
        }
    
    def _setup_current_constraint(self):
        """Setup current constraint for dynamic mode"""
        if hasattr(self, 'current_dynamic_constraint'):
            self.destination = self.current_dynamic_constraint['destination']
            self.last_destination = self.destination
            self.destination_class = [item[0] for item in self.current_dynamic_constraint['landmarks']]
            self.classes = self._process_classes(self.base_classes, self.destination_class)
            
            # Setup single constraint structure for compatibility
            self.sub_constraints = {"0": self.current_dynamic_constraint['constraint']}
            self.constraints_check = [False]
            self.current_constraint_step = 0
            
            print(f"!!!!!!!!!!!!!!! Dynamic constraint {len(self.completed_constraints)}: {self.destination}")
    
    def _query_next_dynamic_constraint(self, current_rgb: np.ndarray):
        """Query LLM for the next constraint in dynamic mode"""
        if self.llm_model is None:
            return False
            
        self.rgb_history.append(current_rgb)
        
        next_constraint_info = self._query_llm_for_next_constraint(
            self.instruction, current_rgb, self.completed_constraints, len(self.completed_constraints)
        )
        
        # Check if we have a valid next constraint
        if next_constraint_info.get('task_completed', False):
            print("LLM indicates task is completed!")
            return False
        elif (next_constraint_info.get('next_constraint') and 
            len(next_constraint_info['next_constraint']) > 0):
            
            if hasattr(self, 'current_dynamic_constraint'):
                self.completed_constraints.append(self.current_dynamic_constraint.copy())
            
            self.current_dynamic_constraint = {
                "constraint": next_constraint_info['next_constraint'],
                "destination": next_constraint_info['next_destination'],
                "landmarks": next_constraint_info['landmarks'],
                "directions": next_constraint_info['directions']
            }
            
            self._setup_current_constraint()
            return True
        else:
            print("No more constraints available from LLM")
            return False
    
    
    def _process_one_step_floor(self, one_step_full_map: np.ndarray, kernel_size: int=3) -> np.ndarray:
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        one_step_full_map = remove_small_objects(one_step_full_map.astype(bool), min_size=64)
        
        obstacles = one_step_full_map[0, ...].astype(bool)
        explored_area = one_step_full_map[1, ...].astype(bool)
        objects = np.sum(one_step_full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)
        navigable = np.logical_or.reduce(one_step_full_map[map_channels:, ...][navigable_index])
        navigable = np.logical_and(navigable, np.logical_not(objects))
        
        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=disk(kernel_size))
        
        return floor
        
    def _process_map(self, step: int, full_map: np.ndarray, kernel_size: int=3) -> tuple:
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        full_map = remove_small_objects(full_map.astype(bool), min_size=64)
        
        obstacles = full_map[0, ...].astype(bool)
        explored_area = full_map[1, ...].astype(bool)
        objects = np.sum(full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)
        
        selem = disk(kernel_size)
        obstacles_closed = binary_closing(obstacles, footprint=selem)
        objects_closed = binary_closing(objects, footprint=selem)
        navigable = np.logical_or.reduce(full_map[map_channels:, ...][navigable_index])
        navigable = np.logical_and(navigable, np.logical_not(objects))
        navigable_closed = binary_closing(navigable, footprint=selem)
        
        untraversible = np.logical_or(objects_closed, obstacles_closed)
        untraversible[navigable_closed == 1] = 0
        untraversible = remove_small_objects(untraversible, min_size=64)
        untraversible = binary_closing(untraversible, footprint=disk(3))
        traversible = np.logical_not(untraversible)

        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=selem)
        traversible = np.logical_or(floor, traversible)
        
        explored_area = binary_closing(explored_area, footprint=selem)
        contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = np.zeros(full_map.shape[-2:], dtype=np.uint8)
        image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)
        frontiers = np.logical_and(floor, image)
        frontiers = remove_small_objects(frontiers.astype(bool), min_size=64)

        return traversible, floor, frontiers.astype(np.uint8)
    
    def restore_from_trajectory_point(self, traj_point):
        self.envs.reset()
        print([traj_point["pose"][:3], traj_point["pose"][3]])
        self.envs.call(['set_agent_state'], [traj_point["pose"][:3], traj_point["pose"][3]])
        self.policy.load_state_dict(traj_point["policy_state"])
        self.mapping_module.load_state_dict(traj_point["mapping_state"])
        self.value_map_module.load_state_dict(traj_point["value_map_state"])
        self.history_module.load_state_dict(traj_point["history_state"])
        self.rgb_history = copy.deepcopy(traj_point["llm_state"]["rgb_history"])
        self.completed_constraints = copy.deepcopy(traj_point["llm_state"]["completed_constraints"])
        if traj_point["llm_state"]["current_dynamic_constraint"] is not None:
            self.current_dynamic_constraint = copy.deepcopy(traj_point["llm_state"]["current_dynamic_constraint"])
        self.instruction = traj_point["instruction"]

    def _maps_initialization(self, online_llm: bool = True):
        obs = self.envs.reset() #type(obs): list
        self._process_llm_reply(obs[0], online_mode=online_llm)
        self.current_episode_id = self.envs.current_episodes()[0].episode_id
        print("current episode id: ", self.current_episode_id)
        
        if self.save_rgb_history:
            self.episode_rgb_dir = os.path.join(
                self.config.RESULTS_DIR, 
                f"rgb_history_ep_{self.current_episode_id}"
            )
            os.makedirs(self.episode_rgb_dir, exist_ok=True)
        
        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        batch_obs = self._batch_obs(obs)
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
        self.mapping_module(batch_obs, poses)
        full_map, full_pose, _ = self.mapping_module.update_map(0, self.detected_classes, self.current_episode_id)
        self.mapping_module.one_step_full_map.fill_(0.)
        self.mapping_module.one_step_local_map.fill_(0.)
    
    def _look_around(self):
        print("\n========== LOOK AROUND ==========\n")
        full_pose, obs, dones, infos = None, None, None, None
        for step in range(0, 12):
            actions = []
            for _ in range(self.config.NUM_ENVIRONMENTS):
                actions.append({"action": HabitatSimActions.TURN_LEFT})
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            if dones[0]:
                return full_pose, obs, dones, infos
            batch_obs = self._batch_obs(obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses)
            full_map, full_pose, one_step_full_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            self.traversible, self.floor, self.frontiers = self._process_map(step, full_map[0])
            self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])
                        
            blip_value = self.value_map_module.get_blip_value(Image.fromarray(obs[0]['rgb']), self.destination)
            blip_value = blip_value.detach().cpu().numpy()
            value_map = self.value_map_module(step, full_map[0], self.floor, self.one_step_floor, 
                                              self.collision_map, blip_value, full_pose[0], 
                                  self.detected_classes, self.current_episode_id)
        self._action = self.policy(self.value_map_module.value_map[1], self.collision_map,
                                    full_map[0], self.floor, self.traversible, 
                                    full_pose[0], self.frontiers, self.detected_classes,
                                    self.destination_class, self.classes, False, one_step_full_map[0], 
                                    self.current_detections, self.current_episode_id, False, step)
        
        return full_pose, obs, dones, infos
    
    
    def reset(self) -> None:
        self.classes = []
        self.current_detections = None
        self.detected_classes = OrderedSet()
        self.floor = np.zeros(self.map_shape)
        self.one_step_floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversible = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)
        
        # Reset LLM-related attributes
        self.rgb_history = []
        self.trajectory_rewards = []
        self.dist_reward = []
        
        # Reset dynamic constraints attributes
        self.completed_constraints = []
        self.current_constraint_step = 0
        if hasattr(self, 'current_dynamic_constraint'):
            delattr(self, 'current_dynamic_constraint')
        
        self.policy.reset()
        self.mapping_module.reset()
        self.value_map_module.reset()
        self.history_module.reset()
    
    def rollout(self, online_llm: bool = True):
        """
        Execute a whole episode which consists of a sequence of sub-steps
        
        Args:
            online_llm: Whether to use online LLM queries or offline llm_reply
        """
        self._maps_initialization(online_llm)
        full_pose, obs, dones, infos = self._look_around()
        print("\n ========== START TO NAVIGATE ==========\n")
        
        trajectory_points = []
        direction_points = []
        constraint_steps = 0
        collided = 0
        empty_value_map = 0
        direction_map_exist = False
        replan = False
        start_to_wait = False
        search_destination = False
        last_action, current_action = None, None
        last_pose, start_check_pose = None, None
        current_pose = full_pose[0]
        self._action2 = None
        
        # Initialize based on mode
        if self.use_dynamic_constraints:
            current_idx = 0
            current_constraint = self.sub_constraints.get("0", [])
            all_constraint_types = [item[0] for item in current_constraint]
            if hasattr(self, 'current_dynamic_constraint'):
                landmarks = self.current_dynamic_constraint['landmarks']
                self.destination_class = [item[0] for item in landmarks]
                self.classes = self._process_classes(self.base_classes, self.destination_class)
        else:
            current_idx = self.constraints_check.index(False)
            landmarks = self.decisions[str(current_idx)]['landmarks']
            self.destination_class = [item[0] for item in landmarks]
            self.classes = self._process_classes(self.base_classes, self.destination_class)
            current_constraint = self.sub_constraints[str(current_idx)]
            all_constraint_types = [item[0] for item in current_constraint]
        
        no_constraint_steps = 0  # Track steps without active constraints
        
        last_distance = infos[0]['position']['distance'][-1]

        for step in range(12, self.max_step):
            print(f"\nepisode:{self.current_episode_id}, step:{step}")
            print(f"instr: {self.instruction}")
            
            if self.use_dynamic_constraints:
                if hasattr(self, 'current_dynamic_constraint'):
                    print(f"dynamic_constraint_{len(self.completed_constraints)}: {self.current_dynamic_constraint['destination']}")
                else:
                    print("dynamic_constraint: No current constraint")
            else:
                print(f"sub_instr_{current_idx}: {self.sub_instructions[current_idx]}")
            
            constraint_steps += 1
            
            
            # Save RGB observation if enabled
            self._save_rgb_to_file(obs[0]['rgb'], step)
            
            position = full_pose[0][:2] * 100 / self.resolution
            heading = full_pose[0][-1]
            print("full pose: ", full_pose[0])
            y, x = min(int(position[0]), self.map_shape[0] - 1), min(int(position[1]), self.map_shape[1] - 1)
            self.visited[x, y] = 1
            trajectory_points.append((y, x))
            direction_points.append(np.array([x, y]))
            if len(trajectory_points) > 2:
                del trajectory_points[0]
            if len(direction_points) > 5:
                del direction_points[0]
            
            history_map = self.history_module(trajectory_points, step, self.current_episode_id)

            if "direction constraint" in all_constraint_types and start_check_pose is None:
                start_check_pose = full_pose[0]
                
            constraint_check_condition = (
                sum(self.constraints_check) < len(self.sub_instructions)
            )

            save_history = False            
            
            if constraint_check_condition:
                if (len(current_constraint) > 0 
                    and current_constraint[0][0] == "direction constraint" 
                    and not direction_map_exist):
                    direction = current_constraint[0][1]
                    if len(direction_points) < 5:
                        current_position = direction_points[-1]
                        last_five_position = direction_points[-1]
                    else:
                        current_position = direction_points[-1]
                        last_five_position = direction_points[0]
                    direction_map = self.direction_module(current_position, last_five_position, heading,
                                                          direction, step, self.current_episode_id)
                    direction_map_exist = True
                else:
                    direction_map = np.ones(self.map_shape)
                
                check = self.constraints_monitor(current_constraint, obs[0], 
                                                self.current_detections, self.classes, 
                                                current_pose, start_check_pose)
                print(current_constraint, check)
                if (len(current_constraint) > 0 
                    and current_constraint[0][0] == "direction constraint" 
                    and check[0] == True):
                    direction_map = np.ones(self.map_shape)
                
                if len(check) == 0:
                    print("empty constraint")
                elif sum(check) < len(check):
                    current_constraint = [current_constraint[i] 
                                          for i in range(len(current_constraint)) 
                                          if not check[i]]
                    all_constraint_types = [item[0] for item in current_constraint]
                    
                # Calculate reward when constraint changes or completes
                if (sum(check) == len(check) or constraint_steps >= self.max_constraint_steps):
                    if not start_to_wait:
                        start_to_wait = True
                        if self.use_dynamic_constraints:
                            self.constraints_check[0] = True
                        else:
                            self.constraints_check[current_idx] = True
                        
                if start_to_wait and (constraint_steps >= self.min_constraint_steps):
                    if False in self.constraints_check:
                        current_idx = self.constraints_check.index(False)
                        print(f"sub_instr_{current_idx}: {self.sub_instructions[current_idx]}")
                        landmarks = self.decisions[str(current_idx)]['landmarks']
                        if len(landmarks) > 0:
                            self.destination_class = [item[0] for item in landmarks]
                            self.classes = self._process_classes(self.base_classes, self.destination_class)
                        last_constraint = current_constraint  # Save for reward calculation
                        current_constraint = self.sub_constraints[str(current_idx)]
                        all_constraint_types = [item[0] for item in current_constraint]
                        current_pose, start_check_pose = None, None
                    else:
                        current_constraint, all_constraint_types = [], []
                        print("all constraints are done")
                    constraint_steps = 0
                    start_to_wait = False

            if save_history:
                self.trajectory_history.append({
                    "step": step,
                    "pose": full_pose[0].copy(),
                    "rgb": obs[0]['rgb'].copy(),
                    "obs": copy.deepcopy(obs[0]),
                    "infos": copy.deepcopy(infos[0]),
                    "instruction": self.instruction,
                    "policy_state": copy.deepcopy(self.policy.state_dict()),
                    "mapping_state": copy.deepcopy(self.mapping_module.state_dict()),
                    "value_map_state": copy.deepcopy(self.value_map_module.state_dict()),
                    "history_state": copy.deepcopy(self.history_module.state_dict()),
                    "llm_state": {
                        "completed_constraints": copy.deepcopy(self.completed_constraints),
                        "current_dynamic_constraint": copy.deepcopy(getattr(self, 'current_dynamic_constraint', None)),
                    },
                })


            print("current constraint: ", current_constraint)
            print("constraint_steps: ", constraint_steps)
            
            # Track steps without constraints in dynamic mode
            if self.use_dynamic_constraints and len(current_constraint) == 0:
                no_constraint_steps += 1

                if no_constraint_steps > 10:  # Arbitrary threshold
                    actions = []
                    for _ in range(self.config.NUM_ENVIRONMENTS):
                        actions.append({"action": HabitatSimActions.STOP})
                    outputs = self.envs.step(actions)
                    self._calculate_metric(infos)
                    break
            else:
                no_constraint_steps = 0
            
                
            if self.use_dynamic_constraints:
                if hasattr(self, 'current_dynamic_constraint') and len(current_constraint) > 0:
                    if current_constraint[0][0] != "direction constraint":
                        self.destination = self.current_dynamic_constraint['destination']
                elif len(current_constraint) == 0:
                    # Use final destination from original LLM reply if available, or default to exploration
                    if hasattr(self, 'llm_reply') and 'destination' in self.llm_reply:
                        self.destination = self.llm_reply['destination']
                    else:
                        self.destination = "explore to find goal"
                    print(f"No current constraints, destination set to: {self.destination}")
            else:
                if len(current_constraint) > 0 and current_constraint[0][0] != "direction constraint":
                    new_destination = current_constraint[0][1]
                    if current_idx >= len(self.sub_instructions) - 1:
                        self.destination = self.llm_reply['destination']
                    else:
                        self.destination = new_destination
                if len(current_constraint) == 0 and current_idx >=len(self.sub_constraints) - 1:
                    self.destination = self.llm_reply['destination']
                
            if self.destination != self.last_destination:
                self.value_map_module.value_map[...] *= 0.5
                self.last_destination = self.destination
                
            if np.sum(self.value_map_module.value_map[1].astype(bool)) <= 24**2:
                empty_value_map += 1
                constraint_steps = 0
            else:
                empty_value_map = 0 
            if empty_value_map >= 5:
                full_pose, obs, dones, infos = self._look_around()
                if dones[0]:
                    self._calculate_metric(infos)
                    break
                empty_value_map = 0
                constraint_steps = 0
            
            actions = []
            for _ in range(self.config.NUM_ENVIRONMENTS):
                if self.keyboard_control:
                    self._action2 =self._use_keyboard_control() 
                    actions.append(self._action2)
                else:
                    actions.append(self._action)
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            
            current_dist = infos[0]['position']['distance'][-1]
            print(f"current distance: {current_dist}, last distance: {last_distance}")
            if dones[0]:
                self._calculate_metric(infos)
                break
            self.dist_reward.append(-(current_dist - last_distance))
            last_distance = current_dist
            batch_obs = self._batch_obs(obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses)
            full_map, full_pose, one_step_full_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            
            self.traversible, self.floor, self.frontiers = self._process_map(step, full_map[0])
            self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])
            
            last_pose = current_pose
            current_pose = full_pose[0]
            if last_pose is not None and current_pose is not None:
                displacement = calculate_displacement(last_pose, current_pose, self.resolution)
                if displacement < 0.2 * 100 / self.resolution:
                    collided += 1
                else:
                    collided = 0
                    replan = False
                if collided >= 30:
                    replan = True
                    print(f"{self.current_episode_id}: {collided}\n")
                    fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR, 
                                        f"r{self.local_rank}_w{self.world_size}_collision_stuck.txt")
                    with open(fname, "a") as f:
                        f.writelines(f"id: {str(self.current_episode_id)}; step: {str(step)}; collided: {str(collided)}\n")
                
            last_action = current_action
            current_action = self._action
            if last_pose is not None and current_action["action"] == 1:
                collision_map = collision_check_fmm(last_pose, current_pose, self.resolution, 
                                                self.mapping_module.map_shape)
                self.collision_map = np.logical_or(self.collision_map, collision_map)
            
            blip_value = self.value_map_module.get_blip_value(Image.fromarray(obs[0]['rgb']), self.destination)
            blip_value = blip_value.detach().cpu().numpy()
            value_map = self.value_map_module(step, full_map[0], self.floor, self.one_step_floor, self.collision_map, 
                                  blip_value, full_pose[0], self.detected_classes, self.current_episode_id)
            self._action = self.policy(self.value_map_module.value_map[1] * history_map, self.collision_map,
                                    full_map[0], self.floor, self.traversible, 
                                    full_pose[0], self.frontiers, self.detected_classes,
                                    self.destination_class, self.classes, search_destination, 
                                    one_step_full_map[0], self.current_detections, 
                                    self.current_episode_id, replan, step)
        
        with open(f"trajectory_ep_{self.current_episode_id}.pkl", "wb") as f:
            pickle.dump(self.trajectory_history, f)
        return self._calculate_final_trajectory_reward()
    
    def set_llm_model(self, processor, llm_model, requires_grad = False):
        """Set the LLM model for online constraint generation"""
        self.llm_model = llm_model
        self.processor = processor
        self.llm_model.requires_grad_(requires_grad)
    
    def set_save_rgb_history(self, save_rgb: bool):
        """Enable/disable saving RGB history to files"""
        self.save_rgb_history = save_rgb
    
    def set_dynamic_constraints_mode(self, enabled: bool):
        """Enable/disable dynamic constraints mode"""
        self.use_dynamic_constraints = enabled
        print(f"Dynamic constraints mode: {'enabled' if enabled else 'disabled'}")
    
    def get_trajectory_reward_dynamic(self, processor, llm_model, save_rgb_history: bool = False) -> float:
        """
        Public interface: Get trajectory reward using dynamic constraint mode
        
        Args:
            processor: The processor for the LLM model
            llm_model: The LLM model to use for dynamic constraint generation
            save_rgb_history: Whether to save RGB observations to files
            
        Returns:
            Total trajectory reward (sum of all constraint rewards)
        """
        # Set up for this evaluation
        self.set_llm_model(processor, llm_model)
        self.set_save_rgb_history(save_rgb_history)
        self.set_dynamic_constraints_mode(True)
        
        self.reset()
                
        total_reward = self.rollout(online_llm=True)
        
        return total_reward
    
    def get_rgb_history(self) -> List[np.ndarray]:
        """Get the RGB observation history"""
        return self.rgb_history.copy()
    
    def clear_history(self):
        """Clear RGB and reward histories"""
        self.rgb_history = []
        self.dist_reward = []
    
    def eval(self, online_llm: bool = False, dynamic_mode: bool = False):
        """
        Evaluate episodes
        
        Args:
            online_llm: Whether to use online LLM queries instead of pre-computed llm_reply
            dynamic_mode: Whether to use dynamic constraint generation mode
        """
        if dynamic_mode:
            self.set_dynamic_constraints_mode(True)
            print("Using dynamic constraint generation mode")
        
        self._set_eval_config()
        self._init_envs()
        self._collect_val_traj()
        self._initialize_policy()
        
        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
            
        self.state_eps = {}
        t1 = time.time()
        for i in tqdm(range(30)):
            trajectory_reward = self.rollout(online_llm=online_llm)
            if online_llm:
                print(f"Episode {i} trajectory reward: {trajectory_reward}")
            self.reset()
        
        print('Success rate:', self.success / self.tot)

        self.envs.close()
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR, 
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)
        t2 = time.time()
        logger.info(f"time: {t2 - t1}s")
        print("test time: ", t2 - t1)