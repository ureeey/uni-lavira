import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from typing import List
import supervision as sv
from habitat import Config
from collections.abc import Sequence
from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.constant import direction_mapping
from transformers import AutoProcessor, Blip2ForConditionalGeneration, Blip2ForImageTextRetrieval

class ConstraintsMonitor(nn.Module):
    def __init__(self, config: Config, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.resolution = config.MAP.MAP_RESOLUTION
        self.turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE
        self.device = device
        self._load_from_disk()
        
    def _create_model(self):
        """ We change this method for compatibility with the new model loading method.
        """
        assert False, "This method is deprecated. Use _load_from_disk instead."

    def _load_from_disk(self):
        # self.model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g").to(self.device)
        self.judge_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16
        ).to('cuda')

        # print('DEVICES' * 100, self.judge_model.device)

        self.processor = AutoProcessor.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
        )
    def location_constraint(self, obs: np.ndarray, scene: str):
        image = Image.fromarray(obs['rgb'].astype(np.uint8))
        question = f"Question: Are you in the {scene}? Answer:"

        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(self.judge_model.device, torch.float16)

        with torch.no_grad():
            generated_ids = self.judge_model.generate(**inputs, max_new_tokens=50)
            answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower().strip()

        print(answer)

        if "yes" in answer:
            return True
        else:
            return False
    
    def object_constraint(self, current_detection: sv.Detections, object: str, classes: List):
        """ 
        use grounded-sam's detections to check object
        """
        class_ids = current_detection.class_id
        class_names = [classes[i] for i in class_ids]
        if object in class_names:
            return True
        else:
            return False
    
    def direction_constraint(self, current_pose: Sequence, last_pose: Sequence, object):
        """ 
        check by geometric relation
        """
        heading = -1 * last_pose[-1]
        current_position, _ = get_agent_position(current_pose, self.resolution)
        last_position, _ = get_agent_position(last_pose, self.resolution)
        position_vector = current_position - last_position
        displacement = np.linalg.norm(position_vector)
        heading_vector = angle_to_vector(heading)
        rotation_matrix = np.array([[0, -1], 
                                [1, 0]])
        heading_vector = np.dot(rotation_matrix, heading_vector)
        if np.array_equal(position_vector, np.array([0., 0.])):
            return False
        degrees, direction = angle_and_direction(heading_vector, position_vector, self.turn_angle + 1)
        if degrees >= 120:
            movement = "backward"
        elif degrees == 0 or degrees == 180 or direction == 1:
            movement = "forward"
        else:
            if direction == 2:
                movement = "left"
            elif direction == 3:
                movement = "right"
        object_direction = direction_mapping.get(object, "ambiguous direction")
        if object_direction == "ambiguous direction":
            print("!Won't check ambiguous direction!")
            return True
        elif movement == object_direction and displacement >= 0.5 * 100 / self.resolution:
            return True
        else:
            return False
    
    def forward(self, 
                constraints: List, obs: np.ndarray, 
                detection: sv.Detections, classes: List,
                current_pose: Sequence, last_pose: Sequence):
        res  = []
        for item in constraints:
            constraint_type, constraint_object = item[:2]
            constraint_type = constraint_type.lower().strip()
            if constraint_type == "location constraint":
                check = self.location_constraint(obs, constraint_object)
                res.append(check)
            elif constraint_type == "object constraint":
                check = self.object_constraint(detection, constraint_object, classes)
                res.append(check)
            elif constraint_type == "direction constraint":
                check = self.direction_constraint(current_pose, last_pose, constraint_object)
                res.append(check)
        
        return res