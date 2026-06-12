import os
import pdb
import queue
import copy
import gzip
import json
import time
import cv2
import re
import tempfile
import textwrap
import warnings
import traceback
import base64
import io
import requests
from openai import OpenAI

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from fastdtw import fastdtw
from typing import List, Any, Dict
from collections import defaultdict
from skimage.morphology import binary_closing, remove_small_objects, disk
from scipy.spatial.transform import Rotation

import torch
from torch import Tensor
from torchvision import transforms

from vlnce_baselines import config

try:
    import supervision as sv
except ImportError:
    logger.info("Warning: supervision not available, some features may not work")

from habitat import logger
from habitat_extensions.measures import NDTW
from habitat.core.simulator import Observations
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.environments import get_env_class
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry

# Import Habitat visualization utilities
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.maps import draw_agent

from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.map.history_map import HistoryMap
from vlnce_baselines.map.direction_map import DirectionMap
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.Policy import FusionMapPolicy
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import gather_list_and_concat, get_device
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.utils.constant import base_classes, map_channels, direction_mapping

from pyinstrument import Profiler
import warnings
warnings.filterwarnings('ignore')

# Keep these imports for compatibility but they won't be used with API calls
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info

ACTION_TO_TEXT = ['stop','forward', 'turn left', 'turn right']
ACTION_TO_DISPLAY = {
    0: 'STOP',
    1: 'FORWARD', 
    2: 'TURN LEFT',
    3: 'TURN RIGHT',
    'stop': 'STOP',
    'forward': 'FORWARD', 
    'turn left': 'TURN LEFT',
    'turn right': 'TURN RIGHT',
}

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
    
    def image_to_base64(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    
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
            
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            
            
            self.stats[stats_key]['calls'] += 1
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"API Call usage - {response.usage}")
                self.stats[stats_key]['input_tokens'] += response.usage.prompt_tokens or 0
                self.stats[stats_key]['output_tokens'] += response.usage.completion_tokens or 0
                self.stats[stats_key]['total_tokens'] += response.usage.total_tokens or 0
                
                logger.info(f"{stats_key.upper()} model usage - Input: {response.usage.prompt_tokens}, "
                           f"Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
            
            print(f'Generating uses {time.time()-t} seconds.')
            import sys
            sys.stdout.flush()
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"API Call error with {'secondary' if use_secondary else 'primary'} model ({model_name}): {e}")
            logger.info('Forcing retry..')
            import time
            time.sleep(30)
            return self.generate(messages, images, max_new_tokens, temperature, use_secondary, **kwargs)
            # return "Error: Failed to get response from API"
    
    def generate_with_primary(self, messages, images=None, max_new_tokens=1024, temperature=0.7, **kwargs):
        return self.generate(messages, images, max_new_tokens, temperature, use_secondary=False, **kwargs)
    
    def generate_with_secondary(self, messages, images=None, max_new_tokens=1024, temperature=0.7, **kwargs):
        if not self.secondary_client:
            logger.info("Warning: Secondary model not configured, falling back to primary model")
            return self.generate_with_primary(messages, images, max_new_tokens, temperature, **kwargs)
        return self.generate(messages, images, max_new_tokens, temperature, use_secondary=True, **kwargs)
    
    def get_model_info(self):
        info = {
            "primary_model": self.model_name,
            "secondary_model": self.secondary_model_name if self.secondary_client else None,
            "has_secondary": self.secondary_client is not None
        }
        return info
    
    def get_usage_stats(self):
        return self.stats.copy()
    
    def print_usage_stats(self):
        total_calls = self.stats['primary']['calls'] + self.stats['secondary']['calls']
        total_tokens = self.stats['primary']['total_tokens'] + self.stats['secondary']['total_tokens']
        
        logger.info("=== MODEL USAGE STATISTICS ===")
        logger.info(f"Primary Model ({self.model_name}):")
        logger.info(f"  - Calls: {self.stats['primary']['calls']}")
        logger.info(f"  - Input tokens: {self.stats['primary']['input_tokens']:,}")
        logger.info(f"  - Output tokens: {self.stats['primary']['output_tokens']:,}")
        logger.info(f"  - Total tokens: {self.stats['primary']['total_tokens']:,}")
        
        if self.secondary_client:
            logger.info(f"Secondary Model ({self.secondary_model_name}):")
            logger.info(f"  - Calls: {self.stats['secondary']['calls']}")
            logger.info(f"  - Input tokens: {self.stats['secondary']['input_tokens']:,}")
            logger.info(f"  - Output tokens: {self.stats['secondary']['output_tokens']:,}")
            logger.info(f"  - Total tokens: {self.stats['secondary']['total_tokens']:,}")
        
        logger.info(f"TOTAL:")
        logger.info(f"  - Total calls: {total_calls}")
        logger.info(f"  - Total tokens: {total_tokens:,}")
        logger.info("===============================")
        
        return {
            'primary': self.stats['primary'].copy(),
            'secondary': self.stats['secondary'].copy(),
            'total_calls': total_calls,
            'total_tokens': total_tokens
        }
    
    def reset_stats(self):
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
    
    def eval(self):
        pass

from vlnce_baselines.utils.depth_utils import get_point_cloud_from_z_t
from vlnce_baselines.utils.rotation_utils import get_r_matrix

@baseline_registry.register_trainer(name="ZS-Evaluator-mp")
class ZeroShotVlnEvaluatorMP(BaseTrainer):
    def __init__(self, config, r2r, segment_module=None, mapping_module=None) -> None:
        super().__init__()
        self.backtrack_steps = 0  
        self.r2r = r2r
        self.device = get_device(config.TORCH_GPU_ID)
        if torch.cuda.is_available() and self.device.type == "cuda":
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
        
        # Detection confidence threshold
        self.confidence_threshold = getattr(config, 'DETECTION_CONFIDENCE_THRESHOLD', 0.0)

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
        
        self.model.eval()
        # Keep processor for potential image preprocessing (though not needed for API calls)
        # self.processor = AutoProcessor.from_pretrained(model_path)
        
        self.rgb_history = []
        self.current_action_description = None

        # RGB saving configuration
        self.save_rgb = False
        self.rgb_save_dir = getattr(config, 'RGB_SAVE_DIR', './saved_rgb_images')
        self.all_rgb_data = []
        
        # Create RGB save directory
        if self.save_rgb:
            os.makedirs(self.rgb_save_dir, exist_ok=True)

        self.rgb_buffer = []
        
        # Target tracking for navigation
        self.visited_targets = []  # List of targets the agent has identified/visited
        self.current_step = 0      # Track current step for navigation decisions
        
        # Distance thresholds for target management (in map units)
        self.target_reached_threshold = getattr(config, 'TARGET_REACHED_THRESHOLD', 15.0)
        self.target_stop_threshold = getattr(config, 'TARGET_STOP_THRESHOLD', 3.0)
        self.target_close_threshold = getattr(config, 'TARGET_CLOSE_THRESHOLD', 8.0)

    def _set_eval_config(self) -> None:
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
        # logger.info("start to initialize environments")
        
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
        )
        logger.info(f"local rank: {self.local_rank}, num of episodes: {self.envs.number_of_episodes}")
        self.detected_classes = OrderedSet()
        # logger.info("initializing environments finished!")
    
    def _collect_val_traj(self) -> None:
        if not self.r2r:
            role = self.config.TASK_CONFIG.DATASET.ROLES
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        if self.r2r:
            with gzip.open(self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split)) as f:
                gt_data = json.load(f)
        else:
            with gzip.open(self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split, role=role[0])) as f:
                gt_data = json.load(f)

        self.gt_data = gt_data
    
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
        metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
        metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
        metric['sdtw'] = metric['ndtw'] * metric['success']
        self.state_eps[ep_id] = metric
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR, 
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)
        logger.info(str(self.state_eps[ep_id]))
    
    def _initialize_policy(self) -> None:
        # logger.info("start to initialize policy")
        self.segment_module = GroundedSAM(self.config, self.device)
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        
        self.history_module = HistoryMap(self.config, self.mapping_module.map_shape)
        self.direction_module = DirectionMap(self.config, self.mapping_module.map_shape)
        self.policy = FusionMapPolicy(self.config, self.mapping_module.map_shape[0])
        self.policy.reset()
    
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
        masks, labels, annotated_images, self.current_detections =\
            self.segment_module.segment(rgb, classes=self.classes)
        self.mapping_module.rgb_vis = annotated_images
        assert len(masks) == len(labels), f"The number of masks not equal to the number of labels!"
        # logger.info("current step detected classes (before filtering): ", labels)
        
        # Filter out detections with confidence < threshold
        filtered_masks = []
        filtered_labels = []
        filtered_indices = []  # Keep track of which detections we keep
        
        for i, label in enumerate(labels):
            # Extract confidence score from label (format: "object_name confidence")
            parts = label.split()
            if len(parts) >= 2:
                try:
                    confidence = float(parts[-1])
                    if confidence >= self.confidence_threshold:
                        filtered_masks.append(masks[i])
                        filtered_labels.append(label)
                        filtered_indices.append(i)
                    else:
                        # logger.info(f"Filtered out low confidence detection: {label} (threshold: {self.confidence_threshold})")
                        pass
                except ValueError:
                    # If parsing confidence fails, keep the detection
                    filtered_masks.append(masks[i])
                    filtered_labels.append(label)
                    filtered_indices.append(i)
            else:
                filtered_masks.append(masks[i])
                filtered_labels.append(label)
                filtered_indices.append(i)
        
        # Convert filtered results back to numpy arrays
        if filtered_masks:
            filtered_masks = np.stack(filtered_masks, axis=0)
        else:
            # Handle case when no detections pass the filter
            if masks.shape == (0,):
                # Original masks is empty, use default dimensions
                filtered_masks = np.empty((0, self.height, self.width), dtype=np.uint8)
            else:
                # Use original masks dimensions
                filtered_masks = np.empty((0, masks.shape[1], masks.shape[2]), dtype=masks.dtype)
        
        if hasattr(self.current_detections, 'class_id'):
            import supervision as sv
            if len(filtered_indices) > 0:
                # Create new supervision Detections with filtered indices
                filtered_class_ids = self.current_detections.class_id[filtered_indices] if self.current_detections.class_id is not None else None
                filtered_xyxy = self.current_detections.xyxy[filtered_indices] if self.current_detections.xyxy is not None else None
                filtered_confidence = self.current_detections.confidence[filtered_indices] if self.current_detections.confidence is not None else None
                filtered_mask = self.current_detections.mask[filtered_indices] if self.current_detections.mask is not None else None
                
                self.current_detections = sv.Detections(
                    xyxy=filtered_xyxy,
                    class_id=filtered_class_ids,
                    confidence=filtered_confidence,
                    mask=filtered_mask
                )
            else:
                # No detections passed the filter, create empty Detections
                self.current_detections = sv.Detections.empty()
        
        # logger.info(f"Filtered detections: {len(filtered_labels)}/{len(labels)} kept")
        # logger.info("current step detected classes (after filtering): ", filtered_labels)
        
        class_names = self._process_labels(filtered_labels)
        masks = self._process_masks(filtered_masks, class_names)
        
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
        if masks.shape[0] > 0:  # Check if there are any masks
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
        # logger.info('max:',depth.max())
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99 # turn too far pixels to invalid
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 1.0 # then turn all invalid pixels to vision_range(100)
        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0
        
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
    
    def _flush_rgb_buffer(self):
        for combined_image, episode_id, filename in self.rgb_buffer:
            if isinstance(combined_image, Image.Image):
                img_folder = os.path.join(self.rgb_save_dir, episode_id)
                img_path = os.path.join(img_folder, filename)
                if not os.path.exists(img_folder):
                    os.makedirs(img_folder)
                combined_image.save(img_path)
        self.rgb_buffer = []
    
    def _save_rgb_frame(self, obs: Observations, step: int, episode_id: str = None, target_coords: tuple = None):
        """Save RGB frame with metadata, instruction text and top-down map"""
        if not self.save_rgb:
            return
            
        rgb_image = obs['rgb']
        episode_id = episode_id or self.current_episode_id
        episode_id = str(episode_id) if not isinstance(episode_id, str) else episode_id

        metadata = {
            'episode_id': episode_id,
            'step': step,
            'timestamp': time.time(),
            'instruction': getattr(self, 'instruction', ''),
            'pose': obs.get('sensor_pose', None),
            'destination': getattr(self, 'destination', 'unknown'),
            'action': getattr(self, '_action', 'unknown')
        }

        
        self._save_individual_rgb(rgb_image, episode_id, step)

        # logger.info(target_coords)
        # Create combined image with RGB, top-down map, and value map heatmap
        combined_image = self._create_combined_image(rgb_image, metadata, step, target_coords)
        self.rgb_buffer.append((combined_image, episode_id, f"combined_step{step:04d}.png"))

        # if len(self.rgb_buffer) >= 20:
        self._flush_rgb_buffer()

        self.all_rgb_data.append(metadata)
    
    def _save_individual_rgb(self, rgb_image, episode_id, step):
        """Save individual RGB image as a separate file"""
        try:
            episode_id = str(episode_id) if episode_id else "unknown"
            
            # Create episode-specific RGB folder
            rgb_folder = os.path.join(self.rgb_save_dir, episode_id, "rgb_only")
            os.makedirs(rgb_folder, exist_ok=True)
            
            # Convert RGB image to proper format
            if isinstance(rgb_image, np.ndarray):
                rgb_display = rgb_image.copy()
            else:
                rgb_display = np.array(rgb_image)
            
            # Save RGB image
            filename = f"rgb_step{step:04d}.png"
            filepath = os.path.join(rgb_folder, filename)
            
            # Convert RGB to BGR for cv2 saving (cv2 expects BGR)
            if len(rgb_display.shape) == 3 and rgb_display.shape[2] == 3:
                rgb_bgr = cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, rgb_bgr)
            else:
                # Fallback: save as PIL Image
                if isinstance(rgb_image, np.ndarray):
                    Image.fromarray(rgb_image).save(filepath)
                else:
                    rgb_image.save(filepath)
            
        except Exception as e:
            logger.info(f"Error saving individual RGB: {e}")
    
    def _save_waypoint_panorama_rgb(self, panorama_frames, waypoint_id, step):
        """Save the 4 directional RGB images for a waypoint"""
        try:
            episode_id = str(self.current_episode_id) if self.current_episode_id else "unknown"
            
            # Create waypoint-specific folder
            waypoint_folder = os.path.join(self.rgb_save_dir, episode_id, "waypoints", f"waypoint_{waypoint_id:02d}_step{step:04d}")
            os.makedirs(waypoint_folder, exist_ok=True)
            
            # Direction names for the 4 cardinal directions
            direction_names = ['forward', 'left', 'behind', 'right']
            
            for i, frame in enumerate(panorama_frames):
                if i >= 4:  # Only save first 4 frames (forward, left, behind, right)
                    break
                    
                rgb_image = frame['rgb']
                direction_name = direction_names[i]
                angle = frame.get('angle', i * 90)
                
                # Convert RGB image to proper format
                if isinstance(rgb_image, np.ndarray):
                    rgb_display = rgb_image.copy()
                else:
                    rgb_display = np.array(rgb_image)
                
                # Save RGB image with direction name
                filename = f"{direction_name}_{angle:03d}deg.png"
                filepath = os.path.join(waypoint_folder, filename)
                
                # Convert RGB to BGR for cv2 saving
                if len(rgb_display.shape) == 3 and rgb_display.shape[2] == 3:
                    rgb_bgr = cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filepath, rgb_bgr)
                else:
                    # Fallback: save as PIL Image
                    if isinstance(rgb_image, np.ndarray):
                        Image.fromarray(rgb_image).save(filepath)
                    else:
                        rgb_image.save(filepath)
            
            logger.info(f"Saved waypoint {waypoint_id} panorama RGB images to: {waypoint_folder}")
            
        except Exception as e:
            logger.info(f"Error saving waypoint panorama RGB: {e}")
    
    def _create_combined_image(self, rgb_image, metadata, step, target_coords=None, reasoning_info=None):
        """Create a combined image with RGB on left, VLM map on right, instruction text, and reasoning info
        
        Args:
            rgb_image: RGB observation image
            metadata: Metadata dict containing episode info
            step: Current step number  
            target_coords: Tuple of (target_map_x, target_map_y) or None
            reasoning_info: Dict containing reasoning and progress_analysis from LLM
        """
        # Create goal tensor for visualization if target coordinates are available
        goal_tensor = None
        if target_coords is not None and target_coords[0] is not None and target_coords[1] is not None:
            goal_tensor = torch.tensor([target_coords[0], target_coords[1]], dtype=torch.float32)
        
        # Generate VLM map
        vlm_map = self.mapping_module.create_vlm_map_from_state(
            self.current_episode_id, 0, goal_tensor, self.detected_classes, step, 
            output_size=(1024, 1024), visited_targets=self.visited_targets, display_last=True
        ).copy()

        cv2.imwrite(f'saved_rgb_images/{self.current_episode_id}/debug_vlm_map_step{step:04d}.png', vlm_map)

        # Convert RGB image to proper format if needed
        if isinstance(rgb_image, np.ndarray):
            rgb_display = rgb_image.copy()
        else:
            rgb_display = np.array(rgb_image)
        
        # Ensure RGB is in correct format (H, W, 3)
        if len(rgb_display.shape) == 3 and rgb_display.shape[2] == 3:
            pass  # Already correct format
        else:
            logger.info(f"Warning: Unexpected RGB shape: {rgb_display.shape}")
        
        # Convert VLM map from BGR to RGB for consistent display
        if len(vlm_map.shape) == 3:
            vlm_map_display = cv2.cvtColor(vlm_map, cv2.COLOR_BGR2RGB)
        else:
            vlm_map_display = vlm_map.copy()
        
        # Get target height from RGB image
        target_height = rgb_display.shape[0]
        
        # Resize VLM map to match RGB height while preserving aspect ratio
        vlm_old_h, vlm_old_w = vlm_map_display.shape[:2]
        if vlm_old_h == 0 or vlm_old_w == 0:
            vlm_target_width = target_height
        else:
            vlm_target_width = int(float(target_height) / vlm_old_h * vlm_old_w)
        
        # Resize VLM map
        vlm_map_resized = cv2.resize(
            vlm_map_display, 
            (vlm_target_width, target_height), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Pad images to same height with white background
        def pad_to_height(img, target_height):
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)
            if img.shape[0] < target_height:
                pad_height = target_height - img.shape[0]
                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                return np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=255)
            return img
        
        rgb_padded = pad_to_height(rgb_display, target_height)
        vlm_map_padded = pad_to_height(vlm_map_resized, target_height)
        
        # Combine horizontally: RGB on left, VLM map on right
        frame = np.concatenate((rgb_padded, vlm_map_padded), axis=1)
        
        # Prepare metadata for text display
        instruction = metadata.get('instruction', 'No instruction')
        destination = metadata.get('destination', 'unknown')
        action = metadata.get('action', 'unknown')
        
        action_display = 'UNKNOWN'
        try:
            if isinstance(action, dict): 
                action = action.get('action', '')
            
            if isinstance(action, str) and action.isdigit():
                action = int(action)
            elif isinstance(action, str):
                pass
            
            # Try to get display action text
            if isinstance(action, int) and 0 <= action < len(ACTION_TO_TEXT):
                action_display = ACTION_TO_DISPLAY.get(action, ACTION_TO_TEXT[action].upper())
            elif isinstance(action, str):
                action_display = ACTION_TO_DISPLAY.get(action, action.upper())
            else:
                action_display = 'UNKNOWN'
        except Exception as e:
            logger.info(f"Error parsing action: {e}")
            action_display = 'UNKNOWN'

        # Create instruction area - increased height to accommodate reasoning info
        instruction_height = 360  # Further increased from 280 to accommodate longer reasoning text
        white_bg = np.ones((instruction_height, frame.shape[1], 3), dtype=np.uint8) * 255
        
        # Text rendering setup
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        line_height = 25
        
        # Function to wrap long text
        def wrap_text(text, max_width, font, font_scale, thickness):
            if not text:
                return [text]
            
            words = text.split(' ')
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
                
                if text_size[0] <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        lines.append(word)
            
            if current_line:
                lines.append(current_line)
            
            return lines if lines else [text]
        
        # Calculate max width for text
        max_text_width = frame.shape[1] - 30
        
        y_pos = 20
        base_text_lines = [
            f"Step: {step} | Episode: {metadata.get('episode_id', 'N/A')} | ACTION: {action_display}",
            f"Destination: {destination}",
            "",
            f"Instruction: {instruction}"
        ]
        
        # Draw text lines
        for line in base_text_lines:
            if line.strip():
                color = (0, 0, 0)
                current_font_scale = font_scale
                current_thickness = thickness
                
                if "Step:" in line and "ACTION:" in line:
                    color = (0, 0, 255)
                    current_font_scale = 0.7
                    current_thickness = 2
                elif "Destination:" in line:
                    color = (0, 165, 255)
                elif "Instruction:" in line:
                    color = (255, 0, 0)
                    # Wrap instruction text
                    if len(line) > 20:
                        instruction_text = line.replace("Instruction: ", "")
                        wrapped_instruction = wrap_text(instruction_text, max_text_width - 120, font, current_font_scale, current_thickness)
                        
                        # Draw "Instruction: " first
                        cv2.putText(
                            white_bg, 
                            "Instruction: ", 
                            (15, y_pos), 
                            font, 
                            current_font_scale, 
                            color, 
                            current_thickness, 
                            lineType=cv2.LINE_AA
                        )
                        y_pos += line_height
                        
                        # Draw wrapped lines with indentation
                        for wrapped_line in wrapped_instruction:
                            cv2.putText(
                                white_bg, 
                                wrapped_line, 
                                (30, y_pos), 
                                font, 
                                current_font_scale, 
                                color, 
                                current_thickness, 
                                lineType=cv2.LINE_AA
                            )
                            y_pos += line_height
                        continue
                
                # Regular text drawing
                cv2.putText(
                    white_bg, 
                    line, 
                    (15, y_pos), 
                    font, 
                    current_font_scale, 
                    color, 
                    current_thickness, 
                    lineType=cv2.LINE_AA
                )
            y_pos += line_height
        
        # Add reasoning and progress analysis information if available
        if reasoning_info:
            y_pos += 5  # Add some spacing
            
            # Draw progress analysis
            progress = reasoning_info.get('progress_analysis', '')
            if progress:
                cv2.putText(
                    white_bg, 
                    "Progress Analysis: ", 
                    (15, y_pos), 
                    font, 
                    0.5,  # Slightly smaller font
                    (0, 128, 0),  # Dark green
                    1, 
                    lineType=cv2.LINE_AA
                )
                y_pos += line_height
                
                # Wrap progress analysis text
                wrapped_progress = wrap_text(progress, max_text_width - 60, font, 0.5, 1)
                for wrapped_line in wrapped_progress:
                    if y_pos < instruction_height - 50:  # Give more space for reasoning below
                        cv2.putText(
                            white_bg, 
                            wrapped_line, 
                            (30, y_pos), 
                            font, 
                            0.5, 
                            (0, 128, 0), 
                            1, 
                            lineType=cv2.LINE_AA
                        )
                        y_pos += 18  # Slightly tighter line spacing for better space utilization
            
            # Draw reasoning
            reasoning = reasoning_info.get('reasoning', '')
            if reasoning and y_pos < instruction_height - 70:  # More generous space check
                y_pos += 5  # Add some spacing
                cv2.putText(
                    white_bg, 
                    "Reasoning: ", 
                    (15, y_pos), 
                    font, 
                    0.5,  # Slightly smaller font
                    (128, 0, 128),  # Purple
                    1, 
                    lineType=cv2.LINE_AA
                )
                y_pos += line_height
                
                # Wrap reasoning text
                wrapped_reasoning = wrap_text(reasoning, max_text_width - 60, font, 0.5, 1)
                for wrapped_line in wrapped_reasoning:
                    if y_pos < instruction_height - 15:  # Use almost all available space
                        cv2.putText(
                            white_bg, 
                            wrapped_line, 
                            (30, y_pos), 
                            font, 
                            0.5, 
                            (128, 0, 128), 
                            1, 
                            lineType=cv2.LINE_AA
                        )
                        y_pos += 18  # Same tighter spacing as progress analysis
        
        # Create header for the two panels
        header_height = 25
        header_bg = np.ones((header_height, frame.shape[1], 3), dtype=np.uint8) * 240
        
        rgb_width = rgb_padded.shape[1]
        vlm_width = vlm_map_padded.shape[1]
        
        headers = ["RGB View", "VLM Navigation Map"]
        header_colors = [(0, 100, 0), (0, 0, 139)]  # Dark green, Dark red
        
        # Position headers in center of each panel
        x_positions = [
            rgb_width // 2,  # RGB center
            rgb_width + vlm_width // 2  # VLM map center
        ]
        
        for i, (header, color, x_pos) in enumerate(zip(headers, header_colors, x_positions)):
            text_size = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            x_centered = max(0, x_pos - text_size[0] // 2)
            
            cv2.putText(
                header_bg,
                header,
                (x_centered, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                lineType=cv2.LINE_AA
            )
        
        # Combine all components: header + main frame + instruction area
        final_frame = np.concatenate([header_bg, frame, white_bg], axis=0)
        
        return Image.fromarray(final_frame.astype(np.uint8))
    
    def _save_all_rgb_metadata(self):
        """Save all RGB metadata to a single file"""
        if not self.save_rgb or not self.all_rgb_data:
            return
            
        metadata_file = os.path.join(self.rgb_save_dir, "all_episodes_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.all_rgb_data, f, indent=2, default=str)
        
        # logger.info(f"Saved RGB metadata for {len(self.all_rgb_data)} frames to {metadata_file}")
    
    def _create_rgb_summary(self):
        """Create a summary of all saved RGB files for each episode"""
        if not self.save_rgb:
            return
        
        try:
            episode_id = str(self.current_episode_id) if self.current_episode_id else "unknown"
            episode_folder = os.path.join(self.rgb_save_dir, episode_id)
            
            if not os.path.exists(episode_folder):
                return
            
            # Create summary info
            summary = {
                'episode_id': episode_id,
                'instruction': getattr(self, 'instruction', ''),
                'total_waypoints': len(self.visited_targets),
                'saved_files': {
                    'combined_images': [],
                    'individual_rgb': [],
                    'waypoint_panoramas': [],
                    'bbox_annotations': []
                }
            }
            
            # Scan for different types of saved files
            for root, dirs, files in os.walk(episode_folder):
                rel_path = os.path.relpath(root, episode_folder)
                
                for file in files:
                    file_path = os.path.join(rel_path, file) if rel_path != '.' else file
                    
                    if file.startswith('combined_step'):
                        summary['saved_files']['combined_images'].append(file_path)
                    elif file.startswith('rgb_step') and 'rgb_only' in root:
                        summary['saved_files']['individual_rgb'].append(file_path)
                    elif file.startswith('bbox_step'):
                        summary['saved_files']['bbox_annotations'].append(file_path)
                    elif 'waypoints' in root and (file.endswith('.png') or file.endswith('.jpg')):
                        summary['saved_files']['waypoint_panoramas'].append(file_path)
            
            # Sort file lists
            for file_type in summary['saved_files']:
                summary['saved_files'][file_type].sort()
            
            # Save summary file
            summary_file = os.path.join(episode_folder, "rgb_files_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Created RGB files summary for episode {episode_id}: {len(summary['saved_files']['individual_rgb'])} RGB files, "
                       f"{len(summary['saved_files']['waypoint_panoramas'])} waypoint panorama files")
            
        except Exception as e:
            logger.info(f"Error creating RGB summary: {e}")
    
    def _process_classes(self, base_class: List, target_class: List) -> List:
        for item in target_class:
            if item in base_class:
                base_class.remove(item)
        base_class.extend(target_class)
        
        return base_class
    
    def img_to_base64(self, img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64
    
    def query_llm(self, obs: Observations, progress_analysis: str = None):
        try:
            instruction = self.instruction
            rgb_image = obs['rgb']
            
            if isinstance(rgb_image, np.ndarray):
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                img = Image.fromarray(rgb_image)
            else:
                img = rgb_image
            
            # Build visited targets history string
            visited_targets_str = ""
            # visited_targets
            if len(self.visited_targets) > 1:
                visited_targets_str = f"\n\nPreviously visited targets:\n"
                for i, target in enumerate(self.visited_targets[:-1], 0):
                    visited_targets_str += f"{i}. {target['description']} (Step {target['step']})\n"
            else:
                visited_targets_str = "\n\nNo targets visited yet."
            
            content = [{
                'type': 'text',
                'text': f'You are a robot performing navigation task. Look at this image and help the robot navigate.'
            }]

            img_base64 = self.img_to_base64(img)
            
            content.append({"type": "image_url", 'image_url': {'url':f"data:image/png;base64,{img_base64}"}})

            
            progress_info = ""
            if progress_analysis:
                progress_info = f"\nProgress Analysis from Navigation Decision: {progress_analysis}\n"

            prompt = f"""Navigation Task: "{instruction}"

Current situation:
- Step: {self.current_step}
- Image size: {self.width}x{self.height} pixels{visited_targets_str}{progress_info}

Your task:
1. Analyze at what stage the current instruction has been completed and what should be done next
2. Identify the most relevant target object/area for what you should do next. Specify ONLY ONE. And it should not be too close to you.
3. Decide if the robot should STOP (if task is completed or very close to final goal)

Response format (JSON):
{{
    "progress": "<assessment of how close to completing the instruction>",
    "reasoning": "<brief explanation of decision>",
    "action": "NAVIGATE" or "STOP",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<description of target object>"
}}

Guidelines:
- If you see the final destination mentioned in instruction, consider STOP action
- If already very close to the goal object, choose STOP
- If still need to navigate, choose NAVIGATE and provide bounding box of next target
- Target description should be specific and clear
- Consider the instruction completion progress based on visited targets
- Use the progress analysis to inform your decision"""
            
            content.append({
                "type": "text", 
                "text": prompt
            })
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
        
            
            # Use the API client to generate response
            output_text = self.model.generate(
                messages=messages,
                max_new_tokens=1024,
                temperature=0
            )
            logger.info('LLM Output:')
            logger.info("%s", output_text)
            
            # Try to parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
                
                # Extract action decision
                action_decision = response_data.get('action', 'NAVIGATE').upper()
                
                # Extract bbox_2d in [x1, y1, x2, y2] format
                bbox_2d = response_data.get('bbox_2d', [self.width // 4, self.height // 4, 3 * self.width // 4, 3 * self.height // 4])
                
                # Ensure we have 4 coordinates
                if len(bbox_2d) >= 4:
                    x1, y1, x2, y2 = bbox_2d[:4]
                else:
                    # Fallback if bbox_2d is malformed
                    x1, y1, x2, y2 = self.width // 4, self.height // 4, 3 * self.width // 4, 3 * self.height // 4
                
                # Convert to x, y, width, height format for internal use
                x = int(x1)
                y = int(y1)
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                bbox = {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'target': response_data.get('target', 'unknown target'),
                    'action': action_decision,
                    'reasoning': response_data.get('reasoning', 'No reasoning provided'),
                    'progress': response_data.get('progress', 'Unknown progress')
                }
                
                # Ensure bbox is within image bounds
                bbox['x1'] = max(0, min(bbox['x1'], self.width - 1))
                bbox['y1'] = max(0, min(bbox['y1'], self.height - 1))
                bbox['x2'] = max(bbox['x1'] + 1, min(bbox['x2'], self.width))
                bbox['y2'] = max(bbox['y1'] + 1, min(bbox['y2'], self.height))
                
                # Update x, y, width, height based on bounded coordinates
                bbox['x'] = bbox['x1']
                bbox['y'] = bbox['y1']
                bbox['width'] = bbox['x2'] - bbox['x1']
                bbox['height'] = bbox['y2'] - bbox['y1']
                
                # logger.info("Parsed response:", bbox)
                # logger.info(f"BBox 2D format - x1:{bbox['x1']}, y1:{bbox['y1']}, x2:{bbox['x2']}, y2:{bbox['y2']}")
                # logger.info(f"Action: {action_decision}, Target: {bbox['target']}")
                # logger.info(f"Reasoning: {bbox['reasoning']}")
                # logger.info(f"Progress: {bbox['progress']}")
                
                # Record this target if it's a new navigation target
                if action_decision == 'NAVIGATE' and bbox['target'] != 'unknown target':
                    target_record = {
                        'step': self.current_step,
                        'description': bbox['target'],
                        'bbox': {
                            'x': bbox['x'], 
                            'y': bbox['y'], 
                            'width': bbox['width'], 
                            'height': bbox['height'],
                            'x1': bbox['x1'],
                            'y1': bbox['y1'],
                            'x2': bbox['x2'],
                            'y2': bbox['y2']
                        },
                        'reasoning': bbox['reasoning']
                        
                    }
                    
                    if self.visited_targets is not None:
                        self.visited_targets[-1].update(target_record)
                        # logger.info(f"Added target to history: {bbox['target']}")
                        # logger.info(self.visited_targets[-1])

                
                # Save RGB image with bounding box annotation
                self._save_rgb_with_bbox(rgb_image, bbox, instruction)
                
                return bbox
            else:
                logger.info("Failed to parse JSON response, using default")
                
        except Exception as e:
            logger.info(f"Error in query_llm: {e}")
            
        # Fallback bbox (center of image)
        x1_fallback = self.width // 4
        y1_fallback = self.height // 4
        x2_fallback = 3 * self.width // 4
        y2_fallback = 3 * self.height // 4
        
        fallback_bbox = {
            'x': x1_fallback,
            'y': y1_fallback,
            'width': x2_fallback - x1_fallback,
            'height': y2_fallback - y1_fallback,
            'x1': x1_fallback,
            'y1': y1_fallback,
            'x2': x2_fallback,
            'y2': y2_fallback,
            'target': 'fallback target',
            'action': 'NAVIGATE',
            'reasoning': 'Fallback due to parsing error',
            'progress': 'Unknown due to error'
        }
        
        # Save RGB image with fallback bounding box
        self._save_rgb_with_bbox(rgb_image, fallback_bbox, instruction)
        
        return fallback_bbox
    
    def _save_rgb_with_bbox(self, rgb_image, bbox, instruction):
        """Save RGB image with bounding box annotation as a separate file"""
        if not self.save_rgb:
            return
            
        try:
            # Convert RGB image to proper format
            if isinstance(rgb_image, np.ndarray):
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                # Make a copy to avoid modifying the original
                annotated_image = rgb_image.copy()
            else:
                annotated_image = np.array(rgb_image)
            
            # Draw bounding box
            x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            target_description = bbox.get('target', 'target')
            action_decision = bbox.get('action', 'NAVIGATE')
            
            # Choose color based on action decision
            box_color = (0, 255, 0) if action_decision == 'NAVIGATE' else (0, 0, 255)  # Green for navigate, Red for stop
            
            # Draw rectangle with thicker border for better visibility
            cv2.rectangle(annotated_image, (x, y), (x + width, y + height), box_color, 4)
            
            # Create enhanced label with action info
            label = f"{action_decision}: {target_description}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9  # Slightly larger font
            thickness = 2
            
            # Get text size to create background with more padding
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Calculate text position with more spacing
            padding = 15  # Increased padding
            text_y = y - padding if y - padding > text_height + padding else y + height + text_height + padding
            
            # Draw text background with more generous padding using same color as box
            bg_x1 = max(0, x - 5)
            bg_y1 = max(0, text_y - text_height - padding)
            bg_x2 = min(self.width, x + text_width + padding * 2)
            bg_y2 = min(self.height, text_y + baseline + 5)
            
            cv2.rectangle(annotated_image, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, -1)
            
            # Draw text with better positioning
            text_x = max(padding, x + 5)
            cv2.putText(annotated_image, label, (text_x, text_y - 5), 
                       font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
            
            # Save to episode-specific folder with chronological naming
            episode_id = str(self.current_episode_id) if self.current_episode_id else "unknown"
            img_folder = os.path.join(self.rgb_save_dir, episode_id)
            
            # Create episode folder if it doesn't exist
            os.makedirs(img_folder, exist_ok=True)
            
            # Use current step for chronological ordering
            filename = f"bbox_step{self.current_step:04d}.png"
            filepath = os.path.join(img_folder, filename)
            
            # Convert RGB to BGR for cv2 saving (cv2 expects BGR)
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, annotated_image_bgr)
            
            # logger.info(f"Saved RGB with bbox annotation: {filepath}")
            
        except Exception as e:
            logger.info(f"Error saving RGB with bbox: {e}")
    
    def _process_llm_reply(self, obs: Observations, first: bool = True):
        """Process initial instruction and setup basic navigation parameters"""
        self.instruction = obs['instruction']['text']
        
        self.destination = "goal"
        self.classes = self.base_classes.copy()
    
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
    
    def _maps_initialization(self):
        obs = self.envs.reset() #type(obs): list
        self.rgb_history.append(obs[0]['rgb'])
        self._process_llm_reply(obs[0])
        self.current_episode_id = self.envs.current_episodes()[0].episode_id
        # logger.info("current episode id: ", self.current_episode_id)

        self._save_rgb_frame(obs[0], 0, self.current_episode_id)
        
        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        batch_obs = self._batch_obs(obs)
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
        self.mapping_module(batch_obs, poses)
        full_map, full_pose, _ = self.mapping_module.update_map(0, self.detected_classes, self.current_episode_id)
        self.mapping_module.one_step_full_map.fill_(0.)
        self.mapping_module.one_step_local_map.fill_(0.)
    
    def get_panorama(self, obs: Observations, step: int):
        
        panorama_frames = []
        
        for turn_step in range(1, 12+1):
            
            turn_action = [{"action": HabitatSimActions.TURN_LEFT}]  
            turn_outputs = self.envs.step(turn_action)
            turn_obs, _, turn_dones, turn_infos = [list(x) for x in zip(*turn_outputs)]
            
            if turn_dones[0]:
                # logger.info("Episode ended during panorama collection")
                # Return signal that episode is done - caller should handle this
                return {'turn_direction': 'episode_done', 'episode_finished': True}
                
            
            panorama_frames.append({
                'rgb': turn_obs[0]['rgb'].copy(),
                'angle': turn_step * 30 % 360,  
                'step': turn_step
            })
            
            
            batch_obs = self._batch_obs(turn_obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in turn_obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses)
            full_map, full_pose, one_step_full_map =\
                self.mapping_module.update_map(step + turn_step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
        panorama_frames = [panorama_frames[-1]] + panorama_frames[:-1]
        # logger.info([x['angle'] for x in panorama_frames])
        # logger.info(f"Collected {len(panorama_frames)} panorama frames")
        
        return panorama_frames[::3]
    
    def decide_turn(self, panorama_frames):
        try:
            instruction = self.instruction
            
            content = [{
                'type': 'text',
                'text': f'You are analyzing a 360-degree panorama view to help a robot decide which direction to turn. Current step: {self.current_step}.'
            }]
            
            view_definitions = [
                {'angle': 0,   'name': 'front',  'label': 'Image 1: The current FORWARD view.'},
                {'angle': 90,  'name': 'left',   'label': 'Image 2: The view after turning 90° to the LEFT.'},
                {'angle': 180, 'name': 'behind', 'label': 'Image 3: The view directly BEHIND (180° turn).'},
                {'angle': 270, 'name': 'right',  'label': 'Image 4: The view after turning 90° to the RIGHT.'}
            ]

            
            for view in view_definitions:
                angle = view['angle']
                frame_idx = angle // 90
                if frame_idx < len(panorama_frames):
                    frame = panorama_frames[frame_idx]
                    rgb_image = frame['rgb']
                    
                    if isinstance(rgb_image, np.ndarray):
                        if rgb_image.dtype != np.uint8:
                            rgb_image = (rgb_image * 255).astype(np.uint8)
                        img = Image.fromarray(rgb_image)
                    else:
                        img = rgb_image
                    # img.save(f"{view['label']}.png")
                    img_base64 = self.img_to_base64(img)
                    content.append({"type": "image_url", 'image_url':{'url': f"data:image/png;base64,{img_base64}"}})
                    content.append({"type": "text", "text": view['label']})
            
            
            # Build visited targets history string
            visited_targets_str = ""
            if self.visited_targets:
                target_descriptions = [target['description'] for target in self.visited_targets[:-1] if 'description' in target]
                visited_targets_str = f"\n- Visited targets: {', '.join(target_descriptions)}"
            else:
                visited_targets_str = "\n- Visited targets: None yet"
            progress_analysis = self.visited_targets[-1]['progress']
            
            prompt = f"""Navigation Task: "{instruction}"

Current situation:
- {visited_targets_str}
- Progress analysis: {progress_analysis}

You have a 360-degree panorama view of the current location. The images above show views at 0°, 90° left, 180° behind, and 90° right from the current position.

Your task:
1. Identify the most relevant target object/area for the next step of the instruction
2. Decide if the robot should turn when:
   - Current view is blocked or obstructed
   - No clear task-relevant objects are visible in the current view
   - The instruction explicitly indicates a direction change is needed

Analyze these panorama views and decide:
1. Which direction (left/right/behind) would be most beneficial for completing the navigation task?
2. Or should the robot stay in the current orientation (no turn)?

Consider:
- Current progress in completing the instruction
- What specific step should be done next
- Visibility of task-relevant objects and pathways
- Whether current view provides clear navigation options
- Potential target objects mentioned in the instruction

Response format (JSON):
{{
    "description": "<Describe pictures from four views respectively>",
    "reasoning": "<explanation for the turn decision based on task progress and visibility>",
    "turn_direction": "left" or "right" or "no_turn" or "behind",
}}

Guidelines:
- "left" means turn left from current position
- "right" means turn right from current position  
- "no_turn" means current orientation is best for continuing the task
- "behind" means turn 180 degrees to face the opposite direction
- Turn when current view is obstructed or lacks task-relevant targets
- Consider the instruction completion progress when deciding direction
"""
            
            content.append({
                "type": "text", 
                "text": prompt
            })
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # logger.info(prompt)

            
            
            # Use the API client to generate response
            output_text = self.model.generate(
                messages=messages,
                max_new_tokens=1024,
                temperature=0,
                use_secondary=True
            )
            
            logger.info("Panorama analysis response:")
            logger.info("%s", output_text)
            
            import json
            import re
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
                
                turn_direction = response_data.get('turn_direction', 'no_turn')
                reasoning = response_data.get('reasoning', 'No reasoning provided')
                
                # logger.info(f"Progress analysis: {progress_analysis}")
                # logger.info(f"Panorama decision: {turn_direction}")
                # logger.info(f"Reasoning: {reasoning}")
                
                turn_angle = 90.0

                if turn_direction == 'behind':
                    turn_angle = 180.0
                elif turn_direction == 'no_turn':
                    turn_angle = 0.0

                return {
                    'turn_direction': turn_direction,
                    'reasoning': reasoning,
                    'best_angle_estimate': turn_angle,
                }
            else:
                logger.info("Failed to parse panorama analysis JSON response")
                
        except Exception as e:
            logger.info(f"Error in panorama analysis: {e}")
            import traceback
            traceback.print_exc()
        
        # Fallback: no turn
        return {
            'turn_direction': 'no_turn',
            'reasoning': 'Fallback due to analysis error',
            'best_angle_estimate': 0,
            'episode_finished': False
        }
    
    def _look_around(self):
        full_pose, obs, dones, infos = None, None, None, None
        for step in range(0, 12):
            self._action = HabitatSimActions.TURN_LEFT
            actions = []
            for _ in range(self.config.NUM_ENVIRONMENTS):
                actions.append({"action": HabitatSimActions.TURN_LEFT})
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            # self.rgb_history.append(obs[0]['rgb'])
            if dones[0]:
                return full_pose, obs, dones, infos
            
            # Save RGB frame during look around phase
            self._save_rgb_frame(obs[0], step, self.current_episode_id)
            
            batch_obs = self._batch_obs(obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses)
            full_map, full_pose, one_step_full_map =\
                self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            self.traversible, self.floor, self.frontiers = self._process_map(step, full_map[0])
            self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])

        return full_pose, obs, dones, infos
    
    def _use_keyboard_control(self):
        a = input("action:")
        if a == 'w':
           return {"action": 1}
        elif a == 'a':
            return {"action": 2}
        elif a == 'd':
            return {"action": 3}
        else:
            return {"action": 0}
    
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
        self.current_action = None
        
        self.rgb_history = []
        self.current_action_description = None
        
        self.all_rgb_data = []
        self.rgb_buffer = []
        
        # Reset target tracking
        self.visited_targets = []
        self.current_step = 0
        self.backtrack_steps = 0
        
        self.policy.reset()
        self.mapping_module.reset()
        self.history_module.reset()
    
    def _get_camera_intrinsics(self) -> np.ndarray:
        """Get camera intrinsics matrix for depth projection"""
        hfov = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        width = self.width
        height = self.height
        vfov = 2 * np.arctan(height / width * np.tan(hfov / 2))

        fx = width / (2 * np.tan(np.deg2rad(hfov / 2)))
        fy = height / (2 * np.tan(np.deg2rad(vfov / 2)))
        cx = width / 2
        cy = height / 2

        intrinsics = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])
        return intrinsics
    
    def navigate_or_backtrack(self):
        instruction = self.instruction
        
        
        panorama_images = self.visited_targets[-1]['panorama_frames'] if self.visited_targets else []
        
        
        history_content = []
        
        
        for i, target in enumerate(self.visited_targets[:-1]):  
            if 'init_image' in target:
                history_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                history_content.append({"type": "text", "text": f"Waypoint {i}: Initial view"})
            
            if 'turn_action' in target:
                history_content.append({"type": "text", "text": f"Action: {target['turn_action']}"})
            
            if 'dir_image' in target:
                history_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                history_content.append({"type": "text", "text": f"After turn view"})
            
            if 'description' in target:
                history_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})
            
        
        
        current_views = []
        view_definitions = [
            {'angle': 0, 'name': 'forward', 'label': 'Current FORWARD view'},
            {'angle': 90, 'name': 'left', 'label': 'View after turning LEFT'},
            {'angle': 180, 'name': 'behind', 'label': 'View after turning BEHIND'},
            {'angle': 270, 'name': 'right', 'label': 'View after turning RIGHT'}
        ]
        
        for view in view_definitions:
            angle = view['angle']
            frame_idx = angle // 90
            if frame_idx < len(panorama_images):
                rgb_image = panorama_images[frame_idx]['rgb']
                if isinstance(rgb_image, np.ndarray):
                    if rgb_image.dtype != np.uint8:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_image)
                else:
                    img = rgb_image
                current_views.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
                current_views.append({"type": "text", "text": view['label']})
        
        
        
        num_waypoints = len([t for t in self.visited_targets[:-1] if 'description' in t])
        should_consider_backtrack = 1
        
        
        content = [{"type": "text", "text": f"Navigation Task: \"{instruction}\"\n\nNavigation History:"}]
        content.extend(history_content)
        content.append({"type": "text", "text": "\nCurrent 4-directional views:"})
        content.extend(current_views)
        
        if should_consider_backtrack and num_waypoints > 0:
            
            waypoint_list = ""
            for i, target in enumerate(self.visited_targets[:-1]):
                if 'description' in target:
                    waypoint_list += f"  - Waypoint {i}: {target['description']}\n"
            
            prompt = f"""Based on the navigation history and current 4-directional views, decide the next action:

Available waypoints for backtracking:
{waypoint_list}

Choose one of these actions:
1. navigate to forward - continue straight ahead
2. navigate to left - turn left and go forward  
3. navigate to right - turn right and go forward
4. navigate to behind - turn around and go forward
5. backtrack to <waypoint_id> - return to a previous waypoint

Response format (JSON):
{{
    "progress_analysis": "<assessment of current progress toward instruction completion>",
    "reasoning": "<explanation of chosen action>",
    "action": "navigate to forward|left|right|behind" or "backtrack to <waypoint_id>"
}}

Guidelines:
- Consider instruction completion progress
- Backtrack only if current path seems unproductive or dead-end
- Choose direction that best advances toward goal"""
        else:
            
            prompt = f"""Based on the navigation history and current 4-directional views, decide the next navigation direction:

Choose the best direction:
1. navigate to forward - continue straight ahead
2. navigate to left - turn left and go forward
3. navigate to right - turn right and go forward  
4. navigate to behind - turn around and go forward

Response format (JSON):
{{
    "progress_analysis": "<assessment of current progress toward instruction completion>",
    "reasoning": "<explanation of chosen direction>",
    "action": "navigate to forward|left|right|behind"
}}

Guidelines:
- Consider instruction completion progress
- Choose direction that best advances toward goal
- Look for relevant objects and clear paths"""
        
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        
        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=1024,
            temperature=0,
            use_secondary=True  
        )

        logger.info('4o-response:')
        logger.info(f"{output_text}")
        
        
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            response_data = json.loads(json_match.group())
            action = response_data.get('action', 'navigate to forward')
            progress_analysis = response_data.get('progress_analysis', '')
            reasoning = response_data.get('reasoning', '')
            action = action.lower()
            if action.startswith('backtrack to'):
                waypoint_id = action.split('backtrack to ')[-1].strip()
                if waypoint_id.startswith('waypoint'):
                    waypoint_id = waypoint_id.split('waypoint')[-1].strip()
                # logger.info('Waypoint:%s', waypoint_id)
                try:
                    waypoint_id = int(waypoint_id)
                    return {
                        'action': 'BACKTRACK', 
                        'waypoint': waypoint_id,
                        'progress_analysis': progress_analysis,
                        'reasoning': reasoning
                    }
                except:
                    pass  
            
            
            if 'forward' in action:
                direction = 'forward'
            elif 'left' in action:
                direction = 'left'
            elif 'right' in action:
                direction = 'right'
            elif 'behind' in action:
                direction = 'behind'
            else:
                direction = 'forward'  
            
            return {
                'action': 'NAVIGATE',
                'direction': direction,
                'progress_analysis': progress_analysis,
                'reasoning': reasoning
            }
        
        
        return {
            'action': 'NAVIGATE',
            'direction': 'forward',
            'progress_analysis': 'Unable to analyze due to parsing error',
            'reasoning': 'Fallback to forward navigation'
        }
    
    def get_world_xz_from_pixel(
        self,
        pixel_coords: tuple = None,
        bbox: dict = None,
        depth_image: np.ndarray = None,
        full_pose: np.ndarray = None,
        camera_intrinsics: np.ndarray = None,
        agent_height: float = 0.88
    ) -> np.ndarray:
        import sys
        
        
        if bbox is not None:
            
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            
            h, w = depth_image.shape
            x1 = max(0, min(x1, w-1))
            x2 = max(x1+1, min(x2, w))
            y1 = max(0, min(y1, h-1))
            y2 = max(y1+1, min(y2, h))
            
            
            depth_roi = depth_image[y1:y2, x1:x2]
            
            
            valid_mask = (depth_roi > 0) & np.isfinite(depth_roi) & (depth_roi < 1000.0)
            valid_depths = depth_roi[valid_mask]
            
            if len(valid_depths) == 0:
                logger.info(f"Warning: no valid depth values found in the bounding box region")
                return np.array([12.0, 12.0])
            
            
            median_depth = np.median(valid_depths)
            
            
            depth_diff = np.abs(depth_roi - median_depth)
            depth_diff[~valid_mask] = np.inf  
            
            
            roi_y, roi_x = np.unravel_index(np.argmin(depth_diff), depth_diff.shape)
            
            
            u = x1 + roi_x
            v = y1 + roi_y
            depth = depth_roi[roi_y, roi_x]
            
            
            
            
            
            
        else:
            
            if pixel_coords is None:
                raise ValueError("Either pixel_coords or bbox must be provided")
            
            u, v = pixel_coords
            # logger.info('full_pose=', full_pose)
            
            
            def get_robust_depth(depth_image, u, v, window_size=5):
                h, w = depth_image.shape
                half_window = window_size // 2
                
                
                u_min = max(0, u - half_window)
                u_max = min(w, u + half_window + 1)
                v_min = max(0, v - half_window)  
                v_max = min(h, v + half_window + 1)
                
                
                depth_window = depth_image[v_min:v_max, u_min:u_max]
                
                
                valid_depths = depth_window[
                    (depth_window > 0) & 
                    np.isfinite(depth_window) & 
                    (depth_window < 1000.0)  
                ]
                
                if len(valid_depths) == 0:
                    return None
                
                
                robust_depth = np.median(valid_depths)
                
                return robust_depth
            
            depth = None
            for window_size in [3, 5, 7, 9]:
                depth = get_robust_depth(depth_image, u, v, window_size)
                if depth is not None:
                    break
            
            sys.stdout.flush()
            if depth is None or depth <= 0:
                
                return np.array([12.0, 12.0])
            
            

        
        
        K_inv = np.linalg.inv(camera_intrinsics)
        camera_coords = depth * (K_inv @ np.array([u, v, 1]))
        
        
        aligned_coords = np.array([
            camera_coords[0],
            -camera_coords[1],
            camera_coords[2]
        ])
        local_x, local_y, local_z = aligned_coords

        
        agent_x_world, agent_z_world, heading_rad = full_pose
        heading_rad = np.deg2rad(heading_rad)

        
        
        
        forward_vec = np.array([np.cos(heading_rad), np.sin(heading_rad)])
        
        right_vec = np.array([np.sin(heading_rad), -np.cos(heading_rad)])

        
        
        world_z = agent_z_world + local_z * forward_vec[1] + local_x * right_vec[1]
        world_x = agent_x_world + local_z * forward_vec[0] + local_x * right_vec[0]
        
        return np.array([world_x, world_z])
    
    def rollout(self):
        """
        Execute a whole episode using bounding box target navigation
        """
        
        self._maps_initialization()
        
        look_around_results = self._look_around()
        if look_around_results[1] is None: 
            logger.info("Episode finished during look_around. Exiting rollout.")
            if look_around_results[3]: # infos
                self._calculate_metric(look_around_results[3])
            return

        full_pose, obs, dones, infos = look_around_results

        # logger.info('Sensor pose', obs[0]['sensor_pose'])

        # logger.info("\n ========== START TO NAVIGATE ==========\n")
        
        action_list = []
        going_to_stop = False
        panorama_got = False
        navigate_or_not = False
        collided = 0
        search_destination = False
        last_pose = None
        current_pose = full_pose[0] if full_pose is not None else None
        
        target_map_x, target_map_y = None, None
        waypoint = None 
        
        
        steps_since_target_set = 0
        max_steps_to_target = 30  
        target_set_step = None  
        
        
        full_map = self.mapping_module.get_full_map()
        
        for step in range(12, self.max_step):
            import sys
            sys.stdout.flush()
            # logger.info(action_list, panorama_got)
            # =================================================================
            
            
            # =================================================================
            if dones[0]:
                self._calculate_metric(infos) 
                return
                
            logger.info(f"\nepisode:{self.current_episode_id}, step:{step}")
            # logger.info(f"instr: {self.instruction}")
            # logger.info(f"Targets visited: {len(self.visited_targets)}")

            
            last_pose = current_pose
            current_pose = full_pose[0]
            self.current_step = step
            
            position = current_pose[:2] * 100 / self.resolution
            agent_map_x, agent_map_y = int(position[0]), int(position[1])
            # logger.info("full pose: ", current_pose)

            
            self._save_rgb_frame(obs[0], step, self.current_episode_id, (target_map_x, target_map_y))
            self.rgb_history.append(obs[0]['rgb'])

            # =================================================================
            
            
            
            
            
            
            # =================================================================
            
            
            if not action_list:
                
                if target_map_x is not None and target_map_y is not None and target_set_step is not None:
                    steps_since_target_set = step - target_set_step
                    if steps_since_target_set >= max_steps_to_target:
                        # logger.info(f"Target timeout: {steps_since_target_set} steps since target set, exceeding {max_steps_to_target} limit.")
                        
                        
                        if len(self.visited_targets) > 0:
                            self.visited_targets.pop()
                            # self.visited_targets[-1]['arrival_image'] = Image.fromarray(obs[0]['rgb']) if isinstance(obs[0]['rgb'], np.ndarray) else obs[0]['rgb']
                            
                            # logger.info(f"Popped image for waypoint at step {step}")
                        
                        
                        panorama_got = False
                        navigate_or_not = False
                        target_map_x, target_map_y = None, None
                        target_set_step = None
                        steps_since_target_set = 0
                        # logger.info("Reset navigation state due to timeout.")
                
                
                if target_map_x is not None and target_map_y is not None:
                    distance_to_target = np.sqrt((target_map_x - agent_map_x)**2 + (target_map_y - agent_map_y)**2)
                    # logger.info(f"Agent: ({agent_map_x}, {agent_map_y}), Target: ({target_map_x}, {target_map_y})")
                    # logger.info(f"Distance to target: {distance_to_target:.2f} (threshold: {self.target_reached_threshold})")
                    if distance_to_target < self.target_reached_threshold:
                        # logger.info(f"Target reached! Distance: {distance_to_target:.2f}")
                        
                        
                        if len(self.visited_targets) > 0:
                            dist_calc = lambda target: np.sqrt((target['world_coords'][0] - self.visited_targets[-1]['world_coords'][0])**2 + (target['world_coords'][1] - self.visited_targets[-1]['world_coords'][1])**2) if 'world_coords' in target else float('inf')
                            for target in self.visited_targets[:-1]:
                                if dist_calc(target) < self.target_reached_threshold:
                                    # logger.info('Removed duplicate waypoint due to proximity. Distance: %f', dist_calc(target))
                                    self.visited_targets.pop()
                                    break
                        
                        
                        panorama_got = False  
                        navigate_or_not = False
                        target_map_x, target_map_y = None, None
                        target_set_step = None
                        steps_since_target_set = 0
                        # logger.info("Reset navigation state - target reached.")                    
                
                
                if target_map_x is None and not panorama_got and going_to_stop:
                    # logger.info('Final stop.')
                    action_list.append(0)  # STOP action
                elif target_map_x is None and not navigate_or_not:
                    # logger.info("Step 1: Getting panorama and deciding navigation direction...")
                    
                    current_rgb = obs[0]['rgb'].copy()
                    
                    
                    panorama_frames = self.get_panorama(obs[0], step)
                    if 'episode_finished' in panorama_frames:
                        # logger.info("Episode over detected during panorama collection.")
                        break
                    
                    
                    waypoint_id = len(self.visited_targets)
                    self.visited_targets.append({
                        'step': step,
                        'init_image': Image.fromarray(current_rgb) if isinstance(current_rgb, np.ndarray) else current_rgb,
                        'panorama_frames': panorama_frames,
                        'world_coords': (agent_map_x, agent_map_y)  
                    })
                    
                    
                    if self.save_rgb:
                        self._save_waypoint_panorama_rgb(panorama_frames, waypoint_id, step)
                    
                    
                    decision = self.navigate_or_backtrack()
                    # logger.info(f"Navigation decision: {decision}")
                    
                    if decision.get('action', 'NAVIGATE') == 'BACKTRACK':
                        target_waypoint_id = decision.get('waypoint', 0)
                        if isinstance(target_waypoint_id, int) and target_waypoint_id < len(self.visited_targets) - 1:
                            target_map_x, target_map_y = self.visited_targets[target_waypoint_id]['world_coords']
                            self.visited_targets.pop()  
                            # (f"Backtracking to waypoint {target_waypoint_id} at ({target_map_x}, {target_map_y})")
                            self.visited_targets = self.visited_targets[:target_waypoint_id+1]
                        else:
                            # logger.info("Invalid waypoint ID for backtrack, continuing with navigation")
                            decision['action'] = 'NAVIGATE'
                        panorama_got = True
                    if decision.get('action', 'NAVIGATE') == 'NAVIGATE':
                        navigate_or_not = True
                        direction = decision.get('direction', 'forward')
                        progress_analysis = decision.get('progress_analysis', '')
                        reasoning = decision.get('reasoning', '')
                        
                        
                        self.visited_targets[-1].update({
                            'progress_analysis': progress_analysis,
                            'reasoning': reasoning,
                            'direction_decision': direction
                        })
                        
                        
                        direction_map = {'forward': 0, 'left': 90, 'behind': 180, 'right': 270}
                        target_angle = direction_map.get(direction, 0)
                        frame_idx = target_angle // 90
                        
                        if frame_idx < len(panorama_frames):
                            dir_rgb = panorama_frames[frame_idx]['rgb']
                            self.visited_targets[-1]['dir_image'] = Image.fromarray(dir_rgb) if isinstance(dir_rgb, np.ndarray) else dir_rgb
                            self.visited_targets[-1]['turn_action'] = f"turn {direction}"
                        
                        
                        if direction == 'left':
                            action_list.extend([2] * 3)  
                        elif direction == 'right':
                            action_list.extend([3] * 3)  
                        elif direction == 'behind':
                            action_list.extend([2] * 6)  
                        
                        
                        panorama_got = True
                        # logger.info(f"Step 1 completed: Direction decision = {direction}, added turn actions")

                
                elif target_map_x is None and panorama_got and not action_list:
                    # logger.info("Step 2: Querying LLM for specific target...")
                    
                    
                    progress_analysis = self.visited_targets[-1].get('progress_analysis', '')
                    bbox = self.query_llm(obs[0], progress_analysis)
                    # logger.info(f"LLM response: {bbox}")
                    
                    
                    self.visited_targets[-1].update({
                        'description': bbox.get('target', 'unknown target'),
                        'bbox': bbox,
                        'llm_reasoning': bbox.get('reasoning', ''),
                        'llm_progress': bbox.get('progress', '')
                    })
                    
                    if bbox.get('action', 'NAVIGATE') == 'STOP':
                        # logger.info("LLM decided STOP - going last")
                        going_to_stop = True
                        

                    
                    depth_image = self._preprocess_depth(obs[0]['depth'], 0.1, 5.0) / 100.0
                    coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y2', 0)))
                    
                    
                    while True:
                        target = self.get_world_xz_from_pixel(
                            pixel_coords=coords,
                            depth_image=depth_image,
                            full_pose=current_pose,
                            camera_intrinsics=self._get_camera_intrinsics(),
                            agent_height=0.88
                        )
                        new_target_x = int(target[0] * 100.0 / self.resolution)
                        new_target_y = int(target[1] * 100.0 / self.resolution)
                        new_target_x = max(0, min(new_target_x, self.map_shape[0]-1))
                        new_target_y = max(0, min(new_target_y, self.map_shape[1]-1))
                        
                        if (self.traversible[new_target_y, new_target_x] == 1 or depth_image.max() < 0.1):
                            
                            
                            target_map_x, target_map_y = new_target_x, new_target_y
                            target_set_step = step
                            steps_since_target_set = 0
                            # logger.info(f"Target set at map coordinates: ({target_map_x}, {target_map_y}) at step {step}")
                            
                            
                            waypoint = np.array([target_map_y, target_map_x])
                            navigation_action = self.policy._get_action(
                                current_pose, waypoint, full_map[0], self.traversible, 
                                self.collision_map, step, self.current_episode_id, 
                                self.detected_classes, search_destination
                            )
                            action_list.append(navigation_action)
                            # logger.info(f"Added initial navigation action: {navigation_action}")
                            break
                        depth_image = depth_image - 0.1
                    
                    panorama_got = False  
                    # logger.info("Step 2 completed: Target acquired from LLM")
                
                
                elif target_map_x is not None and target_map_y is not None and not action_list:
                    # logger.info(f"Step 3: Continuing navigation to target ({target_map_x}, {target_map_y})")
                    waypoint = np.array([target_map_y, target_map_x])
                    navigation_action = self.policy._get_action(
                        current_pose, waypoint, full_map[0], self.traversible, 
                        self.collision_map, step, self.current_episode_id, 
                        self.detected_classes, search_destination
                    )
                    action_list.append(navigation_action)
                    # logger.info(f"Added navigation action: {navigation_action}")
            
            
            if not action_list:
                
                if target_map_x is None and panorama_got:
                    pass
                    # logger.info("Action list empty after panorama (no turn needed), will proceed to LLM query in next iteration")
                    
                
                elif target_map_x is not None and target_map_y is not None:
                    pass
                    # logger.info("Warning: Have target but no navigation action generated, this should not happen in normal flow")
                    # logger.info("This might indicate target reached or policy error, adding STOP action")
                    # action_list.append(0)  # STOP action
                
                else:
                    pass
                    # logger.info("Warning: Unexpected empty action_list state, adding STOP action as fallback")
                    # action_list.append(0)  # STOP action
            
            
            if action_list:
                # =================================================================
                
                
                # =================================================================
                self._action = action_list[0]
                action_list.pop(0)
                actions = [{"action": self._action}]

                # logger.info(f'Action actually performed: {self._action}')

                outputs = self.envs.step(actions)
                
                # =================================================================
                
                
                # =================================================================
                obs, _, dones, infos = [list(x) for x in zip(*outputs)]
                # logger.info('Sensor pose', obs[0]['sensor_pose'])

                if not dones[0]:
                    batch_obs = self._batch_obs(obs)
                    poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
                    self.mapping_module(batch_obs, poses)
                    full_map, full_pose, one_step_full_map =\
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
                            # logger.info(f"{self.current_episode_id}: {collided}\n")
                            fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR, 
                                                f"r{self.local_rank}_w{self.world_size}_collision_stuck.txt")
                            with open(fname, "a") as f:
                                f.writelines(f"id: {str(self.current_episode_id)}; step: {str(step)}; collided: {str(collided)}\n")
                        
                    current_action = self._action
                    if last_pose is not None and current_action is not None and current_action == 1:
                        collision_map = collision_check_fmm(last_pose, current_pose, self.resolution, 
                                                        self.mapping_module.map_shape)
                        self.collision_map = np.logical_or(self.collision_map, collision_map)
                    self.traversible[self.collision_map == 1] = 0
                else:
                    self._calculate_metric(infos)
                    return
            else:
                pass
                
                # logger.info("No action to execute this step, continuing to next iteration")
        self._calculate_metric(infos)
        self._flush_rgb_buffer()
        
        
        self._create_rgb_summary()
        
        
        current_stats = self.model.get_usage_stats()
        logger.info(f"Episode {self.current_episode_id} completed. Current token usage: "
                   f"Primary({current_stats['primary']['calls']} calls, {current_stats['primary']['total_tokens']} tokens), "
                   f"Secondary({current_stats['secondary']['calls']} calls, {current_stats['secondary']['total_tokens']} tokens)")

    def eval(self):
        self._set_eval_config()
        self._init_envs()
        self._collect_val_traj()
        self._initialize_policy()
        
        
        self.model.reset_stats()
        logger.info("Reset model usage statistics")
        
        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
            
        self.state_eps = {}
        t1 = time.time()
        for i in tqdm(range(eps_to_eval)):
            self.rollout()
            self.reset()
            
            
            if (i + 1) % 10 == 0:
                logger.info(f"=== Progress: {i + 1}/{eps_to_eval} episodes completed ===")
                self.model.print_usage_stats()
                    
        self.envs.close()
        
        
        logger.info("=== FINAL MODEL USAGE STATISTICS ===")
        final_stats = self.model.print_usage_stats()
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR, 
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)
        
        
        stats_fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR, 
                                  f"model_usage_stats_{split}_r{self.local_rank}_w{self.world_size}.json")
        with open(stats_fname, "w") as f:
            json.dump(final_stats, f, indent=2)
        logger.info(f"Model usage statistics saved to: {stats_fname}")
        
        # Save all RGB metadata at the end
        self._save_all_rgb_metadata()
        
        t2 = time.time()
        logger.info(f"time: {t2 - t1}s")
        logger.info("test time: %d", t2 - t1)

def merge_model_usage_stats(stats_dir, split="val_unseen"):
    import glob
    import json
    
    
    pattern = os.path.join(stats_dir, f"model_usage_stats_{split}_r*_w*.json")
    stat_files = glob.glob(pattern)
    
    if not stat_files:
        print(f"No model usage stat files found in {stats_dir} with pattern {pattern}")
        return
    
    
    merged_stats = {
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
        },
        'total_calls': 0,
        'total_tokens': 0,
        'num_processes': 0,
        'process_stats': []
    }
    
    
    for stat_file in stat_files:
        try:
            with open(stat_file, 'r') as f:
                stats = json.load(f)
            
            merged_stats['primary']['calls'] += stats['primary']['calls']
            merged_stats['primary']['input_tokens'] += stats['primary']['input_tokens']
            merged_stats['primary']['output_tokens'] += stats['primary']['output_tokens']
            merged_stats['primary']['total_tokens'] += stats['primary']['total_tokens']
            
            merged_stats['secondary']['calls'] += stats['secondary']['calls']
            merged_stats['secondary']['input_tokens'] += stats['secondary']['input_tokens']
            merged_stats['secondary']['output_tokens'] += stats['secondary']['output_tokens']
            merged_stats['secondary']['total_tokens'] += stats['secondary']['total_tokens']
            
            merged_stats['total_calls'] += stats['total_calls']
            merged_stats['total_tokens'] += stats['total_tokens']
            merged_stats['num_processes'] += 1
            
            
            process_info = {
                'file': os.path.basename(stat_file),
                'stats': stats
            }
            merged_stats['process_stats'].append(process_info)
            
            print(f"Loaded stats from: {stat_file}")
            
        except Exception as e:
            print(f"Error loading {stat_file}: {e}")
    
    
    merged_file = os.path.join(stats_dir, f"merged_model_usage_stats_{split}.json")
    with open(merged_file, 'w') as f:
        json.dump(merged_stats, f, indent=2)
    
    
    print("=== MERGED MODEL USAGE STATISTICS ===")
    print(f"Number of processes: {merged_stats['num_processes']}")
    print(f"Primary model:")
    print(f"  - Total calls: {merged_stats['primary']['calls']:,}")
    print(f"  - Total input tokens: {merged_stats['primary']['input_tokens']:,}")
    print(f"  - Total output tokens: {merged_stats['primary']['output_tokens']:,}")
    print(f"  - Total tokens: {merged_stats['primary']['total_tokens']:,}")
    print(f"Secondary model:")
    print(f"  - Total calls: {merged_stats['secondary']['calls']:,}")
    print(f"  - Total input tokens: {merged_stats['secondary']['input_tokens']:,}")
    print(f"  - Total output tokens: {merged_stats['secondary']['output_tokens']:,}")
    print(f"  - Total tokens: {merged_stats['secondary']['total_tokens']:,}")
    print(f"OVERALL TOTAL:")
    print(f"  - Total calls: {merged_stats['total_calls']:,}")
    print(f"  - Total tokens: {merged_stats['total_tokens']:,}")
    print(f"Merged statistics saved to: {merged_file}")
    print("=====================================")
    
    return merged_stats