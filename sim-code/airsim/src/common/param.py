import argparse
import os
import datetime
from pathlib import Path
from utils.CN import CN
import transformers
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CommonArguments:
    project_prefix: str = field(
        default_factory=lambda: str(Path(str(os.getcwd())).parent.resolve()),
        metadata={"help": "project path"}
    )
    
    run_type: str = field(default="train", metadata={"help": "run_type in [collect, train, eval]"})
    collect_type: str = field(default="dagger", metadata={"help": "collect_type in [dagger]"})
    name: str = field(default='default', metadata={"help": 'experiment name'})
    
    maxInput: int = field(default=500, metadata={"help": "max input instruction"})
    maxWaypoints: int = field(default=500, metadata={"help": 'max action sequence'})
    
    dagger_it: int = field(default=1)
    epochs: int = field(default=10)
    lr: float = field(default=0.00025, metadata={"help": "learning rate"})
    batchSize: int = field(default=8)
    trainer_gpu_device: int = field(default=0, metadata={"help": 'GPU'})
    
    inflection_weight_coef: float = field(default=1.9)
    dagger_mode_load_scene: List[str] = field(default_factory=list)
    dagger_update_size: int = field(default=8000)
    dagger_mode: str = field(default="end", metadata={"help": 'dagger mode in [end middle nearest]'})
    dagger_p: float = field(default=1.0, metadata={"help": 'dagger p'})
    
    tokenizer_use_bert: bool = field(default=True)
    
    simulator_tool_port: int = field(default=30000, metadata={"help": "simulator_tool port"})
    DDP_MASTER_PORT: int = field(default=20001, metadata={"help": "DDP MASTER_PORT"})

    continue_start_from_dagger_it: Optional[int] = field(default=None)
    continue_start_from_checkpoint_path: Optional[str] = field(default=None)

    vlnbert: bool = field(default=False)
    featdropout: float = field(default=0.4)
    action_feature: int = field(default=32)
    
    eval_save_path: Optional[str] = field(default=None)
    dagger_save_path: Optional[str] = field(default=None)
    activate_maps: Optional[List[str]] = field(default_factory=list)

    gpu_id: int = field(default=3, metadata={"help": "simulator gpus"})
    always_help: bool = field(default=False)
    use_gt: bool = field(default=False)

    dataset_path: Optional[str] = field(default=None)
    eval_json_path: Optional[str] = field(default=None)
    train_json_path: Optional[str] = field(default=None)
    object_name_json_path: Optional[str] = field(default=None)
    map_spawn_area_json_path: Optional[str] = field(default=None)
    
@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_grid_pinpoints: Optional[str] = field(default=None)
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=True)
    mm_use_im_start_end: bool = field(default=False)

    
@dataclass

class ModelArguments:
    model_path: Optional[str] = field(default="facebook/opt-350m")
    model_base: Optional[str] = field(default=None)
    traj_model_path: Optional[str] = field(default=None)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    groundingdino_config: Optional[str] = field(default=None)
    groundingdino_model_path: Optional[str] = field(default=None)

    
    api_key: Optional[str] = field(default=None)
    api_base_url: Optional[str] = field(default=None)
    api_model_name: Optional[str] = field(default="gpt-4-vision-preview")

@dataclass
class MAP:
    
    GROUNDING_DINO_CONFIG_PATH: str = field(default="data/grounded_sam/GroundingDINO_SwinT_OGC.py")
    GROUNDING_DINO_CHECKPOINT_PATH: str = field(default="data/grounded_sam/groundingdino_swint_ogc.pth")
    SAM_CHECKPOINT_PATH: str = field(default="data/grounded_sam/sam_vit_h_4b8939.pth")
    RepViTSAM_CHECKPOINT_PATH: str = field(default="data/grounded_sam/repvit_sam.pt")
    SAM_ENCODER_VERSION: str = field(default="vit_h")
    REPVITSAM: int = field(default=1)

    
    BOX_THRESHOLD: float = field(default=0.25)
    TEXT_THRESHOLD: float = field(default=0.25)

    
    FRAME_WIDTH: int = field(default=160)
    FRAME_HEIGHT: int = field(default=120)
    MAP_RESOLUTION: int = field(default=5)
    MAP_SIZE_CM: int = field(default=2400)
    GLOBAL_DOWNSCALING: int = field(default=2)

    
    VISION_RANGE: int = field(default=100)
    HFOV: int = field(default=79)
    AGENT_HEIGHT: float = field(default=0.88)

    
    DU_SCALE: int = field(default=1)
    CAT_PRED_THRESHOLD: float = field(default=5.0)
    EXP_PRED_THRESHOLD: float = field(default=1.0)
    MAP_PRED_THRESHOLD: float = field(default=1.0)
    MAX_SEM_CATEGORIES: int = field(default=16)
    CENTER_RESET_STEPS: int = field(default=25)
    MIN_Z: int = field(default=2)
    DEVICE: str = field(default="cuda:0")  
    NUM_ENVIRONMENTS: int = field(default=1)  

    
    VISUALIZE: bool = field(default=True)
    PRINT_IMAGES: bool = field(default=False)

    
    RESULTS_DIR: str = field(default="data/logs/eval_results/")

@dataclass
class EVAL:
    USE_CKPT_CONFIG: bool = field(default=False)
    SAVE_RESULTS: bool = field(default=True)

    
    MIN_CONSTRAINT_STEPS: int = field(default=10)
    MAX_CONSTRAINT_STEPS: int = field(default=25)
    SCORE_THRESHOLD: float = field(default=0.5)
    VALUE_THRESHOLD: float = field(default=0.30)
    DECISION_THRESHOLD: float = field(default=0.4)
    FMM_WAYPOINT_THRESHOLD: float = field(default=2.0)
    FMM_GOAL_THRESHOLD: float = field(default=1.0)
    CHANGE_THRESHOLD: float = field(default=-0.03)

@dataclass
class AGENT_0:
    HEIGHT: float = field(default=0.88)
    RADIUS: float = field(default=0.1)
    SENSORS: List[str] = field(default_factory=lambda: ['RGB_SENSOR', 'DEPTH_SENSOR'])

@dataclass
class RGB_SENSOR:
    WIDTH: int = field(default=640)
    HEIGHT: int = field(default=480)
    HFOV: int = field(default=79)
    POSITION: List[float] = field(default_factory=lambda: [0, 0.88, 0])

@dataclass
class DEPTH_SENSOR:
    WIDTH: int = field(default=640)
    HEIGHT: int = field(default=480)
    HFOV: int = field(default=79)
    MIN_DEPTH: float = field(default=0.1)
    MAX_DEPTH: float = field(default=5.0)
    POSITION: List[float] = field(default_factory=lambda: [0, 0.88, 0])

@dataclass
class SIMULATOR:
    AGENT_0: AGENT_0 = field(default_factory=AGENT_0)
    RGB_SENSOR: RGB_SENSOR = field(default_factory=RGB_SENSOR)
    DEPTH_SENSOR: DEPTH_SENSOR = field(default_factory=DEPTH_SENSOR)

@dataclass
class ENVIRONMENT:
    MAX_EPISODE_STEPS: int = field(default=300)

@dataclass
class TASK_CONFIG:
    SEED: int = field(default=0)
    ENVIRONMENT: ENVIRONMENT = field(default_factory=ENVIRONMENT)
    SIMULATOR: SIMULATOR = field(default_factory=SIMULATOR)

@dataclass
class HabitatConfig:
    # ============================================================================
    
    # ============================================================================
    ENV_NAME: str = field(default="VLNCEZeroShotEnv")

    # ----------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------
    TORCH_GPU_ID: int = field(default=0)
    TORCH_GPU_IDS: List[int] = field(default_factory=lambda: [0])
    SIMULATOR_GPU_IDS: List[int] = field(default_factory=lambda: [0])
    GPU_NUMBERS: int = field(default=1)

    # ----------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------
    TENSORBOARD_DIR: str = field(default="data/tensorboard_dirs/")
    CHECKPOINT_FOLDER: str = field(default="data/checkpoints/")
    EVAL_CKPT_PATH_DIR: str = field(default="data/checkpoints/")
    RESULTS_DIR: str = field(default="data/logs/eval_results/")
    VIDEO_DIR: str = field(default="data/logs/video/")
    LOG_FILE: str = field(default="test_ZS-Evaluator-mp.log")
    RGB_SAVE_DIR: str = field(default="./saved_rgb_images")
    SAVE_RGB: bool = field(default=True)

    # ============================================================================
    
    # ============================================================================
    MAP: MAP = field(default_factory=MAP)

    # ============================================================================
    
    # ============================================================================
    EVAL: EVAL = field(default_factory=EVAL)

    # ============================================================================
    
    # ============================================================================
    DETECTION_CONFIDENCE_THRESHOLD: float = field(default=0.0)
    TARGET_REACHED_THRESHOLD: float = field(default=15.0)
    TARGET_STOP_THRESHOLD: float = field(default=3.0)
    TARGET_CLOSE_THRESHOLD: float = field(default=8.0)

    # ============================================================================
    
    # ============================================================================
    TASK_CONFIG: TASK_CONFIG = field(default_factory=TASK_CONFIG)

    # ============================================================================
    
    # ============================================================================
    MAX_STEPS_TO_TARGET: int = field(default=30)
    MAX_NEW_TOKENS: int = field(default=1024)
    TEMPERATURE: int = field(default=0)


parser = transformers.HfArgumentParser((CommonArguments, ModelArguments, DataArguments))
args, model_args, data_args = parser.parse_args_into_dataclasses()


data_habitat = HabitatConfig()


data_habitat.MAP.NUM_ENVIRONMENTS = args.batchSize

args.make_dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
args.logger_file_name = '{}/workdir/{}/logs/{}_{}.log'.format(args.project_prefix, args.run_type, args.collect_type, args.make_dir_time)


# args.run_type = 'collect'
assert args.run_type in ['collect', 'train', 'eval'], 'run_type error'
# args.collect_type = 'TF'
assert args.collect_type in ['TF', 'dagger'], 'collect_type error'


args.machines_info = [
    {
        'MACHINE_IP': '127.0.0.1',
        'SOCKET_PORT': int(args.simulator_tool_port),
        'MAX_SCENE_NUM': 16,
        'open_scenes': [],
    },
]


args.TRAIN_VOCAB = Path(args.project_prefix) / 'DATA/data/aerialvln/train_vocab.txt'
args.TRAINVAL_VOCAB = Path(args.project_prefix) / 'DATA/data/aerialvln/train_vocab.txt'
args.vocab_size = 10038


default_config = CN.clone()
default_config.make_dir_time = args.make_dir_time
default_config.freeze()

