"""
Navigation GRPO Configuration
"""

# Navigation GRPO training configuration

# Data configuration
PICKLE_FOLDER = "path/to/VLN-Finetune/data/trajectory_data"  # Path to pickle files
MAX_TRAJECTORY_LENGTH = 100  # Maximum trajectory length to consider
IMAGE_SIZE = (224, 224)  # Image resize dimensions

# Model configuration
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
PROCESSOR_PATH = None  # Will use MODEL_PATH if None

# Training configuration
NUM_EPOCHS = 5
BATCH_SIZE = 1  # Currently only supports 1
LEARNING_RATE = 1e-5
NUM_REPEATS = 3  # Number of times to repeat each trajectory

# GRPO configuration
BETA = 0.1  # KL penalty coefficient
GAMMA = 0.99  # Discount factor
LAMBDA_GAE = 0.95  # GAE lambda
CLIP_RATIO = 0.2  # PPO clip ratio
ENTROPY_COEF = 0.01  # Entropy coefficient
VALUE_COEF = 0.5  # Value function coefficient
MAX_GRAD_NORM = 1.0  # Maximum gradient norm
PPO_EPOCHS = 4  # Number of PPO epochs per update

# Reward configuration
REWARD_TYPE = "navigation"  # Options: navigation, success, distance
REWARD_CONFIG = {
    "success": {
        "success_reward": 1.0,
        "failure_penalty": -0.1
    },
    "distance": {
        "max_reward": 1.0,
        "distance_threshold": 10.0
    }
}

# Device configuration
DEVICE = "cuda"

# Logging and saving
SAVE_EVERY = 1  # Save checkpoint every N epochs
LOG_EVERY = 10  # Log metrics every N steps
OUTPUT_DIR = "./output/navigation_grpo"

# Navigation specific configuration
NAVIGATION_CONFIG_PATH = "path/to/VLN-Finetune/vlnce_baselines/config/default.yaml"

# Generation configuration
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
}
