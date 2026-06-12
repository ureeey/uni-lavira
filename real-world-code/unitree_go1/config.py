"""
Global configuration for the Unitree Go1 LaViRA deployment.

Every environment-specific field is overridable via environment variables so
the repository ships zero absolute paths and zero embedded secrets.

Model naming convention
-----------------------
VA (Vision-Action) – primary model: handles visual perception + action decisions.
LA (Language-Action) – secondary model: handles language reasoning + planning.

Both default to the same local llama.cpp (llama-server) endpoint so a single
quantised model can serve both roles during development.
"""
import argparse
import os
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))


class Config:
    # --- Vision-Action (VA) model -----------------------------------------
    # Primary model: visual grounding + tactical decisions.
    VA_API_KEY   = os.environ.get("VA_API_KEY",   "")
    VA_BASE_URL  = os.environ.get("VA_BASE_URL",  "http://localhost:8000/v1")
    VA_MODEL_NAME = os.environ.get("VA_MODEL_NAME", "Qwen3.5-27B-Q4_K_M")

    # --- Language-Action (LA) model ----------------------------------------
    # Secondary model: language reasoning + strategic planning.
    LA_API_KEY   = os.environ.get("LA_API_KEY",   "")
    LA_BASE_URL  = os.environ.get("LA_BASE_URL",  "http://localhost:8000/v1")
    LA_MODEL_NAME = os.environ.get("LA_MODEL_NAME", "Qwen3.5-27B-Q4_K_M")

    # --- Output directories ------------------------------------------------
    OUTPUT_DIR  = os.environ.get("COBOT_OUTPUT_DIR", "outputs")
    PANORAMA_DIR = "panorama_images"       # relative; rewritten in parse()
    CURRENT_VIEW_IMG = "current_view/current.png"

    # --- Session paths (populated in parse()) ------------------------------
    SESSION_DIR  = ""
    LOG_DIR      = ""
    IMG_DIR      = ""
    IPLANNER_DIR = ""
    BBOX_DIR     = ""

    # --- Robot / iPlanner --------------------------------------------------
    IPLANNER_URL = os.environ.get("IPLANNER_URL", "http://localhost:8888")

    # --- Camera extrinsics (calibration) -----------------------------------
    CAMERA_HEIGHT          = float(os.environ.get("CAMERA_HEIGHT", "0.3"))   # meters
    CAMERA_ROLL_CORRECTION = float(os.environ.get("CAMERA_ROLL_CORRECTION", "0.0"))

    # --- Visual history buffer size ----------------------------------------
    VISUAL_HISTORY_SIZE = int(os.environ.get("VISUAL_HISTORY_SIZE", "10"))

    # --- Number of camera directions (go1 ships 4-view) --------------------
    NUM_DIRECTIONS = int(os.environ.get("NUM_DIRECTIONS", "4"))

    # --- Unitree SDK -------------------------------------------------------
    # Set UNITREE_SDK_PATH to the directory containing robot_interface.so.
    # Leave empty to rely on whatever is already on sys.path / PYTHONPATH.
    UNITREE_SDK_PATH    = os.environ.get("UNITREE_SDK_PATH", "")
    UNITREE_HOST        = os.environ.get("UNITREE_HOST", "192.168.123.161")
    UNITREE_LOCAL_PORT  = int(os.environ.get("UNITREE_LOCAL_PORT",  "8080"))
    UNITREE_REMOTE_PORT = int(os.environ.get("UNITREE_REMOTE_PORT", "8082"))

    # --- Web demo ----------------------------------------------------------
    SERVER_HOST        = os.environ.get("COBOT_HTTP_HOST",  "0.0.0.0")
    SERVER_PORT        = int(os.environ.get("COBOT_HTTP_PORT", "5000"))
    SSL_CERT_PATH      = os.environ.get("COBOT_SSL_CERT",   "cert.pem")
    SSL_KEY_PATH       = os.environ.get("COBOT_SSL_KEY",    "key.pem")
    FLASK_SECRET_KEY   = os.environ.get("FLASK_SECRET_KEY", "change-me")

    # --- Speech recognition (Whisper) --------------------------------------
    LOCAL_WHISPER_MODEL_PATH = os.environ.get(
        "COBOT_WHISPER_MODEL",
        os.path.join(_HERE, "models", "faster-whisper-base"),
    )

    # --- Default instruction (text mode) -----------------------------------
    DEFAULT_INSTRUCTION = os.environ.get(
        "COBOT_DEFAULT_INSTRUCTION", "find a blue robot dog"
    )

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Point Navigation System with 4 Cameras"
        )
        self._setup_arguments()

    def _setup_arguments(self):
        self.parser.add_argument(
            "--task",
            type=str,
            required=True,
            choices=["vln", "eqa", "object_nav", "interact"],
            help=(
                "Task type: vln (Vision Language Nav), eqa (Embodied QA), "
                "object_nav (Object Goal Nav), interact (Web Interaction)"
            ),
        )

        self.parser.add_argument(
            "--instruction",
            type=str,
            default=None,
            help="Text instruction for VLN or EQA tasks",
        )

        self.parser.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help="Directory to save outputs",
        )

        self.parser.add_argument(
            "--api_key",
            type=str,
            default=None,
            help="API key for the VA (primary) model (optional override)",
        )

        self.parser.add_argument(
            "--inference_mode",
            type=str,
            default="local",
            choices=["local", "api"],
            help=(
                "Inference mode: local (use local llama-server) "
                "or api (use remote API)"
            ),
        )

        self.parser.add_argument(
            "--use_remote_llm",
            action="store_true",
            help="Use remote LLM instead of localhost",
        )

    def parse(self):
        args = self.parser.parse_args()

        # Apply inference mode settings
        if args.inference_mode == "local":
            if args.use_remote_llm:
                remote = os.environ.get("REMOTE_LLM_BASE_URL", "http://localhost:8000/v1")
                Config.VA_BASE_URL = remote
                Config.LA_BASE_URL = remote
                print(f"[Config] Remote inference: {remote}")
            else:
                print("[Config] Switching to LOCAL Inference Mode (llama-server)")
                Config.VA_BASE_URL = "http://localhost:8000/v1"
                Config.LA_BASE_URL = "http://localhost:8000/v1"

            # Local servers typically do not require a real API key.
            if args.api_key is None:
                Config.VA_API_KEY = "sk-no-key-required"
                Config.LA_API_KEY = "sk-no-key-required"

        else:
            print("[Config] Using API Inference Mode")
            # In API mode the env-var defaults (VA_API_KEY / LA_API_KEY) are
            # used unless overridden by --api_key below.

        # CLI overrides (only when the caller explicitly passed a value)
        if args.output_dir:
            Config.OUTPUT_DIR = args.output_dir

        if args.api_key:
            Config.VA_API_KEY = args.api_key

        # Honour env-var for instruction when CLI did not provide one
        if args.instruction is None:
            args.instruction = Config.DEFAULT_INSTRUCTION

        # Create timestamped session directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Config.SESSION_DIR  = os.path.join(Config.OUTPUT_DIR, timestamp)
        Config.LOG_DIR      = os.path.join(Config.SESSION_DIR, "logs")
        Config.IMG_DIR      = os.path.join(Config.SESSION_DIR, "images")
        Config.PANORAMA_DIR = os.path.join(Config.IMG_DIR, "panorama")
        Config.IPLANNER_DIR = os.path.join(Config.IMG_DIR, "iplanner")
        Config.BBOX_DIR     = os.path.join(Config.IMG_DIR, "bbox")

        for d in (
            Config.SESSION_DIR,
            Config.LOG_DIR,
            Config.IMG_DIR,
            Config.PANORAMA_DIR,
            Config.IPLANNER_DIR,
            Config.BBOX_DIR,
        ):
            os.makedirs(d, exist_ok=True)

        print(f"[Config] Session Directory: {Config.SESSION_DIR}")
        return args


# Global configuration instance
config_manager = Config()
