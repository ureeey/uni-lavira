"""
Global configuration for the Unitree G1 LaViRA deployment.

Every environment-specific field is overridable via environment variables so
the repository ships zero absolute paths and zero embedded secrets.

Model naming convention
-----------------------
VA (Vision-Action)  – primary model: handles visual perception + tactical
                      bounding-box detection (was "primary" / Qwen2.5-VL in
                      the G1 dev source).
LA (Language-Action) – secondary model: handles language reasoning + strategic
                       navigation planning (was "secondary" / Gemini in the G1
                       dev source).

Both default to the same local llama.cpp (llama-server) endpoint so a single
quantised model can serve both roles during development.
"""

import argparse
import os
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))


class Config:
    # =========================================================================
    # Vision-Action (VA) model — primary, tactical bbox detection
    # =========================================================================
    VA_API_KEY    = os.environ.get("VA_API_KEY",    "")
    VA_BASE_URL   = os.environ.get("VA_BASE_URL",   "http://localhost:8000/v1")
    VA_MODEL_NAME = os.environ.get("VA_MODEL_NAME", "Qwen3.5-27B-Q4_K_M")

    # =========================================================================
    # Language-Action (LA) model — secondary, strategic navigation decisions
    # =========================================================================
    LA_API_KEY    = os.environ.get("LA_API_KEY",    "")
    LA_BASE_URL   = os.environ.get("LA_BASE_URL",   "http://localhost:8000/v1")
    LA_MODEL_NAME = os.environ.get("LA_MODEL_NAME", "Qwen3.5-27B-Q4_K_M")

    # =========================================================================
    # Paths
    # =========================================================================
    OUTPUT_DIR       = os.environ.get("COBOT_OUTPUT_DIR", "outputs")
    PANORAMA_DIR     = "panorama_images"        # relative; rewritten in parse()
    CURRENT_VIEW_IMG = "current_view/current.png"

    # Session paths (populated in parse())
    SESSION_DIR  = ""
    LOG_DIR      = ""
    IMG_DIR      = ""
    IPLANNER_DIR = ""
    BBOX_DIR     = ""

    # =========================================================================
    # Robot / iPlanner
    # =========================================================================
    IPLANNER_URL = os.environ.get("IPLANNER_URL", "http://localhost:8888")

    # =========================================================================
    # G1 Humanoid Robot Parameters
    # =========================================================================
    # G1 stands ~1.32 m tall.  Four Orbbec Gemini 336L RGB-D cameras
    # (front/left/right/rear) provide the panorama; the front camera is
    # head-mounted.  Estimated camera height from ground when standing: ~1.2 m.
    # NOTE: source config.py uses CAMERA_HEIGHT = 1.0 m.  The G1 README
    # documents ~1.2 m.  This mismatch is tracked for later calibration; the
    # source value (1.0) is preserved here as the runtime default.
    CAMERA_HEIGHT         = float(os.environ.get("CAMERA_HEIGHT",         "1.0"))
    CAMERA_PITCH_ANGLE    = float(os.environ.get("CAMERA_PITCH_ANGLE",    "0.0"))
    CAMERA_ROLL_CORRECTION = float(os.environ.get("CAMERA_ROLL_CORRECTION", "0.0"))

    # Yaw bias compensation (rad/s).
    # Negative value forces a right-turn bias to correct left drift.
    YAW_BIAS_COMPENSATION = float(os.environ.get("YAW_BIAS_COMPENSATION", "-0.0"))

    # =========================================================================
    # G1 Motion Parameters
    # =========================================================================
    # G1 humanoid walking speed limits (conservative for safety)
    MAX_FORWARD_SPEED  = float(os.environ.get("MAX_FORWARD_SPEED",  "0.4"))   # m/s
    MAX_LATERAL_SPEED  = float(os.environ.get("MAX_LATERAL_SPEED",  "0.2"))   # m/s
    MAX_YAW_SPEED      = float(os.environ.get("MAX_YAW_SPEED",      "0.5"))   # rad/s

    DEFAULT_WALK_SPEED = float(os.environ.get("DEFAULT_WALK_SPEED", "0.3"))   # m/s
    ROTATION_SPEED     = float(os.environ.get("ROTATION_SPEED",     "0.4"))   # rad/s

    # Navigation tolerances
    GOAL_TOLERANCE  = float(os.environ.get("GOAL_TOLERANCE",  "1.0"))   # meters
    SAFE_DISTANCE   = float(os.environ.get("SAFE_DISTANCE",   "0.5"))   # meters
    REPLAN_INTERVAL = float(os.environ.get("REPLAN_INTERVAL", "0.1"))   # seconds

    # G1 network interface (wired connection to development PC).
    # Accepts either NETWORK_INTERFACE or UNITREE_NET_INTERFACE (the latter
    # mirrors the Unitree SDK naming convention); NETWORK_INTERFACE wins when
    # both are set.
    NETWORK_INTERFACE = os.environ.get(
        "NETWORK_INTERFACE",
        os.environ.get("UNITREE_NET_INTERFACE", "eth0"),
    )

    # =========================================================================
    # Orbbec Gemini 336L camera serials (4-view panorama)
    # =========================================================================
    # The three SDK-streamed cameras (front / left / right) are addressed by
    # USB serial number; the rear camera is a USB 2.0 device addressed by
    # V4L2 serial + a /dev/video* fallback path.  Every serial is overridable
    # via environment variables so the repository ships no device literals.
    ORBBEC_FRONT_SERIAL = os.environ.get("ORBBEC_FRONT_SERIAL", "")
    ORBBEC_LEFT_SERIAL  = os.environ.get("ORBBEC_LEFT_SERIAL",  "")
    ORBBEC_RIGHT_SERIAL = os.environ.get("ORBBEC_RIGHT_SERIAL", "")
    ORBBEC_REAR_SERIAL  = os.environ.get("ORBBEC_REAR_SERIAL",  "")
    # Fallback /dev/video* path used when the rear camera cannot be resolved
    # by serial (USB 2.0 device discovery is unreliable).
    ORBBEC_REAR_DEV_FALLBACK = os.environ.get(
        "ORBBEC_REAR_DEV_FALLBACK", "/dev/video30"
    )

    @classmethod
    def serial_to_position(cls):
        """Build the SDK serial -> camera-position map from configured serials.

        Only non-empty serials are included.  Returns a dict mapping each
        configured serial number to its logical position ('front' / 'left' /
        'right'); the rear camera is handled separately via V4L2.
        """
        mapping = {}
        for serial, position in (
            (cls.ORBBEC_FRONT_SERIAL, "front"),
            (cls.ORBBEC_LEFT_SERIAL, "left"),
            (cls.ORBBEC_RIGHT_SERIAL, "right"),
        ):
            if serial:
                mapping[serial] = position
        return mapping

    # Frame rate for saving front-camera RGB images
    SAVE_RGB_FPS = int(os.environ.get("SAVE_RGB_FPS", "10"))

    # =========================================================================
    # IMU / LiDAR Parameters
    # =========================================================================
    USE_IMU_FOR_ROTATION = os.environ.get("USE_IMU_FOR_ROTATION", "false").lower() == "true"
    USE_LIDAR            = os.environ.get("USE_LIDAR",            "false").lower() == "true"

    # =========================================================================
    # Number of camera directions (G1 = 4-view: front/back/left/right)
    # =========================================================================
    NUM_DIRECTIONS = int(os.environ.get("NUM_DIRECTIONS", "4"))

    # =========================================================================
    # Visual history buffer size (g1 source used the last 5 replan frames)
    # =========================================================================
    VISUAL_HISTORY_SIZE = int(os.environ.get("VISUAL_HISTORY_SIZE", "5"))

    # =========================================================================
    # Web demo
    # =========================================================================
    SERVER_HOST      = os.environ.get("COBOT_HTTP_HOST",  "0.0.0.0")
    SERVER_PORT      = int(os.environ.get("COBOT_HTTP_PORT", "5000"))
    SSL_CERT_PATH    = os.environ.get("COBOT_SSL_CERT",   "cert.pem")
    SSL_KEY_PATH     = os.environ.get("COBOT_SSL_KEY",    "key.pem")
    FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "change-me")

    # =========================================================================
    # Speech recognition (Whisper)
    # =========================================================================
    LOCAL_WHISPER_MODEL_PATH = os.environ.get(
        "COBOT_WHISPER_MODEL",
        os.path.join(_HERE, "models", "faster-whisper-base"),
    )

    # =========================================================================
    # Default instruction (text mode)
    # =========================================================================
    DEFAULT_INSTRUCTION = os.environ.get(
        "COBOT_DEFAULT_INSTRUCTION", "find a blue chair"
    )

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="LaViRA Navigation System (G1 Humanoid + 4x Orbbec Gemini 336L)"
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
            "--network_interface",
            type=str,
            default=None,
            help="Network interface for G1 communication (e.g., eth0)",
        )

        self.parser.add_argument(
            "--camera_height",
            type=float,
            default=None,
            help="Camera height from ground in meters",
        )

        self.parser.add_argument(
            "--iplanner_url",
            type=str,
            default=None,
            help="URL for iPlanner server",
        )

        self.parser.add_argument(
            "--max_steps",
            type=int,
            default=20,
            help="Maximum navigation steps before stopping",
        )

        self.parser.add_argument(
            "--no_lidar",
            action="store_true",
            help="Disable LiDAR odometry and use dead reckoning instead",
        )

        self.parser.add_argument(
            "--save_rgb_fps",
            type=int,
            default=None,
            help="Frame rate for saving front-camera RGB images",
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
            help="Use remote LLM endpoint instead of localhost",
        )

    def parse(self):
        args = self.parser.parse_args()

        # Apply inference-mode settings.
        # Model names are NEVER overwritten here; they always come from env.
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

        # CLI overrides (only applied when the caller explicitly passed a value)
        if args.output_dir:
            Config.OUTPUT_DIR = args.output_dir

        if args.api_key:
            Config.VA_API_KEY = args.api_key

        if args.network_interface:
            Config.NETWORK_INTERFACE = args.network_interface

        if args.camera_height is not None:
            Config.CAMERA_HEIGHT = args.camera_height

        if args.iplanner_url:
            Config.IPLANNER_URL = args.iplanner_url

        if args.no_lidar:
            Config.USE_LIDAR = False

        if args.save_rgb_fps is not None:
            Config.SAVE_RGB_FPS = args.save_rgb_fps

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
        print(f"[Config] Robot: Unitree G1 Humanoid")
        print(f"[Config] Camera Height: {Config.CAMERA_HEIGHT}m")
        print(f"[Config] Network Interface: {Config.NETWORK_INTERFACE}")

        return args


# Global configuration instance
config_manager = Config()
