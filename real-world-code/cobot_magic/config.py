"""
Global configuration for the Cobot Magic LaViRA iPlanner deployment.

Every environment-specific field is overridable via environment variables so
the repository ships zero absolute paths and zero embedded secrets. The
original development script hardcoded machine-specific absolute paths; those
are now externalised through the variables below.

Model naming convention
-----------------------
VA (Vision-Action) – primary model: visual grounding + tactical bbox decisions.
LA (Language-Action) – secondary model: language reasoning + strategic planning.

Cobot Magic uses a single local llama.cpp (llama-server) endpoint, so VA and LA
default to the *same* quantised model. The original development script used
``Qwen3.5-27B-Q4_K_M`` @ ``http://localhost:8000/v1`` with api_key ``EMPTY``; the
OSS default is ``Qwen3.5-27B-Q4_K_M`` @ ``http://localhost:8000/v1``.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


class Config:
    # --- Vision-Action (VA) model -----------------------------------------
    # Primary model: visual grounding + tactical bbox / NAVIGATE / STOP calls.
    VA_API_KEY    = os.environ.get("VA_API_KEY",    "")
    VA_BASE_URL   = os.environ.get("VA_BASE_URL",   "http://localhost:8000/v1")
    VA_MODEL_NAME = os.environ.get("VA_MODEL_NAME", "Qwen3.5-27B-Q4_K_M")

    # --- Language-Action (LA) model ----------------------------------------
    # Secondary model: language reasoning + strategic planning. Cobot Magic is
    # single-endpoint, so LA defaults to the same model/endpoint as VA.
    LA_API_KEY    = os.environ.get("LA_API_KEY",    "")
    LA_BASE_URL   = os.environ.get("LA_BASE_URL",   "http://localhost:8000/v1")
    LA_MODEL_NAME = os.environ.get("LA_MODEL_NAME", "Qwen3.5-27B-Q4_K_M")

    # VLM request timeout (seconds).
    VLM_TIMEOUT = _env_float("VLM_TIMEOUT", 120.0)

    # --- iPlanner ----------------------------------------------------------
    # Replaces the original machine-specific absolute paths to the iPlanner
    # YAML config and .pth checkpoint with repo-relative defaults.
    IPLANNER_CONFIG_PATH = os.environ.get(
        "IPLANNER_CONFIG_PATH", "iplanner/configs/iplanner.yaml"
    )
    IPLANNER_CHECKPOINT = os.environ.get(
        "IPLANNER_CHECKPOINT", "iplanner/checkpoints/iplanner.pth"
    )
    IPLANNER_DEVICE = os.environ.get("IPLANNER_DEVICE", "cuda:0")

    # --- Output root -------------------------------------------------------
    # Replaces the original machine-specific save_root with a repo-relative one.
    OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")

    # --- Session paths (populated in parse()) ------------------------------
    SESSION_DIR: str = ""

    # --- Camera intrinsics used by iPlanner (navdp) ------------------------
    # 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    NAVDP_INTRINSICS: List[List[float]] = [
        [906.78, 0.0,    640.20],
        [0.0,    904.61, 350.64],
        [0.0,    0.0,    1.0],
    ]

    # --- Camera resolution -------------------------------------------------
    RGB_WIDTH    = _env_int("RGB_WIDTH",    1280)
    RGB_HEIGHT   = _env_int("RGB_HEIGHT",   720)
    DEPTH_WIDTH  = _env_int("DEPTH_WIDTH",  1280)
    DEPTH_HEIGHT = _env_int("DEPTH_HEIGHT", 720)

    # --- Depth intrinsics fallback -----------------------------------------
    # Used until /camera/.../camera_info populates the real values.
    DEPTH_FX = _env_float("DEPTH_FX", 600.0)
    DEPTH_CX = _env_float("DEPTH_CX", 640.0)
    DEPTH_CY = _env_float("DEPTH_CY", 360.0)

    # --- TrajectoryFollower (pure-pursuit) limits --------------------------
    FOLLOWER_DT     = _env_float("FOLLOWER_DT",    0.05)
    FOLLOWER_MAX_V  = _env_float("FOLLOWER_MAX_V", 0.15)
    FOLLOWER_MAX_W  = _env_float("FOLLOWER_MAX_W", 0.5)
    FOLLOWER_MIN_V  = _env_float("FOLLOWER_MIN_V", 0.1)
    FOLLOWER_MIN_W  = _env_float("FOLLOWER_MIN_W", 0.2)
    FOLLOWER_ALPHA  = _env_float("FOLLOWER_ALPHA", 0.3)
    FOLLOWER_LOOKAHEAD_DIST = _env_float("FOLLOWER_LOOKAHEAD_DIST", 3.0)

    # --- Navigation control ------------------------------------------------
    ARRIVAL_THRESHOLD = _env_float("ARRIVAL_THRESHOLD", 1.50)   # meters
    REPLAN_INTERVAL   = _env_float("REPLAN_INTERVAL",   0.5)    # seconds
    NAV_TIMEOUT       = _env_float("NAV_TIMEOUT",       250.0)  # seconds

    # --- Rotation ----------------------------------------------------------
    ROTATE_COMPENSATION = _env_float("ROTATE_COMPENSATION", 1.1)
    ROTATE_SPEED        = _env_float("ROTATE_SPEED",        30.0)  # deg/s

    # --- Arm panorama scan -------------------------------------------------
    # Sequence of left/right arm joint targets used to capture side views
    # during the panorama scan.
    TARGET_SEQUENCE = [
        {"left": [0.8, 0.7, 0.0, 0.0, -0.85, 0.0, 0.1],
         "right": [-0.8, 0.7, 0.0, 0.0, -0.85, 0.0, 0.1]},
        {"left": [1.6, 0.7, 0.0, 0.0, -0.85, 0.0, 0.1],
         "right": [-1.6, 0.7, 0.0, 0.0, -0.85, 0.0, 0.1]},
        {"left": [2.4, 0.7, 0.0, 0.0, -0.85, 0.0, 0.1],
         "right": [-2.4, 0.7, 0.0, 0.0, -0.85, 0.0, 0.1]},
    ]
    # Home / zero arm pose (7 joints).
    ZERO_POSITION = [0.0] * 7
    # Per-step arm interpolation increment. The original script multiplied the
    # base profile [0.01]*6 + [0.2] by a factor of 10.
    ARM_SPEED      = _env_float("ARM_SPEED", 10.0)
    ARM_BASE_STEPS = [0.01] * 6 + [0.2]
    ARM_PUBLISH_RATE = _env_int("ARM_PUBLISH_RATE", 40)

    # =========================================================================
    # Web / voice demo (used by the optional `--task interact` web interface)
    # =========================================================================
    SERVER_HOST = os.environ.get("COBOT_HTTP_HOST", "0.0.0.0")
    SERVER_PORT = _env_int("COBOT_HTTP_PORT", 5000)
    SSL_CERT_PATH = os.environ.get("COBOT_SSL_CERT", "cert.pem")
    SSL_KEY_PATH = os.environ.get("COBOT_SSL_KEY", "key.pem")
    FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "change-me")
    LOCAL_WHISPER_MODEL_PATH = os.environ.get(
        "COBOT_WHISPER_MODEL", "./models/faster-whisper-base"
    )
    DEFAULT_INSTRUCTION = os.environ.get("COBOT_DEFAULT_INSTRUCTION", "")

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description="Cobot Magic LaViRA iPlanner navigation"
        )
        self._setup_arguments()

    def _setup_arguments(self) -> None:
        self.parser.add_argument(
            "--task",
            type=str,
            default="object_nav",
            choices=["object_nav", "vln", "eqa", "interact"],
            help=(
                "Task type: object_nav (Object Goal Nav, default), "
                "vln (Vision Language Nav), eqa (Embodied QA), "
                "interact (Web Interaction)"
            ),
        )
        self.parser.add_argument(
            "--instruction",
            type=str,
            default=None,
            help=(
                "Text instruction for the target object or navigation goal "
                "(e.g. \"chair\", \"go to the door on the left\")."
            ),
        )
        self.parser.add_argument(
            "--max_cycles",
            type=int,
            default=10,
            help="Maximum number of navigation cycles.",
        )
        self.parser.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help="Override OUTPUT_ROOT for this run.",
        )

    def parse(self, argv: Optional[List[str]] = None) -> argparse.Namespace:
        args = self.parser.parse_args(argv)

        # CLI override only when the caller explicitly passed a value.
        # Model names always come from env defaults (never clobbered here).
        if args.output_dir:
            Config.OUTPUT_ROOT = args.output_dir

        # Create a timestamped session directory under OUTPUT_ROOT.
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        Config.SESSION_DIR = os.path.join(Config.OUTPUT_ROOT, timestamp)
        os.makedirs(Config.SESSION_DIR, exist_ok=True)

        print(f"[Config] Session Directory: {Config.SESSION_DIR}")
        return args


# Global configuration instance.
config_manager = Config()
