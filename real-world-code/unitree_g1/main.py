#!/usr/bin/env python3
"""
Unitree G1 LaViRA entry point.

Usage
-----
Headless navigation tasks (VLN, EQA, Object Goal Nav):
    python main.py --task vln --instruction "go to the kitchen and find the red cup"
    python main.py --task eqa --instruction "what colour is the sofa?"
    python main.py --task object_nav --instruction "find the fire extinguisher"

Interactive web demo (voice + browser UI):
    python main.py --task interact

All heavy imports (Unitree SDK, Orbbec SDK, ROS, robot stack) are deferred
until after argparse so that ``python main.py --help`` works without any robot
dependencies installed.
"""
from __future__ import annotations

import os
import signal
import sys
import traceback
from typing import Optional

# ---------------------------------------------------------------------------
# Proxy-env unset block (preserve g1 source behaviour)
# Some async libraries raise errors when proxy env-vars are set to unsupported
# schemes (e.g. socks5).  Clear them unconditionally before any network I/O.
# ---------------------------------------------------------------------------
for _key in [
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
]:
    if _key in os.environ:
        del os.environ[_key]

# ---------------------------------------------------------------------------
# Module-global robot instance — used by the signal handler
# ---------------------------------------------------------------------------
_robot_instance = None  # type: Optional[object]


# ---------------------------------------------------------------------------
# Signal handler — best-effort stop, then force-exit
# ---------------------------------------------------------------------------
def _signal_handler(_sig, _frame):
    # type: (int, object) -> None
    """Handle SIGINT: best-effort robot stop, then os._exit(0)."""
    # Avoid circular import: print_info from utils may require cv2 / numpy
    # which might not be installed.  Use plain print as fallback.
    try:
        from utils import print_info
        print_info("Signal received, shutting down...")
    except Exception:
        print("[INFO] Signal received, shutting down...")

    if _robot_instance is not None:
        try:
            _robot_instance.stop_robot()
        except Exception:
            pass
        try:
            _robot_instance.shutdown()
        except Exception:
            pass

    os._exit(0)


# ---------------------------------------------------------------------------
# Interact / web-demo path
# ---------------------------------------------------------------------------
def run_demo(robot):
    # type: (object) -> None
    """Start the voice + web SocketIO demo (blocks until server exits)."""
    from config import Config
    from robot import IntegratedVisionNavController
    from web import app, socketio, set_controller, set_whisper_model

    controller = IntegratedVisionNavController(robot, socketio_instance=socketio)
    set_controller(controller)

    # Optional: load local faster-whisper model (skip gracefully if missing)
    if Config.LOCAL_WHISPER_MODEL_PATH and os.path.exists(
        Config.LOCAL_WHISPER_MODEL_PATH
    ):
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]

            try:
                from utils import print_info, print_success
            except Exception:
                print_info = print_success = print  # type: ignore[assignment]

            print_info("Loading local Whisper model on CPU...")
            whisper_model = WhisperModel(
                Config.LOCAL_WHISPER_MODEL_PATH,
                device="cpu",
                compute_type="int8",
            )
            set_whisper_model(whisper_model)
            print_success("Whisper model loaded.")
        except ImportError:
            try:
                from utils import print_warning
            except Exception:
                print_warning = print  # type: ignore[assignment]
            print_warning("faster_whisper not installed; voice commands disabled.")
        except Exception as exc:
            try:
                from utils import print_error
            except Exception:
                print_error = print  # type: ignore[assignment]
            print_error("Failed to load Whisper model: {}".format(exc))
    else:
        try:
            from utils import print_warning
        except Exception:
            print_warning = print  # type: ignore[assignment]
        print_warning(
            "Whisper model path '{}' not found; "
            "voice commands disabled. Text commands still work.".format(
                Config.LOCAL_WHISPER_MODEL_PATH
            )
        )

    # Determine SSL parameters — only use cert/key if both files exist
    ssl_cert = Config.SSL_CERT_PATH
    ssl_key = Config.SSL_KEY_PATH
    use_ssl = os.path.exists(ssl_cert) and os.path.exists(ssl_key)

    protocol = "https" if use_ssl else "http"
    try:
        from utils import print_info
    except Exception:
        print_info = print  # type: ignore[assignment]
    print_info(
        "Starting web server on {}://{}:{}".format(
            protocol, Config.SERVER_HOST, Config.SERVER_PORT
        )
    )

    socketio_kwargs = {
        "host": Config.SERVER_HOST,
        "port": Config.SERVER_PORT,
        "debug": False,
        "use_reloader": False,
    }
    if use_ssl:
        socketio_kwargs["certfile"] = ssl_cert
        socketio_kwargs["keyfile"] = ssl_key

    socketio.run(app, **socketio_kwargs)


# ---------------------------------------------------------------------------
# Headless navigation tasks (vln / eqa / object_nav)
# ---------------------------------------------------------------------------
def run_headless(robot, args):
    # type: (object, object) -> None
    """Run a single headless navigation task and return when complete."""
    from config import Config
    from tasks import TaskFactory

    try:
        from utils import print_info
    except Exception:
        print_info = print  # type: ignore[assignment]

    instruction = args.instruction or Config.DEFAULT_INSTRUCTION  # type: ignore[union-attr]

    task_cls = TaskFactory(args.task)  # type: ignore[union-attr]
    task = task_cls(robot, instruction=instruction)
    task.run()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    # type: () -> None
    global _robot_instance

    # Install SIGINT handler early — before any heavy import
    signal.signal(signal.SIGINT, _signal_handler)

    # --- 1. Parse arguments (lightweight — no robot stack imported yet) ------
    from config import config_manager

    args = config_manager.parse()

    # --- 2. Lazy-import the robot stack and construct the controller ---------
    try:
        from utils import print_error, print_info, print_success
    except Exception:
        print_error = print_info = print_success = print  # type: ignore[assignment]

    try:
        from robot import RobotController

        robot = RobotController()
        _robot_instance = robot
    except Exception as exc:
        print_error("Failed to initialize robot: {}".format(exc))
        traceback.print_exc()
        return

    # --- 3. G1-specific: bring robot to high-stand posture before navigation -
    print_info("Preparing G1 humanoid for navigation...")
    try:
        robot.high_stand()
        print_success("G1 is standing and ready for navigation")
    except Exception as exc:
        print_info("Stand-up sequence note: {}".format(exc))

    # --- 4. Dispatch to the requested task -----------------------------------
    try:
        if args.task == "interact":
            print_info("Starting interactive web demo...")
            run_demo(robot)
        else:
            print_info("Starting headless task: {}".format(args.task))
            run_headless(robot, args)

    except KeyboardInterrupt:
        print_info("Interrupted by user.")

    except Exception as exc:
        print_error("Runtime error: {}".format(exc))
        traceback.print_exc()

    finally:
        # Best-effort cleanup regardless of how we got here
        if _robot_instance is not None:
            print_info("Stopping robot movement...")
            try:
                _robot_instance.stop_robot()
            except Exception:
                pass

            print_info("Shutting down robot controller...")
            try:
                _robot_instance.shutdown()
            except Exception:
                pass

        print_info("Exiting.")
        os._exit(0)


if __name__ == "__main__":
    main()
