#!/usr/bin/env python3
"""
Unitree Go1 LaViRA entry point.

Usage
-----
Headless navigation tasks (VLN, EQA, Object Goal Nav):
    python main.py --task vln --instruction "go to the chair in front of you"
    python main.py --task eqa --instruction "what colour is the sofa?"
    python main.py --task object_nav --instruction "find the cup"

Interactive web demo (voice + browser UI):
    python main.py --task interact

All heavy imports (ROS, Unitree SDK, robot stack) are deferred until after
argparse so that ``python main.py --help`` works without any robot dependencies
installed.
"""
from __future__ import annotations

import os
import signal
import sys
import time
import traceback
from typing import Optional

# ---------------------------------------------------------------------------
# Module-global robot instance — used by the signal handler
# ---------------------------------------------------------------------------
_robot_instance = None  # type: Optional[object]


# ---------------------------------------------------------------------------
# Signal handler — best-effort stop, then force-exit
# ---------------------------------------------------------------------------
def _signal_handler(_sig: int, _frame: object) -> None:
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
    os._exit(0)


# ---------------------------------------------------------------------------
# Interact / web-demo path
# ---------------------------------------------------------------------------
def run_demo(robot: object) -> None:
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
            print_error(f"Failed to load Whisper model: {exc}")
    else:
        try:
            from utils import print_warning
        except Exception:
            print_warning = print  # type: ignore[assignment]
        print_warning(
            f"Whisper model path '{Config.LOCAL_WHISPER_MODEL_PATH}' not found; "
            "voice commands disabled. Text commands still work."
        )

    # Determine SSL parameters
    ssl_cert = Config.SSL_CERT_PATH
    ssl_key = Config.SSL_KEY_PATH
    use_ssl = os.path.exists(ssl_cert) and os.path.exists(ssl_key)

    protocol = "https" if use_ssl else "http"
    try:
        from utils import print_info
    except Exception:
        print_info = print  # type: ignore[assignment]
    print_info(
        f"Starting web server on {protocol}://{Config.SERVER_HOST}:{Config.SERVER_PORT}"
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
def run_headless(robot: object, args: object) -> None:
    """Run a single headless navigation task and return when complete."""
    from config import Config
    from tasks import TaskFactory

    try:
        from utils import print_info, print_warning
    except Exception:
        print_info = print_warning = print  # type: ignore[assignment]

    instruction = args.instruction or Config.DEFAULT_INSTRUCTION  # type: ignore[union-attr]

    # Warn when the iPlanner client is not ready (navigation may degrade)
    planner_client = getattr(robot, "planner_client", None)
    if planner_client is not None and not getattr(
        planner_client, "initialized", True
    ):
        print_warning("iPlanner client not initialized; navigation may fail.")

    task_cls = TaskFactory(args.task)  # type: ignore[union-attr]
    task = task_cls(robot, instruction=instruction)
    task.run()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    global _robot_instance

    # Install SIGINT handler early — before any heavy import
    signal.signal(signal.SIGINT, _signal_handler)

    # --- 1. Parse arguments (lightweight — no robot stack imported yet) ------
    from config import config_manager

    args = config_manager.parse()

    # --- 2. Lazy-import the robot stack and construct the controller ---------
    try:
        from utils import print_error, print_info
    except Exception:
        print_error = print_info = print  # type: ignore[assignment]

    try:
        from robot import RobotController

        robot = RobotController()
        _robot_instance = robot
    except Exception as exc:
        print_error(f"Failed to initialize robot: {exc}")
        traceback.print_exc()
        return

    # --- 3. Dispatch to the requested task -----------------------------------
    try:
        if args.task == "interact":
            print_info("Starting interactive web demo...")
            run_demo(robot)
        else:
            print_info(f"Starting headless task: {args.task}")
            run_headless(robot, args)

    except KeyboardInterrupt:
        print_info("Interrupted by user.")

    except Exception as exc:
        print_error(f"Runtime error: {exc}")
        traceback.print_exc()

    finally:
        # Best-effort cleanup regardless of how we got here
        if _robot_instance is not None:
            print_info("Stopping robot movement...")
            try:
                _robot_instance.stop_robot()
            except Exception:
                pass

            # Allow the saver to flush any buffered frames before stopping it
            time.sleep(1.0)

            print_info("Stopping image saver...")
            try:
                _robot_instance.stop_saver()
            except Exception:
                pass

            # Join the saver thread with a short timeout
            saver_thread = getattr(_robot_instance, "saver_thread", None)
            if saver_thread is not None and saver_thread.is_alive():
                saver_thread.join(timeout=2.0)

        print_info("Exiting.")
        os._exit(0)


if __name__ == "__main__":
    main()
