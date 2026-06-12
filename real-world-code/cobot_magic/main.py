#!/usr/bin/env python3
"""
Cobot Magic LaViRA iPlanner entry point.

Usage
-----
Headless navigation tasks (Object Goal Nav, VLN, EQA):
    python main.py --task object_nav --instruction "chair"
    python main.py --task object_nav --instruction "find the cup" --max_cycles 15
    python main.py --task vln --instruction "go to the chair in front of you"
    python main.py --task eqa --instruction "what colour is the sofa?"

Interactive web demo (voice + browser UI):
    python main.py --task interact

All heavy imports (ROS, robot stack, torch) are deferred until after argparse so
that ``python main.py --help`` works without any robot dependencies installed.
"""
from __future__ import annotations

import os
import signal
import sys
import traceback
from typing import Optional

# ---------------------------------------------------------------------------
# Module-global robot instance — referenced by the signal handler
# ---------------------------------------------------------------------------
_robot_instance = None  # type: Optional[object]


# ---------------------------------------------------------------------------
# Signal handler — best-effort stop, then force-exit
# ---------------------------------------------------------------------------
def _signal_handler(_sig, _frame):
    # type: (int, object) -> None
    """Handle SIGINT: best-effort robot shutdown, then os._exit(0)."""
    try:
        from utils import print_info
        print_info("Signal received, shutting down...")
    except Exception:
        print("[INFO] Signal received, shutting down...")

    if _robot_instance is not None:
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
    """Start the voice + web SocketIO demo (blocks until the server exits)."""
    from config import Config
    from robot import IntegratedVisionNavController
    from web import app, socketio, set_controller, set_whisper_model

    controller = IntegratedVisionNavController(robot, socketio_instance=socketio)
    set_controller(controller)

    # Optional: load a local faster-whisper model; skip gracefully if unavailable.
    whisper_path = Config.LOCAL_WHISPER_MODEL_PATH
    if whisper_path and os.path.exists(whisper_path):
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]

            try:
                from utils import print_info, print_success
            except Exception:
                print_info = print_success = print  # type: ignore[assignment]

            print_info("Loading local Whisper model on CPU...")
            whisper_model = WhisperModel(
                whisper_path,
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
            print_warning(
                "faster_whisper not installed; voice commands disabled. "
                "Text commands still work."
            )
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
            "Whisper model path '{}' not found; voice commands disabled. "
            "Text commands still work.".format(whisper_path)
        )

    # Determine SSL parameters — only use if both files exist.
    ssl_cert = Config.SSL_CERT_PATH
    ssl_key = Config.SSL_KEY_PATH
    use_ssl = bool(ssl_cert and ssl_key
                   and os.path.exists(ssl_cert) and os.path.exists(ssl_key))

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
# Object-goal navigation (text instruction)
# ---------------------------------------------------------------------------
def run_object_nav(robot, args):
    # type: (object, object) -> None
    """Run a single ObjectNav task and return when complete.

    Passes --instruction straight through; max_cycles controls the cycle limit.
    """
    from tasks import TaskFactory

    try:
        from utils import print_info
    except Exception:
        print_info = print  # type: ignore[assignment]

    task_cls = TaskFactory("object_nav")
    print_info(
        "Starting object_nav: instruction={!r}, max_cycles={}".format(
            args.instruction,  # type: ignore[union-attr]
            args.max_cycles,  # type: ignore[union-attr]
        )
    )
    task = task_cls(
        robot,
        instruction=args.instruction,  # type: ignore[union-attr]
        max_cycles=args.max_cycles,  # type: ignore[union-attr]
    )
    task.run()


# ---------------------------------------------------------------------------
# Headless VLN / EQA tasks
# ---------------------------------------------------------------------------
def run_headless(robot, args):
    # type: (object, object) -> None
    """Run a single VLN or EQA task and return when complete."""
    from config import Config
    from tasks import TaskFactory

    try:
        from utils import print_info
    except Exception:
        print_info = print  # type: ignore[assignment]

    instruction = args.instruction or Config.DEFAULT_INSTRUCTION  # type: ignore[union-attr]
    if not instruction:
        instruction = "navigate forward"

    task_name = args.task  # type: ignore[union-attr]
    print_info("Starting {}: instruction={!r}".format(task_name, instruction))

    task_cls = TaskFactory(task_name)
    task = task_cls(robot, instruction=instruction)
    task.run()


# ---------------------------------------------------------------------------
# Robot construction (lazy ROS import)
# ---------------------------------------------------------------------------
def _build_robot():
    # type: () -> object
    """Initialize the ROS node and construct RobotController.

    ``rospy.init_node`` is called here so that module-level ``import rospy``
    is never triggered at argparse time. ``RobotController.__init__`` does
    its own lazy ROS imports, but the node must exist first.
    """
    import rospy
    from robot import RobotController

    rospy.init_node("cobot_magic_lavira_iplanner")
    robot = RobotController()
    # initialize_models() is called inside RobotController.__init__ already
    # (mirroring the monolith); no separate call needed here.
    return robot


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    # type: () -> None
    global _robot_instance

    # Install SIGINT handler before any heavy import.
    signal.signal(signal.SIGINT, _signal_handler)

    # --- 1. Parse arguments (lightweight — no robot stack imported yet) ------
    from config import config_manager

    args = config_manager.parse()

    # Validate object_nav: --instruction must be provided and non-empty.
    if args.task == "object_nav":
        if not (args.instruction and args.instruction.strip()):
            print(
                "[FATAL] object_nav requires a non-empty --instruction "
                "(the object name, e.g. --instruction \"chair\").",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- 2. Lazy-import robot stack and construct the controller -------------
    try:
        from utils import print_error, print_info
    except Exception:
        print_error = print_info = print  # type: ignore[assignment]

    try:
        robot = _build_robot()
        _robot_instance = robot
    except Exception as exc:
        print_error("Failed to initialize robot: {}".format(exc))
        traceback.print_exc()
        return

    # --- 3. Dispatch to the requested task -----------------------------------
    try:
        if args.task == "interact":
            print_info("Starting interactive web demo...")
            run_demo(robot)

        elif args.task == "object_nav":
            run_object_nav(robot, args)

        else:
            # vln / eqa
            run_headless(robot, args)

    except KeyboardInterrupt:
        print_info("Interrupted by user.")

    except Exception as exc:
        print_error("Runtime error: {}".format(exc))
        traceback.print_exc()

    finally:
        # Best-effort cleanup regardless of how we arrived here.
        if _robot_instance is not None:
            print_info("Shutting down robot...")
            try:
                _robot_instance.shutdown()
            except Exception:
                pass

        print_info("Exiting.")
        os._exit(0)


if __name__ == "__main__":
    main()
