"""
Thin integrated vision-navigation controller for the Cobot Magic web / demo.

This controller does NOT own the per-cycle navigation loop. That loop lives
inside the task objects (``tasks.ObjectNavTask.run`` and friends). The
controller's only jobs are:

- build / hold the active task,
- launch ``task.run()`` on a daemon thread for the demo,
- emit ``status_update`` / ``response`` events over SocketIO when present,
- proxy the live front-camera JPEG to the browser.

For headless / CLI mode, callers can build the task directly via
``tasks.TaskFactory`` and call ``task.run()`` without this controller.
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

from utils import print_error, print_info


class IntegratedVisionNavController:
    """Thin demo wrapper around a navigation task."""

    def __init__(self, robot, socketio_instance=None) -> None:
        self.robot = robot
        self.socketio = socketio_instance

        self.current_task = None
        self.task_thread: Optional[threading.Thread] = None
        self.is_task_running = False

    # ------------------------------------------------------------------ #
    # SocketIO helpers
    # ------------------------------------------------------------------ #
    def _emit_status(self, message: str) -> None:
        if self.socketio is not None:
            self.socketio.emit("status_update", {"message": message})
        print_info(f"[STATUS] {message}")

    def _emit_response(self, message: str) -> None:
        if self.socketio is not None:
            self.socketio.emit("response", {"message": message})
        print_info(f"[RESPONSE] {message}")

    # ------------------------------------------------------------------ #
    # Task management
    # ------------------------------------------------------------------ #
    def set_task(self, task) -> None:
        """Attach an already-built task instance to the controller."""
        self.current_task = task

    def start_new_task(self, instruction: str) -> Tuple[bool, str]:
        """Build / set a task and run ``task.run()`` on a daemon thread.

        Stops any running task first, then launches the new one in the
        background and emits ``status_update`` / ``response`` events over
        SocketIO when present. Defaults to the ObjectNav task via the registry
        to avoid a hard import cycle and keep the controller task-type agnostic.
        """
        if self.current_task is not None and getattr(self.current_task, "is_task_running", False):
            self._emit_status("Stopping previous task...")
            self.current_task.stop_task()
            if self.task_thread is not None and self.task_thread.is_alive():
                self.task_thread.join(timeout=2.0)

        self._emit_status(f"Starting Navigation: {instruction}")

        try:
            from tasks import TaskFactory
            self.current_task = TaskFactory("object_nav")(self.robot, instruction)
            self.set_task(self.current_task)

            self.is_task_running = True
            self.task_thread = threading.Thread(
                target=self._run_task_thread,
                args=(self.current_task,),
                daemon=True,
            )
            self.task_thread.start()
            return True, "ok"
        except Exception as exc:  # noqa: BLE001 - surface init failures to the UI
            print_error(f"Failed to start task: {exc}")
            self._emit_response(f"Task initialization failed: {exc}")
            self.is_task_running = False
            return False, str(exc)

    def _run_task_thread(self, task) -> None:
        try:
            task.run()
            self._emit_response("Task completed.")
        except Exception as exc:  # noqa: BLE001 - report task crashes to the UI
            print_error(f"Task execution error: {exc}")
            self._emit_response(f"Task error: {exc}")
        finally:
            self.is_task_running = False

    def stop_task(self) -> None:
        print_info("Stop signal received")
        self.is_task_running = False
        if self.current_task is not None:
            self.current_task.stop_task()

    # ------------------------------------------------------------------ #
    # Live video proxy
    # ------------------------------------------------------------------ #
    def get_front_image_jpeg(self):
        """Return the latest front-camera frame as JPEG bytes."""
        return self.robot.get_front_image_jpeg()
