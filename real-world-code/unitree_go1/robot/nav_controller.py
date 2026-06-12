"""
Thin integrated vision-navigation controller for the Go1 web / voice demo.

This controller does NOT own the per-cycle navigation loop.  That loop lives
inside the task objects (``tasks.VLNTask.run`` and friends).  The controller's
only jobs are:

- build / hold the active task,
- launch ``task.run()`` on a daemon thread for the web demo,
- mirror the source ``web_interface.py`` threading + SocketIO emit pattern
  (``status_update`` / ``response`` events),
- proxy the live front-camera JPEG to the browser.

For headless / text mode, callers can build the task directly via
``tasks.TaskFactory`` and call ``task.run()`` without this controller.
"""
import threading
from typing import Optional

from utils import print_error, print_info


class IntegratedVisionNavController:
    """Thin web-demo wrapper around a navigation task."""

    def __init__(self, robot, socketio_instance=None):
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

    def start_new_task(self, instruction: str):
        """Build / set a VLN task and run ``task.run()`` on a daemon thread.

        Mirrors source ``web_interface.execute_command``: stop any running
        task first, then launch the new one in the background and emit
        ``status_update`` / ``response`` events over SocketIO when present.
        """
        # Stop existing task if running
        if self.current_task is not None and getattr(self.current_task, "is_task_running", False):
            self._emit_status("Stopping previous task...")
            self.current_task.stop_task()
            if self.task_thread is not None and self.task_thread.is_alive():
                self.task_thread.join(timeout=2.0)

        self._emit_status(f"Starting Navigation: {instruction}")

        try:
            # Build the default VLN task via the registry (avoids a hard import
            # cycle and keeps the controller task-type agnostic).
            from tasks import TaskFactory
            self.current_task = TaskFactory("vln")(self.robot, instruction)
            self.set_task(self.current_task)

            self.is_task_running = True
            self.task_thread = threading.Thread(
                target=self._run_task_thread,
                args=(self.current_task,),
                daemon=True,
            )
            self.task_thread.start()
            return True, "ok"
        except Exception as e:
            print_error(f"Failed to start task: {e}")
            self._emit_response(f"Task initialization failed: {e}")
            self.is_task_running = False
            return False, str(e)

    def _run_task_thread(self, task) -> None:
        try:
            task.run()
            self._emit_response("Task completed.")
        except Exception as e:
            print_error(f"Task execution error: {e}")
            self._emit_response(f"Task error: {e}")
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
