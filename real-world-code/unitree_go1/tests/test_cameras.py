"""
Integration test for camera capture on the Unitree Go1.

This test requires:
- A running ROS Noetic environment with roscore
- Camera drivers publishing to /camera{1,2,3,4}/color|depth/image_raw
- The Unitree SDK available on sys.path

Without ROS/hardware, the test is skipped cleanly via pytest.importorskip.

Run on the robot:
    pytest tests/test_cameras.py -m integration -v
"""
import os
import sys
import pytest

# Skip entire module when rospy is unavailable (CI / non-ROS environments)
rospy = pytest.importorskip("rospy")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.mark.integration
class TestCameraCapture:
    """Hardware-in-the-loop tests for 4-direction panorama capture."""

    @pytest.fixture(autouse=True)
    def setup_robot(self, tmp_path):
        """Initialise RobotController with a temporary output directory."""
        from config import Config
        from robot import RobotController

        Config.OUTPUT_DIR = str(tmp_path)
        Config.IMG_DIR = os.path.join(str(tmp_path), "images")
        Config.PANORAMA_DIR = os.path.join(Config.IMG_DIR, "panorama")
        Config.LOG_DIR = os.path.join(str(tmp_path), "logs")
        os.makedirs(Config.PANORAMA_DIR, exist_ok=True)

        self.robot = RobotController()

        # Allow time for ROS topics to start delivering frames
        rospy.sleep(3.0)

        yield

        # Cleanup
        try:
            self.robot.stop_saver()
            self.robot.stop_robot()
        except Exception:
            pass

    def test_capture_all_directions_succeeds(self):
        """capture_all_directions should return True when cameras are live."""
        result = self.robot.capture_all_directions(step=0)
        assert result is True, (
            "capture_all_directions returned False — check that "
            "/camera{1,2,3,4}/color/image_raw topics are publishing."
        )

    def test_panorama_images_saved(self):
        """Panorama images should be written to the configured directory."""
        from config import Config

        self.robot.capture_all_directions(step=1)
        step_dir = os.path.join(Config.PANORAMA_DIR, "step1")
        saved = [f for f in os.listdir(step_dir) if f.endswith(".png")]
        assert len(saved) > 0, (
            f"No panorama images found in {step_dir} — "
            "verify camera ROS topics and Config.PANORAMA_DIR."
        )
