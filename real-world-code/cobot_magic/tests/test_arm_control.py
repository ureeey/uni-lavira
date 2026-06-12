"""Integration smoke test for the Cobot Magic dual-arm gimbal controller.

Requires a live ROS environment with the Cobot Magic arm topics publishing on
``/puppet/joint_left`` and ``/puppet/joint_right``. Skipped automatically when
``rospy`` is not importable (CI, non-ROS machines).

Run with::

    pytest tests/test_arm_control.py -m integration -v
"""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

# Allow running from the package root without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rospy = pytest.importorskip("rospy")

# Import ROS message types only after rospy is confirmed available.
from sensor_msgs.msg import JointState  # noqa: E402  (after importorskip)
from std_msgs.msg import Header  # noqa: E402

from robot.arm_controller import AutomatedController  # noqa: E402


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Topic connectivity smoke test
# ---------------------------------------------------------------------------

class _ArmStateListener:
    """Subscribe to both puppet arm topics and record first messages."""

    def __init__(self):
        self.left_received = False
        self.right_received = False
        self.left_data = None
        self.right_data = None
        self.lock = threading.Lock()

        rospy.Subscriber("/puppet/joint_left", JointState, self._left_cb)
        rospy.Subscriber("/puppet/joint_right", JointState, self._right_cb)

    def _left_cb(self, msg: JointState) -> None:
        with self.lock:
            if not self.left_received:
                rospy.loginfo("Received first message from /puppet/joint_left")
                self.left_received = True
            self.left_data = msg

    def _right_cb(self, msg: JointState) -> None:
        with self.lock:
            if not self.right_received:
                rospy.loginfo("Received first message from /puppet/joint_right")
                self.right_received = True
            self.right_data = msg

    def wait_for_both(self, timeout: float = 20.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not rospy.is_shutdown():
                with self.lock:
                    if self.left_received and self.right_received:
                        return True
            time.sleep(0.1)
        return False


@pytest.fixture(scope="module")
def ros_node():
    """Initialise a ROS node for the test module."""
    rospy.init_node("test_arm_control", anonymous=True)
    yield
    # rospy does not provide an explicit shutdown for test nodes;
    # the process exits cleanly after the test session.


def test_arm_topics_publish(ros_node):
    """Both puppet arm topics deliver a JointState within 20 seconds."""
    listener = _ArmStateListener()
    assert listener.wait_for_both(timeout=20.0), (
        "Did not receive data from both /puppet/joint_left and "
        "/puppet/joint_right within 20 s — ensure the robot driver is running."
    )


def test_arm_position_has_seven_joints(ros_node):
    """Each arm reports at least 7 joint positions."""
    listener = _ArmStateListener()
    listener.wait_for_both(timeout=20.0)

    with listener.lock:
        left_pos = list(listener.left_data.position) if listener.left_data else []
        right_pos = list(listener.right_data.position) if listener.right_data else []

    assert len(left_pos) >= 7, f"Left arm has only {len(left_pos)} joints"
    assert len(right_pos) >= 7, f"Right arm has only {len(right_pos)} joints"


def test_arm_hold_control_publish(ros_node):
    """Publishing the current position back to /master topics raises no errors."""
    listener = _ArmStateListener()
    if not listener.wait_for_both(timeout=20.0):
        pytest.skip("Arm topics not available — hardware not connected.")

    left_pub = rospy.Publisher("/master/joint_left", JointState, queue_size=10)
    right_pub = rospy.Publisher("/master/joint_right", JointState, queue_size=10)

    # Wait briefly for publishers to connect.
    rospy.sleep(0.5)

    rate = rospy.Rate(40)
    deadline = time.time() + 2.0
    while time.time() < deadline and not rospy.is_shutdown():
        with listener.lock:
            left_data = listener.left_data
            right_data = listener.right_data

        if left_data and right_data:
            ts = rospy.Time.now()

            left_msg = JointState()
            left_msg.header = Header(stamp=ts)
            left_msg.name = [f"joint{i}" for i in range(7)]
            left_msg.position = left_data.position[:7]
            left_pub.publish(left_msg)

            right_msg = JointState()
            right_msg.header = Header(stamp=ts)
            right_msg.name = [f"joint{i}" for i in range(7)]
            right_msg.position = right_data.position[:7]
            right_pub.publish(right_msg)

        rate.sleep()

    rospy.loginfo("Hold-position control test completed without errors.")
