"""Robot layer for Cobot Magic (hardware + motion + navigation).

Exposes the dual-arm mobile-base controller, the panorama arm gimbal, the
in-process iPlanner client, the LA/VA VLM navigation API, and the thin
integrated vision-navigation controller used by the web / demo entry point.
"""
from robot.arm_controller import AutomatedController
from robot.iplanner_client import IPlannerClient
from robot.nav_controller import IntegratedVisionNavController
from robot.navigation_api import LaViRANavigationAPI
from robot.robot_controller import RobotController

__all__ = [
    "RobotController",
    "AutomatedController",
    "IPlannerClient",
    "LaViRANavigationAPI",
    "IntegratedVisionNavController",
]
