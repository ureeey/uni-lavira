#!/usr/bin/env python3
"""
G1 Dead Reckoning Calibration Script
====================================
This script helps determine the correct 'linear_scale' and 'angular_scale'
parameters for the G1 robot controller's dead reckoning system.

This is an interactive hardware tool. It requires the G1 robot to be
powered on and connected via the configured network interface.

Usage:
    python3 tests/calibrate_scales.py
"""

import sys
import os
import time
import math
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot import RobotController
from config import Config
from utils import print_info, print_success, print_warning, print_error, print_action


def calibrate_linear(robot):
    print("\n" + "=" * 50)
    print("LINEAR SCALE CALIBRATION")
    print("=" * 50)
    print("This test will move the robot forward for a fixed duration.")
    print("Please ensure there is at least 2 meters of clear space in front of the robot.")

    target_dist = 1.0  # metres (theoretical)
    speed = 0.3        # m/s
    duration = target_dist / speed

    input("Press Enter to START moving forward...")

    print_action(f"Commanding Forward: {speed} m/s for {duration:.2f} s")
    print_info(f"Theoretical Distance (Unscaled): {target_dist} m")

    # Execute Move
    # Note: We access internal velocity control directly to bypass high-level planners
    robot.target_vx = speed
    robot.target_vy = 0.0
    robot.target_vyaw = 0.0

    time.sleep(duration)

    # Stop
    robot.target_vx = 0.0
    print_success("Stop command sent.")
    time.sleep(1.0)

    # Input Measurement
    while True:
        try:
            measured = float(input("\nEnter the ACTUAL measured distance traveled (in meters): "))
            if measured < 0:
                print("Distance cannot be negative.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Calculate
    # The code uses: d_dist = vx * dt * linear_scale
    # We want: Sum(d_dist) == Measured
    # Sum(vx * dt * scale) == Measured
    # scale * (vx * duration) == Measured
    # scale = Measured / (vx * duration)

    theoretical_raw = speed * duration
    recommended_scale = measured / theoretical_raw

    print("\n" + "-" * 30)
    print(f"Theoretical (Raw): {theoretical_raw:.4f} m")
    print(f"Measured:          {measured:.4f} m")
    print(f"Recommended Linear Scale: {recommended_scale:.4f}")
    print("-" * 30)

    print_info(f"Update 'linear_scale' in robot/robot_controller.py to: {recommended_scale:.2f}")


def calibrate_angular(robot):
    print("\n" + "=" * 50)
    print("ANGULAR SCALE CALIBRATION")
    print("=" * 50)
    print("This test will rotate the robot left for a fixed duration.")
    print("Please ensure the robot has clearance to rotate.")

    target_angle_deg = 90.0
    target_angle_rad = math.radians(target_angle_deg)
    speed = 0.4        # rad/s
    duration = target_angle_rad / speed

    input("Press Enter to START rotating left...")

    print_action(f"Commanding Rotate Left: {speed} rad/s for {duration:.2f} s")
    print_info(f"Theoretical Rotation (Unscaled): {target_angle_deg} degrees")

    # Execute Move
    robot.target_vx = 0.0
    robot.target_vy = 0.0
    robot.target_vyaw = speed

    time.sleep(duration)

    # Stop
    robot.target_vyaw = 0.0
    print_success("Stop command sent.")
    time.sleep(1.0)

    # Input Measurement
    while True:
        try:
            measured = float(input("\nEnter the ACTUAL measured rotation (in degrees): "))
            if measured < 0:
                print("Please enter positive degrees (magnitude).")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Calculate
    theoretical_raw_deg = math.degrees(speed * duration)
    recommended_scale = measured / theoretical_raw_deg

    print("\n" + "-" * 30)
    print(f"Theoretical (Raw): {theoretical_raw_deg:.4f} deg")
    print(f"Measured:          {measured:.4f} deg")
    print(f"Recommended Angular Scale: {recommended_scale:.4f}")
    print("-" * 30)

    print_info(f"Update 'angular_scale' in robot/robot_controller.py to: {recommended_scale:.2f}")


def main():
    print("Initializing Robot Controller...")
    try:
        robot = RobotController()
        print_success("Robot Controller Initialized.")
    except Exception as e:
        print_error(f"Failed to initialize robot: {e}")
        return

    while True:
        print("\n" + "=" * 30)
        print("G1 Calibration Menu")
        print("=" * 30)
        print("1. Calibrate Linear Scale (Move Forward)")
        print("2. Calibrate Angular Scale (Rotate)")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ")

        if choice == '1':
            calibrate_linear(robot)
        elif choice == '2':
            calibrate_angular(robot)
        elif choice == '3':
            print("Exiting...")
            robot.running = False
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
