#!/usr/bin/env python3
"""
Test script to control Unitree G1 robot using keyboard.
Run this script from the project root or tests directory.

This is an interactive hardware tool. It requires the G1 robot to be
powered on and connected via the configured network interface.

Controls:
    W / S : Increase / Decrease forward speed (vx)
    A / D : Increase / Decrease lateral speed (vy)
    Q / E : Increase / Decrease yaw rate (vyaw)
    Space : STOP (reset all velocities to 0)
    ESC   : Quit

Usage:
    python tests/test_g1_keyboard_control.py
"""

import sys
import os
import tty
import termios
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from robot import RobotController
except ImportError:
    print(
        "Error: Could not import RobotController. "
        "Make sure you are in the project root or the environment is set up."
    )
    sys.exit(1)


def get_key():
    """Read a single key from stdin without blocking for Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main():
    print("Initializing RobotController...")
    robot = RobotController()

    print("\n" + "=" * 60)
    print("Unitree G1 Keyboard Control Test")
    print("=" * 60)
    print("Controls (Incremental):")
    print("  W / S : Increase/Decrease Forward Speed (vx)")
    print("  A / D : Increase/Decrease Left Speed (vy)")
    print("  Q / E : Increase/Decrease Turn Speed (vyaw)")
    print("  Space : STOP (Reset all velocities to 0)")
    print("  ESC   : Quit")
    print("-" * 60)
    print("Press keys to adjust velocity...")

    vx = 0.0
    vy = 0.0
    vyaw = 0.0

    step_v = 0.1    # m/s increment
    step_yaw = 0.2  # rad/s increment

    # Safety limits
    MAX_V = 0.6
    MAX_YAW = 1.0

    try:
        while True:
            key = get_key()

            # Handle exit keys
            if key == '\x1b' or key == '\x03':  # ESC or Ctrl+C
                break

            # Handle control keys
            if key == 'w':
                vx += step_v
            elif key == 's':
                vx -= step_v
            elif key == 'a':
                vy += step_v
            elif key == 'd':
                vy -= step_v
            elif key == 'q':
                vyaw += step_yaw
            elif key == 'e':
                vyaw -= step_yaw
            elif key == ' ':
                vx = 0.0
                vy = 0.0
                vyaw = 0.0

            # Clamp values
            vx = max(min(vx, MAX_V), -MAX_V)
            vy = max(min(vy, MAX_V), -MAX_V)
            vyaw = max(min(vyaw, MAX_YAW), -MAX_YAW)

            # Snap small values to zero
            if abs(vx) < 0.01:
                vx = 0.0
            if abs(vy) < 0.01:
                vy = 0.0
            if abs(vyaw) < 0.01:
                vyaw = 0.0

            # Send command via internal velocity API
            robot._set_velocity(vx, vy, vyaw)

            # Print status on the same line
            status = f"\r[STATUS] vx={vx:.2f} m/s, vy={vy:.2f} m/s, yaw_rate={vyaw:.2f} rad/s    "
            sys.stdout.write(status)
            sys.stdout.flush()

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nStopping robot...")
        robot._set_velocity(0, 0, 0)
        time.sleep(0.5)
        robot.running = False
        print("Exited.")


if __name__ == "__main__":
    main()
