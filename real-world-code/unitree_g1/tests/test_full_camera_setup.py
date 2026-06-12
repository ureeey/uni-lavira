#!/usr/bin/env python3
"""
Test script for Full Camera Setup on Unitree G1 (LaViRA).
Target Setup:
- Front: Orbbec Gemini 336L (RGB + Depth) -> USB 3.0 Hub
- Left:  Orbbec Gemini 336L (RGB)         -> USB 3.0 Hub
- Right: Orbbec Gemini 336L (RGB)         -> USB 3.0 Hub
- Rear:  Orbbec Gemini 336L (RGB)         -> USB 2.0 Port

Camera serial numbers are read from environment variables (Config):
    ORBBEC_FRONT_SERIAL, ORBBEC_LEFT_SERIAL, ORBBEC_RIGHT_SERIAL, ORBBEC_REAR_SERIAL

This script uses pyorbbecsdk to:
1. Detect all connected Orbbec cameras.
2. Identify the Rear camera via USB connection type (USB 2.0) or V4L2 fallback.
3. Identify Front/Left/Right cameras (USB 3.0) via Serial Number.
4. Capture and save:
   - RGB images from ALL cameras.
   - Depth images (visualized) ONLY from the Front camera.

Feature:
- Pre-initializes Rear Camera (USB 2.0) via V4L2 to avoid SDK conflict.

Hardware note:
    This script requires the pyorbbecsdk library and physical Orbbec Gemini 336L
    cameras to be connected.  Without hardware it is automatically skipped.

Usage:
    python -m pytest tests/test_full_camera_setup.py -v
    # or:
    python tests/test_full_camera_setup.py
"""

import sys
import os
import time
import subprocess

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Skip the entire module when pyorbbecsdk is not installed
pyorbbecsdk = pytest_mod = None
try:
    import pytest as pytest_mod
except ImportError:
    pass

# Guard: skip module-level import of pyorbbecsdk when running under pytest
# without hardware (pytest.importorskip handles the skip cleanly).
if pytest_mod is not None:
    pyorbbecsdk = pytest_mod.importorskip(
        "pyorbbecsdk",
        reason="pyorbbecsdk not installed — hardware test skipped",
    )
else:
    try:
        import pyorbbecsdk
    except ImportError:
        print("[ERROR] pyorbbecsdk not found. Please install it first.")
        sys.exit(1)

import cv2
import numpy as np

try:
    from utils import print_info, print_success, print_error, print_warning
except ImportError:
    def print_info(msg): print(f"[INFO] {msg}")
    def print_success(msg): print(f"[SUCCESS] {msg}")
    def print_error(msg): print(f"[ERROR] {msg}")
    def print_warning(msg): print(f"[WARNING] {msg}")

from config import Config

# Serial Number Mapping — loaded from Config (env-var driven; no literals shipped)
SERIAL_TO_POSITION = Config.serial_to_position()
# Add rear serial if configured
if Config.ORBBEC_REAR_SERIAL:
    SERIAL_TO_POSITION[Config.ORBBEC_REAR_SERIAL] = "Rear"


def get_video_device_by_serial(target_serials, require_mjpg=False):
    """
    Find /dev/videoX device matching one of the target serial numbers.
    If require_mjpg is True, also checks if the device supports MJPG format.

    Args:
        target_serials: Single serial string or list of serial strings.
        require_mjpg: When True, prefer devices that support MJPG format.

    Returns:
        Device path string (e.g. '/dev/video0') or None if not found.
    """
    if isinstance(target_serials, str):
        target_serials = [target_serials]

    import glob
    video_devs = sorted(glob.glob("/dev/video*"))

    candidates = []

    for dev in video_devs:
        try:
            cmd = ["udevadm", "info", "--query=all", f"--name={dev}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout

            matched = any(
                f"ID_SERIAL_SHORT={serial}" in output
                for serial in target_serials
            )

            if matched:
                if require_mjpg:
                    cmd_fmt = ["v4l2-ctl", "-d", dev, "--list-formats"]
                    res_fmt = subprocess.run(cmd_fmt, capture_output=True, text=True)
                    if "MJPG" in res_fmt.stdout:
                        return dev
                    else:
                        candidates.append(dev)
                else:
                    return dev
        except Exception:
            continue

    if require_mjpg and candidates:
        return candidates[0]

    return None


def main():
    print_info("Starting Full Camera Setup Test...")

    # Define output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
    os.makedirs(output_dir, exist_ok=True)
    print_info(f"Output directory: {output_dir}")

    # --- 0. Pre-initialize Rear Camera (V4L2) ---
    # Done BEFORE SDK initialization because the SDK scan can put the USB 2.0
    # device into a bad state, preventing V4L2 from working later.
    rear_cap = None

    rear_serials = [Config.ORBBEC_REAR_SERIAL] if Config.ORBBEC_REAR_SERIAL else []
    rear_dev_path = (
        get_video_device_by_serial(rear_serials, require_mjpg=True)
        if rear_serials else None
    )

    if not rear_dev_path:
        print_warning(
            f"Rear camera not found by serial number. "
            f"Falling back to {Config.ORBBEC_REAR_DEV_FALLBACK}"
        )
        rear_dev_path = Config.ORBBEC_REAR_DEV_FALLBACK

    if os.path.exists(rear_dev_path):
        print_info(f"Pre-attempting to open Rear Camera at {rear_dev_path} via V4L2...")
        temp_cap = cv2.VideoCapture(rear_dev_path)
        if temp_cap.isOpened():
            temp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Warm up: allow auto-exposure to settle
            for _ in range(60):
                temp_cap.read()
                time.sleep(0.01)

            ret, _ = temp_cap.read()
            if ret:
                print_success(f"  Rear Camera ({rear_dev_path}) pre-initialized successfully.")
                rear_cap = temp_cap
            else:
                print_warning(f"  Rear Camera ({rear_dev_path}) opened but failed to read. Releasing.")
                temp_cap.release()
        else:
            print_warning(f"  Failed to open {rear_dev_path}.")
    else:
        print_warning(f"  Rear camera device {rear_dev_path} not found.")

    # --- Initialize Context ---
    from pyorbbecsdk import Context, OBLogLevel, Pipeline, OBSensorType, OBFormat

    context = Context()
    context.set_logger_level(OBLogLevel.WARNING)

    try:
        device_list = context.query_devices()
        device_count = device_list.get_count()
    except Exception as e:
        print_error(f"Failed to query devices: {e}")
        device_count = 0

    print_info(f"Found {device_count} Orbbec devices via SDK.")

    # Categorize Devices
    rear_candidates_sdk = []
    front_side_candidates = []

    for i in range(device_count):
        device = device_list.get_device_by_index(i)
        device_info = device.get_device_info()
        serial = device_info.get_serial_number()
        connection_type = device_info.get_connection_type()

        position = SERIAL_TO_POSITION.get(serial, "Unknown")
        print_info(f"Device {i} [SN:{serial}]: Type={connection_type}, Position={position}")

        ctype_str = str(connection_type).upper()
        if "2.0" in ctype_str or "USB2" in ctype_str:
            rear_candidates_sdk.append(i)
        else:
            front_side_candidates.append(i)

    print_info(f"Identified {len(rear_candidates_sdk)} Rear candidate(s) (USB 2.0) via SDK.")
    print_info(f"Identified {len(front_side_candidates)} Front/Side candidate(s) (USB 3.0) via SDK.")

    def capture_stream(device_index, tag, stream_type):
        """Capture a single stream from a device and save to output_dir."""
        try:
            device = device_list.get_device_by_index(device_index)
            device_info = device.get_device_info()
            serial = device_info.get_serial_number()
            position_name = SERIAL_TO_POSITION.get(serial, tag)

            stream_name = "RGB" if stream_type == OBSensorType.COLOR_SENSOR else "Depth"
            print_info(f"--- Testing {position_name} {stream_name} (Index {device_index}, SN: {serial}) ---")

            from pyorbbecsdk import Config as OBConfig
            pipeline = Pipeline(device)
            ob_config = OBConfig()

            profiles = pipeline.get_stream_profile_list(stream_type)
            profile = None

            if stream_type == OBSensorType.COLOR_SENSOR:
                try:
                    profile = profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
                except Exception:
                    profile = profiles.get_profile_by_index(0)
            else:
                try:
                    profile = profiles.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
                except Exception:
                    profile = profiles.get_profile_by_index(0)

            if profile:
                ob_config.enable_stream(profile)
            else:
                print_error(f"  No profile found for {stream_name}")
                return

            pipeline.start(ob_config)

            print_info("    Warming up stream...")
            for _ in range(60):
                pipeline.wait_for_frames(100)

            time.sleep(0.5)

            frames = pipeline.wait_for_frames(2000)
            if frames:
                if stream_type == OBSensorType.COLOR_SENSOR:
                    frame = frames.get_color_frame()
                    if frame is not None:
                        data = frame.get_data()
                        if data is not None and (
                            not isinstance(data, (list, tuple, np.ndarray)) or len(data) > 0
                        ):
                            w, h = frame.get_width(), frame.get_height()
                            try:
                                img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                fname = f"{position_name}_rgb_{serial}.png"
                                cv2.imwrite(os.path.join(output_dir, fname), img_bgr)
                                print_success(f"  Saved RGB: {fname}")
                            except Exception as e:
                                print_error(f"  RGB process failed: {e}")
                        else:
                            print_error("  Empty RGB data")
                    else:
                        print_error("  No RGB frame")

                elif stream_type == OBSensorType.DEPTH_SENSOR:
                    frame = frames.get_depth_frame()
                    if frame is not None:
                        data = frame.get_data()
                        if data is not None and (
                            not isinstance(data, (list, tuple, np.ndarray)) or len(data) > 0
                        ):
                            w, h = frame.get_width(), frame.get_height()
                            try:
                                img = np.frombuffer(data, dtype=np.uint16).reshape((h, w))
                                depth_norm = cv2.normalize(
                                    img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                                )
                                depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                                fname = f"{position_name}_depth_vis_{serial}.png"
                                cv2.imwrite(os.path.join(output_dir, fname), depth_vis)
                                print_success(f"  Saved Depth: {fname}")
                            except Exception as e:
                                print_error(f"  Depth process failed: {e}")
                        else:
                            print_error("  Empty Depth data")
                    else:
                        print_error("  No Depth frame")
            else:
                print_error("  Timeout waiting for frames.")

            pipeline.stop()

        except Exception as e:
            print_error(f"  Capture failed: {e}")

    # --- Execute Tests Sequentially ---

    # 1. Test Rear Candidates
    if rear_cap:
        print_info(f"--- Testing Rear (Pre-init V4L2: {rear_dev_path}) ---")
        ret, frame = rear_cap.read()
        if ret and frame is not None:
            fname = "Rear_rgb_fallback.png"
            cv2.imwrite(os.path.join(output_dir, fname), frame)
            print_success(f"  Saved RGB (Fallback): {fname}")
        else:
            print_error("  Fallback capture failed (empty frame)")
        rear_cap.release()

    if rear_candidates_sdk:
        for idx in rear_candidates_sdk:
            capture_stream(idx, "Rear", OBSensorType.COLOR_SENSOR)

    # 2. Test Front/Side Candidates
    if not front_side_candidates:
        print_warning("No Front/Side Cameras (USB 3.0) detected.")
    else:
        for idx in front_side_candidates:
            device = device_list.get_device_by_index(idx)
            serial = device.get_device_info().get_serial_number()
            position = SERIAL_TO_POSITION.get(serial, "Unknown")

            capture_stream(idx, "FrontSide_Candidate", OBSensorType.COLOR_SENSOR)

            if position == "front":
                capture_stream(idx, "FrontSide_Candidate", OBSensorType.DEPTH_SENSOR)
            else:
                print_info(f"Skipping Depth for {position} (SN: {serial}) — only front requires Depth.")

    # --- Summary ---
    print_info("--- Camera Detection Summary ---")

    detected_serials_sdk = []
    try:
        count = device_list.get_count()
        for i in range(count):
            d = device_list.get_device_by_index(i)
            detected_serials_sdk.append(d.get_device_info().get_serial_number())
    except Exception:
        pass

    for serial, pos in SERIAL_TO_POSITION.items():
        found = False
        if pos == "Rear" and rear_cap:
            found = True
        if serial in detected_serials_sdk:
            found = True

        if found:
            print_success(f"  {pos} ({serial}): DETECTED")
        else:
            print_error(f"  {pos} ({serial}): NOT DETECTED (Check Hardware/Connections)")

    print_success("Full Camera Setup Test Complete.")
    print_info(f"Check images in {output_dir}")


def test_full_camera_setup():
    """Pytest entry point — delegates to main()."""
    main()


if __name__ == "__main__":
    main()
