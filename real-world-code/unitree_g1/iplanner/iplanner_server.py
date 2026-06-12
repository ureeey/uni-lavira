"""
iPlanner Server for LaViRA G1
===============================
Flask-based server that runs the iPlanner neural network model
for trajectory planning from depth images and goal points.

Usage:
    python iplanner_server.py --port 8888
"""

import os
import sys
import json
import time
import datetime
import argparse
import atexit

import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify

from iplanner_agent import IPlannerAgent

parser = argparse.ArgumentParser(description="iPlanner Server for LaViRA G1")
parser.add_argument("--port", type=int, default=8888, help="Server port")
parser.add_argument(
    "--config",
    type=str,
    default="./configs/iplanner.yaml",
    help="Path to iPlanner config",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./checkpoints/iplanner.pth",
    help="Path to iPlanner model checkpoint",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    choices=["cpu", "cuda"],
    help="Device for inference",
)
args = parser.parse_known_args()[0]

app = Flask(__name__)
iplanner_navigator = None


@app.route("/navigator_reset", methods=["POST"])
def iplanner_reset():
    """Initialize or reset the iPlanner model."""
    global iplanner_navigator
    intrinsic = np.array(request.get_json().get("intrinsic"))
    threshold = np.array(request.get_json().get("stop_threshold"))
    batchsize = np.array(request.get_json().get("batch_size"))

    if iplanner_navigator is None:
        iplanner_navigator = IPlannerAgent(
            intrinsic,
            model_path=args.checkpoint,
            model_config_path=args.config,
            device=args.device,
        )
        print(f"[iPlanner] Model loaded on {args.device}")

    return jsonify({"algo": "iplanner"})


@app.route("/navigator_reset_env", methods=["POST"])
def iplanner_reset_env():
    """Reset environment (placeholder)."""
    return jsonify({"algo": "iplanner"})


def process_goal(goal, range=5.0):
    """Clip goal coordinates to the valid planning range."""
    return_goal = np.clip(goal, -range, range)
    return return_goal


@app.route("/pointgoal_step", methods=["POST"])
def iplanner_step_pointgoal():
    """
    Process a single planning step.

    Expects:
        - image: RGB image file (PNG)
        - depth: Depth image file (PNG, in 0.1 mm units)
        - goal_data: JSON with goal_x and goal_y arrays

    Returns:
        - trajectory: Planned waypoints [batch, N, 3]
        - all_values: Fear/cost values
    """
    global iplanner_navigator

    image_file = request.files["image"]
    depth_file = request.files["depth"]
    goal_data = json.loads(request.form.get("goal_data"))
    goal_x = np.array(goal_data["goal_x"])
    goal_y = np.array(goal_data["goal_y"])
    goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)
    goal = process_goal(goal)
    batch_size = goal.shape[0]

    # Process RGB image
    image = Image.open(image_file.stream)
    image = image.convert("RGB")
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    # Process depth image (0.1 mm -> meters)
    depth = Image.open(depth_file.stream)
    depth = depth.convert("I")
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    # Run planner
    _, trajectory, fear = iplanner_navigator.step_pointgoal(depth, goal)

    return jsonify(
        {
            "trajectory": trajectory.cpu().numpy().tolist(),
            "all_trajectory": trajectory.cpu().numpy()[None, :, :, :].tolist(),
            "all_values": fear.cpu().numpy().tolist(),
        }
    )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "model_loaded": iplanner_navigator is not None,
            "device": args.device,
        }
    )


def cleanup():
    """Cleanup on server exit."""
    print("[iPlanner] Server shutting down...")


atexit.register(cleanup)


if __name__ == "__main__":
    print(f"[iPlanner] Starting server on port {args.port}")
    print(f"[iPlanner] Config: {args.config}")
    print(f"[iPlanner] Checkpoint: {args.checkpoint}")
    print(f"[iPlanner] Device: {args.device}")
    app.run(host="0.0.0.0", port=args.port)
