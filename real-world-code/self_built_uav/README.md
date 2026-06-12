# Self-Built UAV — Real-World Aerial Navigation for Uni-LaViRA

ROS deployment of the training-free **Uni-LaViRA** stack on a self-built quadrotor: the same **language → vision → robot action** *translation*, here producing flight actions for real-world aerial navigation, with the VLN node, flight-control support node, and sensor drivers built for this airframe.

## Overview

- `model_set_node`: flight control support node for waypoint following and offboard setpoint publishing
- `vln_node`: vision-language navigation node
- `livox_ros_driver2`: Livox ROS driver
- `realsense`: RealSense ROS wrapper
- `FAST_LIO`: FAST-LIO2 LiDAR-inertial odometry (provides the `/Odometry` topic; git submodule)

## Dependencies

Python dependency installation can be based on `~/uni-lavira_ws/src/requirements.txt`.

NavDP model download and startup instructions:

- https://github.com/InternRobotics/NavDP

FAST-LIO2 startup reference:

- https://github.com/hku-mars/FAST_LIO

### Third-party submodules

The drivers and the NavDP planner are referenced as git submodules (declared in the repository-root `.gitmodules`). After cloning the repository, fetch them with:

`git submodule update --init --recursive`

> **Version note:** the RealSense and Livox drivers are pulled from their public GitHub repositories (`realsenseai/realsense-ros`, `Livox-SDK/livox_ros_driver2`); please check ROS-distro compatibility for your setup. NavDP runs as a separate server (`navdp_server.py --port 8888 --checkpoint <path>`) before launching the VLN node.

## Build

These are catkin packages. Place this directory's contents under the `src/` folder of a catkin workspace (e.g. `~/uni-lavira_ws/src/`), then fetch the submodules with `git submodule update --init --recursive` before building.

Build the Livox ROS driver first:

```bash
cd ~/uni-lavira_ws/src/livox_ros_driver2
./build.sh ROS1
```

After building, source the workspace:

```bash
source devel/setup.bash
```

## Startup Order

### 1. Start RealSense

```bash
roslaunch realsense2_camera rs_camera_vins.launch
```

### 2. Start FAST-LIO2

Please follow the official FAST-LIO startup instructions for your LiDAR configuration:

- https://github.com/hku-mars/FAST_LIO

### 3. Start the VLN node

The workspace currently contains [vln_node/launch/indoor_eval.launch](vln_node/launch/indoor_eval.launch).

```bash
roslaunch vln_node indoor_eval.launch
```

If you use a local wrapper or renamed launch file such as `direction_eval.launch`, make sure it points to the same node entry and configuration.

### 4. Start the flight control support node

```bash
roslaunch model_set_node model_set_node.launch
```

## Warning

If this system is run on the onboard computer, the aircraft will automatically take off after `model_set_node` is started.

Use caution and ensure the vehicle is in a safe environment before launching:

```bash
roslaunch model_set_node model_set_node.launch
```

## Flight Controller

The flight controller used by this project is `NxtPX4v2`.

## Notes

- Keep the NavDP model environment consistent with the official NavDP repository.
- Check camera, LiDAR, and MAVROS topic availability before launching the navigation stack.
- The `vln_node` launch file currently present in this repository is `indoor_eval.launch`.

## Citation

```bibtex
@article{ding2026unilavira,
  title   = {Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation},
  author  = {Ding, Hongyu and Zhang, Sizhuo and Xu, Ziming and Guo, Jinwen and Liu, Hongxiu and Cheng, Xingzhi and Chen, Zixuan and Qi, Haifei and Wang, Duo and Xu, Hao and Shi, Jieqi and Zhang, Yifan and Huo, Jing and Cheng, Jian and Gao, Yang and Luo, Jiebo},
  journal = {arXiv preprint arXiv:2605.27582},
  year    = {2026}
}
@article{ding2025lavira,
  title   = {LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision Language Navigation in Continuous Environments},
  author  = {Ding, Hongyu and Xu, Ziming and Fang, Yudong and Wu, You and Chen, Zixuan and Shi, Jieqi and Huo, Jing and Zhang, Yifan and Gao, Yang},
  journal = {arXiv preprint arXiv:2510.19655},
  year    = {2025}
}
```
