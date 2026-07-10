# Agilex Cobot Magic — Uni-LaViRA 真实世界多任务导航

免训练的 **Uni-LaViRA** 技术栈——将预训练大模型通过 **语言 → 视觉 → 机器人动作** 的*翻译*方式——在 Agilex Cobot Magic 轮式双臂平台上运行 **VLN**、**ObjectNav** 和 **EQA**，无需微调；仅底层控制器与平台相关。

## 任务

| `--task` | 目标 |
|----------|------|
| `vln` | 遵循自然语言路径指令。 |
| `object_nav` | 到达指定物体（文本目标，如 `"椅子"`）。 |
| `eqa` | 导航到目标位置，然后回答关于场景的问题。 |
| `interact` | Web + 语音演示（运行时给定指令）。 |

所有任务共享同一流水线和同一套 LA/VA 模型；仅指令不同。

## 流水线

每个周期：机械臂按固定关节序列扫描，采集 7 视角 45° 全景图 → **LA** 更新 markdown TODO 列表并选择朝向（或 `stop`）→ 底盘转向该方向 → **VA** 返回 `STOP` 或下一路径点的边界框 → 边界框通过正前方相机深度反投影为机器人坐标系下的目标，由进程内的 iPlanner 通过纯追踪和持续重规划驱动。

## 目录结构

```
cobot_magic/
├── main.py            # 入口；--task {object_nav,vln,eqa,interact}
├── config.py          # 可通过环境变量覆盖的配置
├── prompts.py
├── utils.py
├── run_llama_server.sh   # 启动本地 llama.cpp 模型服务器
├── ai_client/         # LaViRAVisionClient（双端点 LA + VA）
├── robot/             # RobotController、arm_controller、iplanner_client、navigation_api、nav_controller
├── tasks/             # vln、object_nav、eqa（+ Factory/Registry）
├── iplanner/          # iPlanner / NavDP 模型（进程内运行；权重未随代码发布）
├── web/               # Flask + SocketIO 语音/文本演示
├── tools/             # make_videos.py
└── tests/
```

## 安装

```bash
conda create -n cobot_magic python=3.10 && conda activate cobot_magic
pip install -r requirements.txt
```

此外还需要：
- ROS Noetic，包含机械臂驱动（`/puppet/joint_left|right`）、正前方 RealSense 相机（`/camera/color/image_raw`、`/camera/aligned_depth_to_color/image_raw`）和侧方相机（`/camera_l|_r/color/image_raw`）。
- iPlanner 权重文件放置在 `iplanner/checkpoints/iplanner.pth`。

## 模型（llama.cpp）

一个本地 `Qwen3.5-27B-Q4_K_M` GGUF 模型通过 `llama-server`（OpenAI 兼容的 `/v1` 接口）同时服务 LA 和 VA：

```bash
llama-server --model path/to/Qwen3.5-27B-Q4_K_M.gguf --mmproj path/to/mmproj.gguf \
    --alias Qwen3.5-27B-Q4_K_M --host 0.0.0.0 --port 8000 --ctx-size 8192 --n-gpu-layers 999
```

`run_llama_server.sh` 封装了上述命令——可通过环境变量覆盖路径：

```bash
MODEL_PATH=path/to/Qwen3.5-27B-Q4_K_M.gguf MMPROJ_PATH=path/to/mmproj.gguf bash run_llama_server.sh
```

`--mmproj`（视觉投影器）是 LA 和 VA 调用所必需的。如需使用远程 API，请改为设置 `LA_*` / `VA_*` 环境变量。

## 运行

```bash
roscore
roslaunch your_cobot_magic_bringup.launch   # 机械臂驱动 + 相机

python main.py --task object_nav --instruction "椅子"
python main.py --task vln        --instruction "去左边的门"
python main.py --task eqa        --instruction "沙发是什么颜色的？"
python main.py --task interact              # Web + 语音演示，访问 https://<机器人IP>:5000
```

Web 演示需要自签名 `cert.pem`/`key.pem` 证书；语音功能需要本地 faster-whisper 模型。运行过程中可通过 `python tools/make_videos.py` 将帧编码为视频。

## 环境变量

关键环境变量（均可覆盖；默认值在 `config.py` 中）：

| 变量 | 默认值 |
|----------|---------|
| `LA_BASE_URL` / `LA_MODEL_NAME` | `http://localhost:8000/v1` / `Qwen3.5-27B-Q4_K_M` |
| `VA_BASE_URL` / `VA_MODEL_NAME` | 与 LA 相同（单本地模型） |
| `IPLANNER_CHECKPOINT` / `IPLANNER_DEVICE` | `iplanner/checkpoints/iplanner.pth` / `cuda:0` |
| `OUTPUT_ROOT` / `FLASK_SECRET_KEY` | `outputs` / `change-me` |

## 引用

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
