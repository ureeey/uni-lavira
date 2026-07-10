# CLAUDE.md（中文版）

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指导。

## 项目概述

Uni-LaViRA 是一个**免训练**的具身导航框架，将导航建模为三层**翻译**——**语言 → 视觉 → 机器人动作**——使用预训练的多模态大语言模型，无需微调。同一套架构覆盖四项导航任务（VLN、ObjectNav、EQA、Aerial VLN）和四种异构机器人平台。

## 仓库结构

每个子目录是**自包含**的，拥有独立的 conda 环境、依赖和 README。没有共享的 Python 包或统一的构建系统。

```
sim-code/
  habitat/         # 室内仿真：VLN-CE (R2R/RxR)、HM3D-v2、HM3D-OVON、MP3D-EQA
  airsim/          # 空中 VLN 仿真（TravelUAV / AirSim）
real-world-code/
  cobot_magic/     # Agilex Cobot Magic（轮式双臂平台）
  unitree_g1/      # Unitree G1 人形机器人
  unitree_go1/     # Unitree Go1 四足机器人
  self_built_uav/  # 自研四旋翼无人机（ROS/catkin）
```

每台地面机器人都通过共享流水线运行 `vln` / `object_nav` / `eqa` / `interact` 全部任务；仅指令不同。

## 核心架构 — LA/VA 流水线

系统使用**两个独立的大模型端点**（语言代理 + 视觉代理）：

1. **LA（语言代理）**：接收多视角全景图，更新 markdown 格式的 TODO 列表，决定下一步朝向（前/左/右/后）或 STOP。通常使用强推理模型（如 Gemini 3.5 Flash）。
2. **VA（视觉代理）**：在机器人转向后，接收正前方 RGB-D 图像，返回下一路径点周围的边界框（`x1,y1,x2,y2`）或 `STOP`。通常使用视觉模型（如 Qwen3.5-27B）。
3. **局部规划器**：通过深度图将边界框反投影为世界坐标目标，然后使用 FMM（快速行进法）、NavDP（面向楼梯的学习型规划器）或 iPlanner 向其驱动。

LA 和 VA 均通过 `LaViRA_API` 类（`sim-code/habitat/vlnce_baselines/utils/api.py`）访问 OpenAI 兼容 API，该类封装了两个 `openai.OpenAI` 客户端。

### 单步循环（仿真）

1. **无目标 → 全景采集**：代理原地旋转 360°（12×30° 转动），在 90° 间隔处采集 4 帧。
2. **LA 决策**：向 LA 发送全景图 + 历史记录 → 决定方向或 STOP。
3. **VA 目标**：转向选定方向，向 VA 查询正前方 RGB 帧上的边界框 → 反投影为世界坐标。
4. **局部导航**：FMM/NavDP/iPlanner 驱动代理向目标路径点前进。到达或超时（15 步）后，重置并返回步骤 1。
5. **STOP 二次确认**：当 LA 发出 STOP 信号时，重新采集全景图并再次向 LA 查询确认。最多 3 次拒绝后强制停止。

### 核心源文件（Habitat 仿真）

| 文件 | 作用 |
|------|------|
| `sim-code/habitat/run_mp.py` | 入口：多进程评估编排器。创建工作进程，每个运行 `ZeroShotVlnEvaluatorMP`。 |
| `sim-code/habitat/vlnce_baselines/ZS_Evaluator_mp.py` | 主评估器（`ZeroShotVlnEvaluatorMP`）：包含 rollout 循环、地图处理、全景采集和目标管理（约 2200 行）。 |
| `sim-code/habitat/vlnce_baselines/agent.py` | `VLMReasoningAgent`：prompt 构建、LA 导航/回溯/STOP 逻辑、VA 边界框查询、TODO 列表记忆。从评估器中拆分以提高可读性。 |
| `sim-code/habitat/vlnce_baselines/utils/api.py` | `LaViRA_API`：双端点 OpenAI 兼容客户端。追踪每个模型的 token 用量。 |
| `sim-code/habitat/vlnce_baselines/models/Policy.py` | `FusionMapPolicy`：基于 FMM 的局部规划。结合价值地图、碰撞地图和检测类别来选择短期目标。 |
| `sim-code/habitat/vlnce_baselines/map/semantic_prediction.py` | `GroundedSAM`：开放词汇语义分割（GroundingDINO + SAM/RepViTSAM）。 |
| `sim-code/habitat/vlnce_baselines/map/mapping.py` | `Semantic_Mapping`：构建和维护以自我为中心的自顶向下地图。 |
| `sim-code/habitat/vlnce_baselines/prompts/` | 任务特定的 prompt 模板（`prompts_vln.py`、`prompts_objnav.py`、`prompts_eqa.py`）。 |

### 核心源文件（真实世界 — 以 cobot_magic 为例）

三个地面机器人目录共享相同的架构：

| 文件 | 作用 |
|------|------|
| `main.py` | 入口；`--task {vln,object_nav,eqa,interact}` |
| `config.py` | 可通过环境变量覆盖的配置 |
| `ai_client/vision_client.py` | `LaViRAVisionClient` — 双端点 LA + VA（本地 llama.cpp 服务器或远程 API） |
| `robot/robot_controller.py` | 主机器人编排（全景采集、LA/VA 循环、导航调度） |
| `robot/nav_controller.py` | 基于 iPlanner 的纯追踪局部导航 |
| `tasks/` | 各任务入口（vln、object_nav、eqa），共享同一流水线 |
| `iplanner/` | iPlanner 模型和服务端（从 RGB-D + 目标预测轨迹） |
| `web/app.py` | 带语音输入的 Flask + SocketIO Web 演示 |

## 环境变量

所有模型配置均通过环境变量完成（参见各子目录下的 `.env.example`）：

| 变量 | 用途 |
|----------|---------|
| `LA_API_KEY`、`LA_BASE_URL`、`LA_MODEL_NAME` | 语言代理端点 |
| `VA_API_KEY`、`VA_BASE_URL`、`VA_MODEL_NAME` | 视觉代理端点 |
| `HF_HUB_OFFLINE=1` | 离线评估所需（阻止 HuggingFace Hub 调用） |
| `BERT_LOCAL_PATH` | GroundingDINO 文本编码器所需的本地 `bert-base-uncased` 路径 |
| `CUDA_VISIBLE_DEVICES` | GPU 选择（默认 `0`） |
| `NPROC` | 仿真评估的并行工作进程数（默认 20） |

推荐模型：LA = `gemini-3.5-flash`（或 `gemini-3.1-pro`）；VA = `qwen3.5-27b`。本地部署时，单个 `Qwen3.5-27B-Q4_K_M` GGUF 配合 `llama-server` 可同时服务两种角色。

## 运行仿真评估

在 `sim-code/habitat/` 目录下：

```bash
conda activate lavira
bash eval_scripts/vlnce_r2r.sh    # VLN-CE R2R
bash eval_scripts/vlnce_rxr.sh    # VLN-CE RxR
bash eval_scripts/hm3d_v2.sh      # HM3D-v2 ObjectNav
bash eval_scripts/hm3d_ovon.sh    # HM3D-OVON
bash eval_scripts/mp3d_eqa.sh     # MP3D-EQA
```

每个脚本使用对应的 YAML 配置和指向 100 集分层子集的 `--episode-file` 调用 `python run_mp.py`。关键参数：
- `--nprocesses` / `NPROC`：并行工作进程数（默认 20）
- `--use-navdp`：启用 NavDP 用于楼梯导航
- `--no-fmm`：禁用 FMM（强制所有步骤使用 NavDP/iPlanner）
- `--episode-file`：包含 episode ID 列表的 JSON 文件路径
- `--debug-episodes`：逗号分隔的 episode ID，用于调试
- `--resume-from-log`：跳过之前日志中已存在的 episode

可视化服务：`python server.py` → 浏览器访问 `http://localhost:9999/?root=saved_rgb_images/<exp_name>`。

## 运行真实机器人部署

每个机器人目录遵循相同的模式：

```bash
conda activate <robot_env>
# 启动模型服务器（llama.cpp）或设置 LA_*/VA_* 环境变量指向远程 API
# 启动局部规划器（iPlanner 服务器或 NavDP 服务器）
python main.py --task vln        --instruction "去左边的门"
python main.py --task object_nav --instruction "椅子"
python main.py --task eqa        --instruction "沙发是什么颜色的？"
python main.py --task interact   # Web + 语音演示
```

## Docker（Habitat 仿真）

```bash
cd sim-code/habitat
docker build -t lavira-oss:dev .
docker run -it --gpus all --env-file .env \
  -v $(pwd):/workspace/lavira-code \
  -v path/to/data:/workspace/lavira-code/data \
  -w /workspace/lavira-code \
  lavira-oss:dev bash
```

Dockerfile 在 Ubuntu 22.04 + CUDA 11.8 基础上，从固定 commit 源码构建 habitat-sim 0.1.7、habitat-lab、GroundingDINO 和 Segment-Anything。

## 数据布局（Habitat 仿真）

```
data/
├── grounded_sam/              # 模型权重
│   ├── groundingdino_swint_ogc.pth
│   ├── GroundingDINO_SwinT_OGC.py
│   ├── repvit_sam.pt
│   ├── sam_vit_h_4b8939.pth
│   └── bert-base-uncased/     # GroundingDINO 文本编码器（本地副本）
├── datasets/
│   ├── stratified_samples/    # 随代码发布：100 集 ID 列表
│   └── episodes/              # 来自 Google Drive 压缩包
└── scene_datasets/
    ├── mp3d/                  # Matterport3D 场景（90 个场景）
    └── hm3d/hm3d_v0.2/val/    # HM3D-Semantics v0.2 验证场景（100 个场景）
```

NavDP 权重：`sim-code/habitat/navdp/navdp-cross-modal.ckpt`（不在 `data/` 目录下）。

## 核心设计决策

- **双模型分离（LA vs VA）**：语言推理和视觉定位使用不同的模型/端点。LA 模型从不直接查看原始图像——它接收描述后的全景图。VA 模型接收 RGB-D 帧并返回边界框。
- **全流程免微调**：所有模型均为现成预训练模型直接使用。框架本质上是纯 prompt 工程 + 坐标变换。
- **基于地图的状态**：仿真器维护一个以自我为中心的自顶向下语义地图（480×480 网格，5cm 分辨率），通过 GroundedSAM 分割逐步构建。该地图为 FMM 规划器提供输入。
- **全景图作为状态表示**：LA 决策基于 4 视角全景图（0°、90°、180°、270°），而非单帧图像。这为 LA 模型提供了完整的场景感知能力。
- **TODO 列表记忆**：LA 模型在步骤间维护一个 markdown 格式的 TODO 列表以支持长程推理，并通过独立的 LLM 调用来验证一致性。
- **带回溯重规划**：当陷入困境时，代理可以回溯到之前的路径点并从那里重新规划，可选择性地对 VA 模型进行二次查询。
- **多进程评估**：每个 GPU 运行多个工作进程（建议每 GPU ≤2 个）。Episode 可以静态分配或通过动态共享队列（`DYNAMIC_QUEUE` 配置项）分配。

## 引用

```bibtex
@article{ding2026unilavira,
  title={Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation},
  author={Ding, Hongyu and others},
  journal={arXiv preprint arXiv:2605.27582},
  year={2026}
}
```
