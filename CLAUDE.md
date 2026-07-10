# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Uni-LaViRA is a **training-free** framework for embodied navigation that casts navigation as a three-layer translation — **Language → Vision → Robot Action** — using pretrained multimodal LLMs without fine-tuning. One architecture spans four navigation tasks (VLN, ObjectNav, EQA, Aerial VLN) and four heterogeneous robot platforms.

## Repository Structure

Each subdirectory is **self-contained** with its own conda environment, dependencies, and README. There is no shared Python package or unified build system.

```
sim-code/
  habitat/         # Indoor simulation: VLN-CE (R2R/RxR), HM3D-v2, HM3D-OVON, MP3D-EQA
  airsim/          # Aerial VLN simulation (TravelUAV / AirSim)
real-world-code/
  cobot_magic/     # Agilex Cobot Magic (wheeled bimanual platform)
  unitree_g1/      # Unitree G1 humanoid
  unitree_go1/     # Unitree Go1 quadruped
  self_built_uav/  # Custom quadrotor (ROS/catkin)
```

Each ground robot runs all of `vln` / `object_nav` / `eqa` / `interact` from a shared pipeline; only the instruction differs.

## Core Architecture — The LA/VA Pipeline

The system uses **two separate LLM endpoints** (Language Agent + Vision Agent):

1. **LA (Language Agent)**: Receives a multi-view panorama, updates a markdown TODO list, and decides the next heading (forward/left/right/behind) or STOP. Typically a strong reasoning model (e.g., Gemini 3.5 Flash).
2. **VA (Vision Agent)**: Given the front-facing RGB-D image after turning, returns a bounding box (`x1,y1,x2,y2`) around the next waypoint, or `STOP`. Typically a vision-capable model (e.g., Qwen3.5-27B).
3. **Local Planner**: Back-projects the bbox through depth to a world-coordinate goal, then drives toward it using FMM (Fast Marching Method), NavDP (learned planner for stairs), or iPlanner.

Both LA and VA are accessed through an OpenAI-compatible API via the `LaViRA_API` class (`sim-code/habitat/vlnce_baselines/utils/api.py`), which wraps two `openai.OpenAI` clients.

### Per-Step Cycle (simulation)

1. **No target set → Panorama collection**: Agent spins 360° (12×30° turns), collecting 4 frames at 90° intervals.
2. **LA decision**: Query LA with the panorama + history → decide direction or STOP.
3. **VA target**: Turn to chosen direction, query VA for a bbox on the front RGB frame → back-project to world coordinate.
4. **Local navigation**: FMM/NavDP/iPlanner drives toward the target waypoint. On arrival or timeout (15 steps), reset and return to step 1.
5. **STOP double-check**: When LA signals STOP, a fresh panorama is collected and LA is re-queried to confirm. Up to 3 rejections before forced stop.

### Key Source Files (Habitat simulation)

| File | Role |
|------|------|
| `sim-code/habitat/run_mp.py` | Entry point: multiprocess evaluation orchestrator. Spawns workers, each running `ZeroShotVlnEvaluatorMP`. |
| `sim-code/habitat/vlnce_baselines/ZS_Evaluator_mp.py` | Main evaluator (`ZeroShotVlnEvaluatorMP`): owns the rollout loop, map processing, panorama collection, and target management (~2200 lines). |
| `sim-code/habitat/vlnce_baselines/agent.py` | `VLMReasoningAgent`: prompt construction, LA navigate/backtrack/STOP logic, VA bbox query, TODO-list memory. Split from the evaluator for readability. |
| `sim-code/habitat/vlnce_baselines/utils/api.py` | `LaViRA_API`: dual-endpoint OpenAI-compatible client. Tracks per-model token usage. |
| `sim-code/habitat/vlnce_baselines/models/Policy.py` | `FusionMapPolicy`: FMM-based local planning. Combines value maps, collision maps, and detected classes to pick short-term goals. |
| `sim-code/habitat/vlnce_baselines/map/semantic_prediction.py` | `GroundedSAM`: open-vocabulary semantic segmentation (GroundingDINO + SAM/RepViTSAM). |
| `sim-code/habitat/vlnce_baselines/map/mapping.py` | `Semantic_Mapping`: builds and maintains the egocentric top-down map. |
| `sim-code/habitat/vlnce_baselines/prompts/` | Task-specific prompt templates (`prompts_vln.py`, `prompts_objnav.py`, `prompts_eqa.py`). |

### Key Source Files (Real-world — cobot_magic as reference)

All three ground-robot directories share the same architecture:

| File | Role |
|------|------|
| `main.py` | Entry point; `--task {vln,object_nav,eqa,interact}` |
| `config.py` | Env-var overridable configuration |
| `ai_client/vision_client.py` | `LaViRAVisionClient` — two-endpoint LA + VA (local llama.cpp server or remote API) |
| `robot/robot_controller.py` | Main robot orchestration (panorama capture, LA/VA loop, navigation dispatch) |
| `robot/nav_controller.py` | iPlanner-based local navigation with pure-pursuit |
| `tasks/` | Per-task entry points (vln, object_nav, eqa) sharing the same pipeline |
| `iplanner/` | iPlanner model and server (trajectory prediction from RGB-D + goal) |
| `web/app.py` | Flask + SocketIO web demo with voice input |

## Environment Variables

All model configuration is via environment variables (see `.env.example` in each subdirectory):

| Variable | Purpose |
|----------|---------|
| `LA_API_KEY`, `LA_BASE_URL`, `LA_MODEL_NAME` | Language Agent endpoint |
| `VA_API_KEY`, `VA_BASE_URL`, `VA_MODEL_NAME` | Vision Agent endpoint |
| `HF_HUB_OFFLINE=1` | Required for offline eval (prevents HuggingFace hub calls) |
| `BERT_LOCAL_PATH` | Path to local `bert-base-uncased` for GroundingDINO text encoder |
| `CUDA_VISIBLE_DEVICES` | GPU selection (defaults to `0`) |
| `NPROC` | Number of parallel workers for simulation eval (default 20) |

Recommended models: LA = `gemini-3.5-flash` (or `gemini-3.1-pro`); VA = `qwen3.5-27b`. For local deployment, a single `Qwen3.5-27B-Q4_K_M` GGUF with `llama-server` serves both roles.

## Running Simulation Evaluation

From `sim-code/habitat/`:

```bash
conda activate lavira
bash eval_scripts/vlnce_r2r.sh    # VLN-CE R2R
bash eval_scripts/vlnce_rxr.sh    # VLN-CE RxR
bash eval_scripts/hm3d_v2.sh      # HM3D-v2 ObjectNav
bash eval_scripts/hm3d_ovon.sh    # HM3D-OVON
bash eval_scripts/mp3d_eqa.sh     # MP3D-EQA
```

Each script calls `python run_mp.py` with the appropriate YAML config and an `--episode-file` pointing to a 100-episode stratified subset. Key flags:
- `--nprocesses` / `NPROC`: parallel workers (default 20)
- `--use-navdp`: enable NavDP for stair navigation
- `--no-fmm`: disable FMM (forces NavDP/iPlanner for all steps)
- `--episode-file`: path to JSON list of episode IDs
- `--debug-episodes`: comma-separated episode IDs for debugging
- `--resume-from-log`: skip episodes already present in a previous log

Visualization server: `python server.py` → browse to `http://localhost:9999/?root=saved_rgb_images/<exp_name>`.

## Running Real-Robot Deployment

Each robot directory follows the same pattern:

```bash
conda activate <robot_env>
# Start the model server (llama.cpp) or set LA_*/VA_* env vars for remote API
# Start the local planner (iPlanner server or NavDP server)
python main.py --task vln        --instruction "go to the door on the left"
python main.py --task object_nav --instruction "chair"
python main.py --task eqa        --instruction "what colour is the sofa?"
python main.py --task interact   # web + voice demo
```

## Docker (Habitat simulation)

```bash
cd sim-code/habitat
docker build -t lavira-oss:dev .
docker run -it --gpus all --env-file .env \
  -v $(pwd):/workspace/lavira-code \
  -v path/to/data:/workspace/lavira-code/data \
  -w /workspace/lavira-code \
  lavira-oss:dev bash
```

The Dockerfile builds Ubuntu 22.04 + CUDA 11.8 with habitat-sim 0.1.7, habitat-lab, GroundingDINO, and Segment-Anything all from source at pinned commits.

## Data Layout (Habitat simulation)

```
data/
├── grounded_sam/              # Model checkpoints
│   ├── groundingdino_swint_ogc.pth
│   ├── GroundingDINO_SwinT_OGC.py
│   ├── repvit_sam.pt
│   ├── sam_vit_h_4b8939.pth
│   └── bert-base-uncased/     # GroundingDINO text encoder (local copy)
├── datasets/
│   ├── stratified_samples/    # Ships with code: 100-episode ID lists
│   └── episodes/              # From Google Drive bundle
└── scene_datasets/
    ├── mp3d/                  # Matterport3D scenes (90 scenes)
    └── hm3d/hm3d_v0.2/val/    # HM3D-Semantics v0.2 val scenes (100 scenes)
```

NavDP checkpoint: `sim-code/habitat/navdp/navdp-cross-modal.ckpt` (not under `data/`).

## Key Design Decisions

- **Two-model split (LA vs VA)**: Language reasoning and visual grounding use separate models/endpoints. The LA model never sees raw images directly — it receives described panoramas. The VA model receives RGB-D frames and returns bounding boxes.
- **No fine-tuning anywhere**: All models are used off-the-shelf. The framework is pure prompt engineering + coordinate transformation.
- **Map-based state**: The simulator maintains an egocentric top-down semantic map (480×480 grid, 5cm resolution) built incrementally from GroundedSAM segmentations. This map feeds the FMM planner.
- **Panorama as state representation**: LA decisions are made from 4-view panoramas (0°, 90°, 180°, 270°), not single frames. This gives the LA model full situational awareness.
- **TODO-list memory**: The LA model maintains a markdown TODO list across steps for long-horizon reasoning, verified by a separate LLM call for consistency.
- **Backtrack with replanning**: When stuck, the agent can backtrack to a previous waypoint and replan from there, optionally with a second-chance re-query of the VA model.
- **Multiprocess evaluation**: Each GPU runs multiple workers (recommended ≤2 per GPU). Episodes are distributed statically or via a dynamic shared queue (`DYNAMIC_QUEUE` config option).

## Citation

```bibtex
@article{ding2026unilavira,
  title={Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation},
  author={Ding, Hongyu and others},
  journal={arXiv preprint arXiv:2605.27582},
  year={2026}
}
```
