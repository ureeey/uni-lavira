# `bash eval_scripts/vlnce_r2r.sh` 完整调用追踪

## 概览

```
vlnce_r2r.sh
  └─ python run_mp.py  (多进程编排)
       └─ ZeroShotVlnEvaluatorMP  (trainer, 每个 worker 一个实例)
            ├─ _init_envs()         → construct_envs() → 加载 Habitat 场景
            ├─ _collect_val_traj()  → 加载 GT 轨迹 (用于 NDTW)
            ├─ _initialize_policy() → GroundedSAM + Semantic_Mapping + FusionMapPolicy
            └─ eval() / eval_dynamic()
                 └─ for each episode:
                      ├─ _maps_initialization()  → envs.reset() + 首次建图
                      ├─ rollout()               → 主循环 (≤500 步)
                      │    ├─ [Step 1] get_panorama()       → 12×TURN_LEFT 采集全景
                      │    ├─ [Step 2] agent.navigate_or_backtrack()  → LA 决策方向
                      │    ├─ [Step 3] agent.query_llm()             → VA 返回 bbox
                      │    ├─ [Step 3] policy._get_action()          → FMM 局部规划
                      │    ├─ [Step 4] envs.step()                   → 执行动作
                      │    ├─ [Step 4] mapping_module.update_map()   → 更新地图
                      │    └─ [Step 4] _process_map()                → 提取可通行区域
                      └─ _calculate_metric()  → 计算 SPL/NDTW/Success
```

---

## 第 1 层：Shell 脚本

**文件**: `eval_scripts/vlnce_r2r.sh`

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export BERT_LOCAL_PATH=data/grounded_sam/bert-base-uncased
export TOKENIZERS_PARALLELISM=false
# ... 更多环境变量 ...

python run_mp.py \
    --exp-name ${TIMESTAMP} \
    --run-type eval \
    --exp-config vlnce_baselines/config/r2r.yaml \
    --nprocesses ${NPROC:-20} \
    --use-navdp \
    --episode-file data/datasets/stratified_samples/vlnce_r2r.json \
    NUM_ENVIRONMENTS 1 \
    TRAINER_NAME ZS-Evaluator-mp \
    TORCH_GPU_IDS [0] \
    SIMULATOR_GPU_IDS [0]
```

传递的参数:
- `--exp-config r2r.yaml` — 实验配置
- `--nprocesses 20` — 20 个并行 worker
- `--use-navdp` — 启用 NavDP（楼梯导航）
- `--episode-file` — 100 集的 JSON ID 列表（分层抽样）
- `TRAINER_NAME ZS-Evaluator-mp` — 注册的 trainer 名称

---

## 第 2 层：`run_mp.py` — 多进程编排

**文件**: `sim-code/habitat/run_mp.py`  
**入口函数**: `run_exp()` → `worker()` / `worker_dynamic()`

### 2.1 配置初始化

```python
config = get_config(exp_config, opts)  # 合并 r2r.yaml + 命令行覆盖
```

`get_config()` 在 `vlnce_baselines/config/default.py` 中定义:
1. 从 `habitat_baselines.config.default._C` 创建基础配置
2. 合并本地默认值 `_C`（MAP、EVAL 配置段）
3. 加载用户指定的 YAML 文件 (`r2r.yaml`)
4. 应用命令行覆盖（如 `NUM_ENVIRONMENTS 1`）

`r2r.yaml` 中关键配置:
- `BASE_TASK_CONFIG_PATH: habitat_extensions/config/r2r.yaml` — 底层 Habitat 任务配置
- `TASK_TYPE: VLN`
- `MAP.MAP_RESOLUTION: 5`（5cm/格）
- `MAP.MAP_SIZE_CM: 2400`（24m×24m 地图）
- `EVAL.USE_TODO_LIST: True`（默认）

### 2.2 Episode 分发

```python
episode_info_list = get_episode_ids_from_config(config)  # 从数据集 JSON 读取全部 episode
# 若指定 --episode-file，过滤到该 JSON 中的 ID 集合
# 分配到 N 个 worker
split_episode_infos = [episode_info_list[i::nprocesses] for i in range(nprocesses)]
```

每个 worker 得到一个 episode 子集，写入 `data/logs/running_log/worker_{rank}_assignments_{exp_name}.json`。

### 2.3 Worker 启动

```python
# 静态分配模式（DYNAMIC_QUEUE=False）
for cfg in configs:
    p = mp.Process(target=worker, args=(cfg,))
    p.start()

# 动态队列模式（DYNAMIC_QUEUE=True，默认）
# 所有 episode 放入共享 Queue，worker 按需弹出
for cfg in configs:
    p = mp.Process(target=worker_dynamic, args=(cfg, ep_queue))
    p.start()
```

每个 worker 调用 `baseline_registry.get_trainer("ZS-Evaluator-mp")`（即 `ZeroShotVlnEvaluatorMP`），然后调用 `trainer.eval()` 或 `trainer.eval_dynamic(ep_queue)`。

---

## 第 3 层：`ZeroShotVlnEvaluatorMP` — Trainer 初始化

**文件**: `sim-code/habitat/vlnce_baselines/ZS_Evaluator_mp.py`  
**类**: `ZeroShotVlnEvaluatorMP` (注册名 `"ZS-Evaluator-mp"`)

### 3.1 `__init__` 中创建的组件

```python
class ZeroShotVlnEvaluatorMP(BaseTrainer):
    def __init__(self, config, r2r):
        # 1. 视觉器
        self.visualizer = LaViRAVisualizer(...)

        # 2. LA/VA 推理代理
        self.agent = VLMReasoningAgent(
            visualizer, task_type="VLN",
            use_guideline=True, use_working_memory=False,
            use_todo_list=True, backtrack_second_chance=True,
        )
        # VLMReasoningAgent 内部创建 LaViRA_API（双 OpenAI 客户端）

        # 3. NavDP（若 --use-navdp）
        self.navdp_agent = NavDP_Agent(...)  # 加载 navdp-cross-modal.ckpt

        # 4. 标记
        self.use_fmm = True       # FMM 是默认局部规划器
        self.use_iplanner = False # iPlanner 已移除
```

### 3.2 `eval()` — 静态分配模式

```python
def eval(self):
    self._init_envs()          # → construct_envs() 构建 Habitat 环境
    self._collect_val_traj()   # 加载 GT 轨迹 JSON（NDTW 计算用）
    self._initialize_policy()  # → 创建 GroundedSAM + Semantic_Mapping + FusionMapPolicy
    self.agent.reset()         # 重置代理状态
    self.agent.model.reset_stats()  # 重置 token 计数

    for i in range(eps_to_eval):
        self.rollout()   # ★ 核心循环
        self.reset()     # 清空状态（地图、目标、检测）

    self.envs.close()
    # 保存统计: stats_ep_ckpt_*.json, model_usage_stats_*.json, nav_stats_*.json
```

### 3.3 `eval_dynamic()` — 动态队列模式（默认）

与 `eval()` 的核心区别：
- 仅做一次 `_initialize_policy()` 和 `_collect_val_traj()`
- 循环从 `ep_queue` 弹出单个 episode
- 每个 episode 重新 `_init_envs()`（因为 `EPISODES_ALLOWED` 变了）
- 重用 GroundedSAM、Semantic_Mapping、FusionMapPolicy

---

## 第 4 层：`rollout()` — 单 Episode 主循环

**位置**: `ZS_Evaluator_mp.py` 第 863 行  
**最大步数**: 500（`ENVIRONMENT.MAX_EPISODE_STEPS`）  
**核心概念**: 4 态循环——无目标全景采集 → LA 决策 → VA 目标标注 → 局部导航

### 4.1 初始化

```python
def rollout(self):
    obs, full_pose = self._maps_initialization()  # envs.reset() + 首次建图
    # ...
    step = 0
    while step < self.max_step:  # 主循环
```

`_maps_initialization()` 内部:
1. `self.envs.reset()` → 获得初始观测
2. 提取指令文本（VLN: `obs[0]['instruction']['text']`）
3. `self.mapping_module.init_map_and_pose()`
4. `self.mapping_module(batch_obs, poses, 0)` → 首次地图更新

### 4.2 主循环四态

```
                    +------------------+
                    |                  |
                    v                  |
   [Step 1] 无目标 + 无全景            |
       |                               |
       | get_panorama() (12 步旋转)     |
       v                               |
   [Step 2] 有全景 + 无行动队列         |
       |                               |
       | agent.navigate_or_backtrack() |  (LA 决策: 方向/回溯/STOP)
       v                               |
   [Step 3] 有全景 + 有方向 + 无bbox    |
       |                               |
       | agent.query_llm()             |  (VA 标注: 边界框 → 世界坐标)
       | policy._get_action()          |  (FMM 局部规划: 短期目标)
       v                               |
   [Step 4] 有目标 + 有行动队列         |
       |                               |
       | envs.step()                   |  (执行动作)
       | mapping_module.update_map()   |  (更新语义地图)
       | _process_map()                |  (可通行区域 + 前沿)
       |                               |
       +--- 到达目标/超时 → 返回 Step 1 -+
```

### 4.2.1 Step 1: 全景采集

```python
# 当 target_map_x is None 且 not panorama_got 且 not navigate_or_not
panorama_frames = self.get_panorama(obs[0], step)
# get_panorama() 内: for 12 步 TURN_LEFT → 采集 4 帧 (0°, 90°, 180°, 270°)
step += 12  # 旋转消耗 12 步
```

每步旋转后也会更新地图:
```python
self.mapping_module(batch_obs, poses, self.current_step)
full_map, full_pose, _ = self.mapping_module.update_map(...)
```

### 4.2.2 Step 2: LA 决策

```python
# 当 target_map_x is None 且 navigate_or_not is False
decision = self.agent.navigate_or_backtrack(
    instruction, visited_targets, feedback, episode_id, step, history_images
)
# 返回值: {'action': 'NAVIGATE'|'BACKTRACK'|'STOP',
#           'direction': 'forward'|'left'|'right'|'behind',
#           'stop_signal': bool, 'waypoint': int,
#           'progress_analysis': str, 'reasoning': str}
```

**`VLMReasoningAgent.navigate_or_backtrack()`** (`agent.py:238`):
1. 构建 prompt: `_build_nav_prompt(...)` → 包含 TODO 列表、负面约束、已访问路径点
2. 将 4 视角全景图编码为 base64，放入消息列表
3. 加入连续历史图像（`history_images`，每 `HISTORY_INTERVAL=2` 步一帧）
4. 调用 `self.model.generate(messages, use_la=True, ...)` → **LA 模型 API 调用**
5. 解析返回的 JSON 得到 action 和 direction
6. 若 action == 'NAVIGATE' → 后续做 TODO 列表一致性验证（独立 LLM 调用）

如果 action == 'NAVIGATE':
```python
# 将方向转为旋转序列
if direction == 'left':    action_list.extend([TURN_LEFT] * 3)   # 90°
elif direction == 'right': action_list.extend([TURN_RIGHT] * 3)
elif direction == 'behind':action_list.extend([TURN_LEFT] * 6)   # 180°
# forward 无需旋转
panorama_got = True
navigate_or_not = True
```

如果 action == 'BACKTRACK':
```python
# 找回溯目标路径点的世界坐标
# 若 backtrack_second_chance=True:
#   → replan_at_backtrack() 重新查询 LA
#   → query_llm() 重新查询 VA 获取新 bbox
#   → 反投影到新目标坐标
# 否则: 直接使用回溯路径点的世界坐标
target_map_x, target_map_y = ...
target_set_step = step
```

如果 action == 'STOP':
```python
current_la_action = 'STOP'
going_to_stop = True
# 下一步进入 double_check_stop 流程
```

### 4.2.3 STOP 二次确认

```python
# 当 LA 决定 STOP，采集新全景，确认:
if target_map_x is None and not panorama_got and going_to_stop:
    panorama_frames = self.get_panorama(obs[0], step)
    step += 12
    should_stop, stop_response = self.agent.double_check_stop(
        instruction, panorama_frames, visited_targets, ...
    )
```

**`VLMReasoningAgent.double_check_stop()`** (`agent.py:1134`):
1. 构建专门的 STOP 验证 prompt
2. 将新全景图发送给 LA 模型
3. 返回 `(should_stop: bool, response: dict)`
4. 若拒绝 → `self.consecutive_stop_failures += 1`，追加反馈文本
5. 若连续 3 次拒绝 → 强制 STOP
6. 若通过 → `action_list.append(STOP)`，退出 episode

### 4.2.4 Step 3: VA 目标标注 + 局部规划

```python
# 当 target_map_x is None 且 panorama_got 且 action_list 为空
bbox = self.agent.query_llm(
    instruction, visited_targets,
    rgb_image=obs[0]['rgb'],    # 当前正前方 RGB
    depth_image=obs[0]['depth'], # 当前正前方深度
    ...
)
```

**`VLMReasoningAgent.query_llm()`** (`agent.py:861`):
1. 构建 VA prompt（包含指令、历史路径点、进度分析、楼梯标记）
2. 将当前 RGB 图像编码为 base64
3. 调用 `self.model.generate(messages, use_la=False, ...)` → **VA 模型 API 调用**
4. 解析 JSON 返回 `{x1, y1, x2, y2, target, reasoning, progress, ...}`
5. 设置 `self.stair = (target == 'stairs')`

VA 返回边界框后，反投影为世界坐标:
```python
# 将 bbox 中心像素 → 世界坐标
target = get_world_xz_from_pixel(
    pixel_coords=(center_x, bottom_y),
    depth_image=depth_image,
    full_pose=current_pose,
    camera_intrinsics=...
)
target_map_x = int(target[0] * 100.0 / resolution)
target_map_y = int(target[1] * 100.0 / resolution)
```

然后选择局部规划器并产生第一个动作:

```python
if self.agent.stair and (use_navdp or use_iplanner):
    # 楼梯 → 使用 NavDP/iPlanner
    action_list.append(navdp/iplaner_action)
elif self.use_fmm:
    # 正常 → 使用 FMM
    navigation_action = self.policy._get_action(
        current_pose, waypoint, full_map, traversible, collision_map, ...
    )
    action_list.append(navigation_action)
```

**`FusionMapPolicy._get_action()`** (`models/Policy.py:46`):
1. 调用 `self.policy.forward()` → `SuperPixelPolicy` 评估价值地图上的超像素
2. 若 `search_destination` → `_search_destination()` 评估是否发现目标物体
3. 创建 `FMMPlanner`，设置 `set_goal(goal)`
4. `planner.get_short_term_goal(position)` → 计算短期目标
5. `angle_and_direction(heading_vector, waypoint_vector)` → 转为 Habitat 动作

### 4.2.5 Step 4: 动作执行与地图更新

```python
# 当 action_list 非空
self._action = action_list.pop(0)
outputs = self.envs.step([{"action": self._action}])
step += 1
obs, _, dones, infos = [list(x) for x in zip(*outputs)]

# 更新地图
batch_obs = self._batch_obs(obs)
self.mapping_module(batch_obs, poses, current_step)
full_map, full_pose, one_step_full_map = \
    self.mapping_module.update_map(step, detected_classes, episode_id)

# 处理地图 → 提取可通行区域
self.traversable, self.floor, self.frontiers = self._process_map(step, full_map[0])
```

**`_batch_obs()` 内部调用链**:
1. `_concat_obs(obs)` → 拼接 RGB + Depth
2. `_preprocess_state(state)`:
   - `self._get_sem_pred(rgb)` → **`GroundedSAM.segment()`** (GroundingDINO 检测 + SAM 分割)
   - `self._preprocess_depth()` → 深度缩放和无效值处理
   - 下采样 4× (640→160, 480→120)
3. 各观测通道补齐后堆叠为 batch tensor

**`Semantic_Mapping.update_map()` ** (`map/mapping.py`):
- 将当前帧语义分割投影到自顶向下地图
- 累积更新 `full_map` (480×480, 通道: [obstacle, explored, class1, class2, ...])
- 更新 agent 位姿

**`_process_map()`**:
- 分离障碍物/已探索/物体/可通行区域
- 形态学闭运算
- 提取前沿 (frontiers): 已探索边界

### 4.2.6 目标到达检测

```python
# 在下一轮循环开头检测:
if target_map_x is not None:  # 有活跃目标
    distance_to_target = sqrt((tx - ax)² + (ty - ay)²)
    if distance_to_target < target_reached_threshold:  # 默认 0.75m
        # 重置目标状态 → 返回 Step 1
        target_map_x, target_map_y = None, None
        panorama_got = False
        navigate_or_not = False
```

目标超时检测（15 步）:
```python
if steps_since_target_set >= max_steps_to_target:
    # 放弃当前目标，返回 Step 1
    target_map_x, target_map_y = None, None
```

### 4.2.7 Episode 终止

- `dones[0] == True` → `_calculate_metric()` → exit
- `step >= max_step` (500) → `_calculate_metric(is_timeout=True)` → exit
- EQA: STOP 后 Oracle-QA → `_eqa_done_requested` → exit

---

## 第 5 层：LLM API 调用

**文件**: `vlnce_baselines/utils/api.py`  
**类**: `LaViRA_API`

```python
class LaViRA_API:
    def __init__(self):
        self.la_client = OpenAI(api_key=LA_API_KEY, base_url=LA_BASE_URL)
        self.va_client = OpenAI(api_key=VA_API_KEY, base_url=VA_BASE_URL)

    def generate(self, messages, use_la=False, ...):
        if use_la:
            response = self.la_client.chat.completions.create(
                model=LA_MODEL_NAME,         # e.g. "gemini-3.5-flash"
                messages=messages,
                max_completion_tokens=4096,
                timeout=120,
            )
        else:
            response = self.va_client.chat.completions.create(
                model=VA_MODEL_NAME,         # e.g. "qwen3.5-27b"
                messages=messages,
                max_tokens=1024,
                timeout=120,
                reasoning_effort='low',      # VA 不需要深度推理
            )
        # 追踪 token 使用量
        self.stats[model]['calls'] += 1
        self.stats[model]['input_tokens'] += usage.prompt_tokens
        self.stats[model]['output_tokens'] += usage.completion_tokens
```

### LA 调用时机（每次 episode 约 3-10 次）
| 调用 | 方法 | 目的 |
|------|------|------|
| LA Prompt (导航) | `navigate_or_backtrack()` | 从全景图决定方向 |
| LA Prompt (回溯重规划) | `replan_at_backtrack()` | 回溯后重新选择方向 |
| LA Prompt (STOP 确认) | `double_check_stop()` | 二次确认是否该停止 |
| LA Prompt (Last Object) | `_ensure_last_object()` | 提取 VLN 指令中的最终目标 |
| LA Prompt (TODO 验证) | TODO consistency check | 验证 TODO 列表一致性 |

### VA 调用时机（每个路径点约 1 次）
| 调用 | 方法 | 目的 |
|------|------|------|
| VA Prompt | `query_llm()` | 从 RGB 图像返回下一路径点的 bbox |
| VA Prompt (回溯) | `replan_at_backtrack()` 内 | 回溯后重新标注 bbox |

---

## 第 6 层：结果统计

### Episode 级别 (`_calculate_metric`)
```python
# 计算 NDTW (Normalized Dynamic Time Warping)
dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
metric['ndtw'] = exp(-dtw_distance / (len(gt_path) * 3.0))
metric['success'] = 1.0 if distance_to_goal <= 3.0 else 0.0
metric['spl'] = success * gt_length / max(gt_length, path_length)
metric['sdtw'] = ndtw * success
```

### 全局汇总 (`run_mp.py` 主进程)
```python
# 合并所有 worker 的 stats_ep_ckpt_*.json
# 计算平均 Success, SPL, NDTW, SDTW
# 合并 model_usage_stats (LA/VA 总 token 消耗)
# 合并 nav_stats (总回溯次数、总路径点数)
```

---

## 完整调用图

```
vlnce_r2r.sh
│
├─ export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 BERT_LOCAL_PATH=... NPROC=20
│
└─ python run_mp.py --exp-config r2r.yaml ... TRAINER_NAME ZS-Evaluator-mp
   │
   ├─ get_config(r2r.yaml)  ─── 合并 default.py + r2r.yaml + CLI overrides
   │
   ├─ get_episode_ids_from_config()  ─── 读取 data/datasets/episodes/vlnce_r2r.json.gz
   │   └─ episode_file 过滤  ─── data/datasets/stratified_samples/vlnce_r2r.json (100 IDs)
   │
   ├─ split → 20 workers
   │
   └─ for each worker (mp.Process):
       │
       ├─ baseline_registry.get_trainer("ZS-Evaluator-mp")
       │   └─ ZeroShotVlnEvaluatorMP.__init__()
       │       ├─ LaViRAVisualizer()
       │       ├─ VLMReasoningAgent()
       │       │   └─ LaViRA_API(la_client, va_client)
       │       └─ NavDP_Agent()  [--use-navdp]
       │
       ├─ trainer.eval() / eval_dynamic()
       │   │
       │   ├─ _init_envs()  ─── construct_envs() → Habitat Env
       │   ├─ _collect_val_traj()  ─── 加载 GT 轨迹
       │   ├─ _initialize_policy()
       │   │   ├─ GroundedSAM(config, device)
       │   │   │   ├─ GroundingDINO (开放词汇检测)
       │   │   │   └─ SAM / RepViTSAM (分割)
       │   │   ├─ Semantic_Mapping(config.MAP)
       │   │   └─ FusionMapPolicy(config)
       │   │       └─ SuperPixelPolicy(config)
       │   │
       │   └─ for each episode:
       │       ├─ rollout()
       │       │   │
       │       │   ├─ _maps_initialization()
       │       │   │   ├─ envs.reset()
       │       │   │   ├─ mapping_module.init_map_and_pose()
       │       │   │   └─ mapping_module(batch_obs, poses, 0)
       │       │   │
       │       │   └─ while step < 500:
       │       │       │
       │       │       ├─ [无目标, 无全景]
       │       │       │   └─ get_panorama(obs, step)
       │       │       │       ├─ for 12 turns: envs.step(TURN_LEFT)
       │       │       │       ├─ 每步: mapping_module(batch_obs, poses)
       │       │       │       └─ 每步: mapping_module.update_map()
       │       │       │
       │       │       ├─ [有全景, 无决策]
       │       │       │   └─ agent.navigate_or_backtrack()
       │       │       │       ├─ _build_nav_prompt()  ─── prompt 模板 + TODO
       │       │       │       ├─ _get_initial_views()  ─── 4 视角 base64
       │       │       │       ├─ model.generate(use_la=True)  ★ LA API
       │       │       │       └─ TODO consistency check  ★ LA API
       │       │       │
       │       │       ├─ [STOP 信号]
       │       │       │   ├─ get_panorama()  ─── 重新采集全景
       │       │       │   └─ agent.double_check_stop()  ★ LA API
       │       │       │
       │       │       ├─ [BACKTRACK]
       │       │       │   ├─ agent.replan_at_backtrack()  ★ LA API
       │       │       │   └─ agent.query_llm()  ★ VA API (新 bbox)
       │       │       │
       │       │       ├─ [有方向, 无目标]
       │       │       │   ├─ agent.query_llm()  ★ VA API
       │       │       │   │   └─ model.generate(use_la=False)
       │       │       │   ├─ get_world_xz_from_pixel()  ─── bbox 中心 → 世界坐标
       │       │       │   └─ policy._get_action()  ─── FMM 局部规划
       │       │       │       ├─ SuperPixelPolicy.forward()  ─── 超像素价值评估
       │       │       │       └─ FMMPlanner.get_short_term_goal()  ─── 短期目标
       │       │       │
       │       │       ├─ [有目标, 无动作] (Step 3 后续)
       │       │       │   └─ policy._get_action()  ─── FMM 持续导航
       │       │       │
       │       │       ├─ [动作执行]
       │       │       │   ├─ envs.step([{action}])  ─── Habitat 仿真步进
       │       │       │   ├─ _concat_obs() → _preprocess_state()
       │       │       │   │   └─ GroundedSAM.segment(rgb)  ─── 语义分割
       │       │       │   ├─ mapping_module(batch_obs, poses)
       │       │       │   ├─ mapping_module.update_map()
       │       │       │   └─ _process_map()  ─── 可通行区域/前沿
       │       │       │
       │       │       └─ [目标到达/超时] → 重置状态, 回到 Step 1
       │       │
       │       ├─ reset()  ─── 清空 episode 状态
       │       └─ _calculate_metric()  ─── NDTW/SPL/Success
       │
       └─ 保存统计文件:
           ├─ stats_ep_ckpt_{split}_r{rank}_w{world}.json
           ├─ model_usage_stats_{split}_r{rank}_w{world}.json
           └─ nav_stats_{split}_r{rank}_w{world}.json

[主进程]
└─ 合并所有 worker 统计 → 打印最终 Success/SPL/NDTW/Token 用量
```

---

## 关键文件索引

| 文件 | 作用 |
|------|------|
| `eval_scripts/vlnce_r2r.sh` | Shell 入口 |
| `run_mp.py` | 多进程编排, worker 分发 |
| `vlnce_baselines/config/default.py` | 默认配置 + `get_config()` |
| `vlnce_baselines/config/r2r.yaml` | VLN R2R 任务配置 |
| `habitat_extensions/config/r2r.yaml` | Habitat 底层环境配置 |
| `vlnce_baselines/ZS_Evaluator_mp.py` | `ZeroShotVlnEvaluatorMP`: rollout 循环, 地图处理 |
| `vlnce_baselines/agent.py` | `VLMReasoningAgent`: LA/VA prompt 与决策 |
| `vlnce_baselines/utils/api.py` | `LaViRA_API`: 双端点 OpenAI 客户端 |
| `vlnce_baselines/models/Policy.py` | `FusionMapPolicy`: FMM 局部规划 |
| `vlnce_baselines/models/fmm_planner.py` | `FMMPlanner`: 快速行进法 |
| `vlnce_baselines/map/semantic_prediction.py` | `GroundedSAM`: 语义分割 |
| `vlnce_baselines/map/mapping.py` | `Semantic_Mapping`: 自顶向下地图 |
| `vlnce_baselines/prompts/prompts_vln.py` | VLN 任务 prompt 模板 |
