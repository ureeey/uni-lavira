# LaViRA Pipeline 完整分析

> 从 `run_mp.py` 入口到 HTTP 请求发出, 含 action_list、target_map_x/y、FMM、坐标系说明

## 整体架构

```
eval_scripts/hm3d_ovon.sh
  └─ python run_mp.py --exp-config ... --episode-file ...
       └─ run_exp()                                      run_mp.py:100
            ├─ 读取 episode 列表、按 worker 分配
            └─ mp.Process(target=worker_dynamic, ...)     run_mp.py:298
                 └─ ZeroShotVlnEvaluatorMP.eval_dynamic()
                      └─ rollout()                        ZS_Evaluator_mp.py:880
                           │
                           ├─ [LOOP] while step < 500     ZS_Evaluator_mp.py:945
                           │   │
                           │   ├─ 1) 保存历史图            ZS_Evaluator_mp.py:992-1002
                           │   │   每 history_interval 步存一张 640×480 RGB
                           │   │   存入 self.history_images，永不截断
                           │   │
                           │   ├─ 2) 全景采集              ZS_Evaluator_mp.py:727-758
                           │   │   转12次×30°，采样4方向(0°/90°/180°/270°)
                           │   │   step += 12
                           │   │
                           │   ├─ 3) 创建 waypoint         ZS_Evaluator_mp.py:1124-1133
                           │   │   visited_targets.append({panorama, image, ...})
                           │   │
                           │   ├─ 4) ★ LA 决策             ZS_Evaluator_mp.py:1145-1152
                           │   │   agent.navigate_or_backtrack(...)
                           │   │   → 见下方详细展开
                           │   │
                           │   ├─ 5) VA 获取 bbox          agent.query_llm()
                           │   │   朝 LA 决定的方向看 → VA 返回 bbox
                           │   │
                           │   ├─ 6) 局部导航              FMM/NavDP
                           │   │   到达或超时(15步) → 重置 → 回步骤2
                           │   │
                           │   └─ 7) STOP 双重确认        ZS_Evaluator_mp.py:1065-1111
                           │       重新全景 → LA 再确认，最多拒绝3次
                           │
                           └─ episode 结束 → _calculate_metric()
```

---

## 阶段 1: 入口 — run_mp.py

**文件:** `run_mp.py`

| 步骤 | 行号 | 说明 |
|------|------|------|
| argparse 解析 | 477-550 | `--exp-config`, `--nprocesses`, `--episode-file`, `--use-navdp`, `--no-fmm` |
| `get_episode_ids_from_config()` | 39-61 | 从 `data/datasets/episodes/hm3d_ovon.json.gz` 读取全部 3000 个 episodes |
| 过滤 episode 列表 | 157-161 | `--episode-file` 指定的 JSON 文件匹配出 100 个 ID |
| 构建 worker configs | 240-256 | `DYNAMIC_QUEUE=True`: 构建 N 个 worker 配置，共享 `ep_queue` |
| 启动多进程 | 291-305 | `mp.Process(target=worker_dynamic, args=(cfg, ep_queue))` |
| `worker_dynamic()` | 441-475 | 初始化 trainer → `trainer.eval_dynamic(ep_queue)` |
| 等待完成 + 汇总 | 310-404 | 30s 心跳检查，合并 stats，输出最终指标 |

---

## 阶段 2: Evaluator 初始化 — ZS_Evaluator_mp.py

**文件:** `vlnce_baselines/ZS_Evaluator_mp.py`

### `__init__()` — 行 73-216

```
ZeroShotVlnEvaluatorMP.__init__(config)
  │
  ├─ 语义模型: GroundedSAM (GroundingDINO + RepViTSAM)   # 行 39 import
  ├─ 地图: Semantic_Mapping (480×480, 5cm/pixel)         # 行 35 import
  ├─ 规划器: FusionMapPolicy (FMM)                        # 行 36 import
  ├─ 局部导航: NavDP_Agent (可选)                          # 行 191-207
  ├─ 可视化: LaViRAVisualizer                              # 行 114-115
  │
  └─ ★ VLMReasoningAgent                                  # 行 151-161
       ├─ task_type = config.TASK_TYPE (e.g., "OBJNAV")
       ├─ use_guideline = True
       ├─ use_working_memory = False
       ├─ use_todo_list = True
       └─ backtrack_second_chance = True
```

### 关键配置项

| 配置项 | 变量 | 行号 | 来源 | 默认值 |
|--------|------|------|------|--------|
| 连续历史开关 | `self.use_continuous_history` | 130 | `config.EVAL.USE_CONTINUOUS_HISTORY` | `True` |
| 历史采样间隔 | `self.history_interval` | 131 | `config.EVAL.HISTORY_INTERVAL` | 2 |
| TODO list | `self.use_todo_list` | 149 | `config.EVAL.USE_TODO_LIST` | `True` |
| Working Memory | `self.agent(use_working_memory=)` | 155 | 写死 | `False` |
| RGB 分辨率 W×H | `self.width` / `self.height` | 83-84 | `config.TASK_CONFIG.SIMULATOR.RGB_SENSOR` | **640×480** |

### `reset()` — 行 688-708

每个 episode 开始时重置:
```python
self.visited_targets = []
self.history_images = []    # 清空历史图像
self.current_step = 0
self.policy.reset()
self.mapping_module.reset()
self.agent.reset()
```

---

## 阶段 3: Episode 主循环 — rollout()

**文件:** `vlnce_baselines/ZS_Evaluator_mp.py`  
**函数:** `rollout()` — 行 880

### 状态变量

```
step: 0 → 全景后 +12 → 转向/导航逐步 +1 → ...
history_images: [{step: 0, image: PIL.Image}, {step: 2, image}, {step: 4, ...}, ...]
visited_targets: [{step, init_image, panorama_frames[4], world_coords(x,y), full_pose}, ...]
target_map_x / target_map_y: 当前导航目标在 map 上的坐标
action_list: [action, ...] — 待执行的低层动作队列
```

### 循环体 (while step < 500, 行 945)

#### 步骤 1: 保存连续历史图像 (行 992-1002)

```python
if self.use_continuous_history and step % self.history_interval == 0:
    rgb_to_save = obs[0]['rgb'].copy()        # 640×480 uint8
    img_save = Image.fromarray(rgb_to_save)
    self.history_images.append({'step': step, 'image': img_save})
    # ⚠ 只 append，从不删除 → 无界增长
```

#### 步骤 2: 全景采集 (行 1113-1118)

```python
panorama_frames = self.get_panorama(obs[0], step)   # 行 727
step += 12
```

`get_panorama()` (行 727-758) 内部:
- 12 次 `TURN_LEFT` (每次 30°)
- 返回每 3 帧采 1 帧 → 4 张图 (0°, 90°, 180°, 270°)

#### 步骤 3: 创建 waypoint (行 1124-1133)

```python
self.visited_targets.append({
    'step': step,
    'init_image': Image.fromarray(current_rgb),
    'panorama_frames': panorama_frames,   # 4 张全景图
    'world_coords': (agent_map_x, agent_map_y),
    'full_pose': current_pose
})
self.total_waypoints += 1
```

#### 步骤 4: ★ LA 决策 (行 1145-1152)

```python
decision = self.agent.navigate_or_backtrack(
    instruction=self.instruction,
    visited_targets=self.visited_targets,
    feedback=combined_feedback,
    episode_id=self.current_episode_id,
    step=step,
    history_images=self.history_images if self.use_continuous_history else None
    #                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                                 传入全部累积的历史图像!
)
```

#### 步骤 5-7: VA bbox → 局部导航 → STOP 确认

LA 返回方向后，调 `agent.query_llm()` 让 VA 输出 bbox → back-project 到世界坐标 → FMM/NavDP 驱动 → 到达或超时后回到步骤 2。

STOP 时触发 double-check (行 1065-1111): 重新全景 → 再问 LA → 最多拒绝3次后强制停止。

---

## 阶段 4: ★ LA 请求构建 — agent.py

**文件:** `vlnce_baselines/agent.py`  
**函数:** `navigate_or_backtrack()` — 行 238

### 4.1 构建 history_content (行 247-342)

**连续历史模式** (`history_images is not None`):

```python
for item in history_images:           # 遍历 ALL 累积图像
    img = item['image']               # PIL.Image, 原始 640×480
    history_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}
        #                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                                        行 158-162: PNG 编码, 无 resize
    })
```

中间插入 segment header (e.g., `"Waypoint 0 -> Waypoint 1: "`) 标记图像属于哪个导航段。

**回退模式** (`history_images is None`, 行 328-342):

```python
for i, target in enumerate(visited_targets[:-1]):   # 排除当前路点
    # 每个历史 waypoint 最多 2 张图: init_image + dir_image
    history_content.append(base64(target['init_image']))
    history_content.append(base64(target['dir_image']))
```

### 4.2 构建 current_views (行 344-374)

```python
view_definitions = [
    {'angle': 0, 'name': 'forward',   'label': 'Current FORWARD view:'},
    {'angle': 90, 'name': 'left',     'label': 'Current LEFT view:'},
    {'angle': 180, 'name': 'behind',  'label': 'Current BEHIND view:'},
    {'angle': 270, 'name': 'right',   'label': 'Current RIGHT view:'}
]
# 从 panorama_frames[4] 中按 angle→index 取对应帧
for view in view_definitions:
    frame_idx = int((angle / 360.0) * num_frames) % num_frames
    img = panorama_frames[frame_idx]['rgb']
    current_views.append({"type": "image_url", "url": f"data:image/png;base64,{img_to_base64(img)}"})
    # → 4 张当前全景 base64 PNG
```

### 4.3 可选: Working Memory (行 388-415)

当 `use_working_memory=True` 且 `len(visited_targets) >= 2` 时，额外加入上一 waypoint 的 4 张全景图。

**当前配置:** `use_working_memory=False` → **此段不执行**。

### 4.4 构建文本 prompt (行 418-543)

```python
content = [{"type": "text", "text": f"Navigation Task: \"{instruction}\"\n\nNavigation History:"}]
content.extend(history_content)                          # N 张历史图
content.append({"type": "text", "text": "Current 4-directional views:"})
content.extend(current_views)                            # 4 张当前全景
content.append({"type": "text", "text": prompt})         # 决策模板文本

prompt = _build_nav_prompt(...)                          # 行 142-156
# 使用模板 LA_PROMPT_BACKTRACK 或 LA_PROMPT_NO_BACKTRACK
# 包含: TODO list, 可用动作列表, feedback, negative_constraints
```

### 4.5 `img_to_base64()` — 行 158-162

```python
def img_to_base64(self, img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")     # ⚠ PNG 无损编码, 无 resize
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
```

对比 `api.py` 中 VA 路径的 `image_to_base64()` (行 169-176) 使用 **JPEG** 格式，同样的 640×480 图 JPEG ~50KB vs PNG ~350KB。

### 4.6 最终组装 + 调用 (行 547-564)

```python
messages = [{"role": "user", "content": content}]

output_text = self.model.generate(
    messages=messages,
    max_new_tokens=8192,
    temperature=0.7,
    use_la=True,         # → 使用 LA endpoint
    log_path=log_path,
)
```

---

## 阶段 5: HTTP 请求发出 — api.py

**文件:** `vlnce_baselines/utils/api.py`

### `LaViRA_API.__init__()` — 行 126-163

```python
self.la_client = OpenAI(
    api_key=la_api_key,         # 环境变量 LA_API_KEY
    base_url=la_base_url,       # 环境变量 LA_BASE_URL
    timeout=2000,
    http_client=_build_http_client(),  # httpx, 强制 IPv4
)
self.la_model_name = la_model_name    # 环境变量 LA_MODEL_NAME
```

### `LaViRA_API.generate()` — 行 231-359

```python
def generate(self, messages, max_new_tokens=1024, temperature=0.7, use_la=False, ...):
    # 1. 选择客户端
    if use_la:
        client = self.la_client
        model_name = self.la_model_name      # e.g., "qwen3.6-plus"

    # 2. 计算 payload 大小
    _msg_str = json.dumps(messages, ensure_ascii=False)
    _payload_kb = len(_msg_str.encode('utf-8')) / 1024
    # → log: "▐ LA #11 → qwen3.6-plus  22297 KB"

    # 3. ★ 发送 HTTP POST
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,                   # ← 22MB JSON body
        max_completion_tokens=max_new_tokens, # 8192
        temperature=0.7,
        timeout=120,
        extra_body={'enable_thinking': False},
    )
    # └→ POST {base_url}/chat/completions
    #    Header: Authorization: Bearer {api_key}
    #    Body: {"model": "qwen3.6-plus", "messages": [...], "max_completion_tokens": 8192, ...}

    # 4. 错误重试 (行 334-359)
    except Exception as e:
        # 413 RequestTooLarge → 重试最多 5 次, 间隔 30s
        # 不可恢复错误 (data_inspection_failed, invalid_request_error) → 直接放弃

    # 5. 成功 → 解析返回
    content = response.choices[0].message.content
    return content
```

---

## Payload 增长分析

### 每 Waypoint 新增图像

```
全景采集: step 0→12, 保存历史图于 steps 0,2,4,6,8,10,12  = 7 张
导航阶段: step 13→~18, 保存历史图于 steps 14,16,18         = 3 张
────────────────────────────────────────────────────────
每 waypoint 新增 ~10 张 history_images
```

### 第 N 次 LA 调用的 payload 构成

| 组件 | 图像数 | 编码 | 单张大小 | 小计 |
|------|--------|------|----------|------|
| history_images | ~N×10 张 | PNG base64, 640×480 原图 | ~350 KB | **主导** |
| current_views | 4 张 | PNG base64, 640×480 | ~350 KB | ~1.4 MB |
| Working Memory | 0 (关闭) | — | — | 0 |
| 文本 prompt | — | UTF-8 | — | ~3 KB |

**第 11 次调用估算:** 110 张历史图 × 350KB + 4 张当前全景 × 350KB ≈ **40MB** (实际 ~22MB，因部分偶数步被 page jump 跳过)

### 根因总结

1. **`history_images` 无界累积** — `ZS_Evaluator_mp.py:1002` 只 append 不 trim
2. **PNG 无损编码** — `agent.py:160` 用 PNG 而非 JPEG，图片 3-5 倍于必要大小
3. **全量重发** — `agent.py:290` 遍历全部历史图像，无滑动窗口
4. **无 resize** — 640×480 原图直接编码，即使模型会将大图 downsample

---

## 关键代码位置速查

| 作用 | 文件 | 行号 |
|------|------|------|
| 入口 main | `run_mp.py` | 477 |
| worker 入口 | `run_mp.py` | 441 |
| Evaluator `__init__` | `ZS_Evaluator_mp.py` | 73 |
| Evaluator `reset` | `ZS_Evaluator_mp.py` | 688 |
| 全景采集 `get_panorama` | `ZS_Evaluator_mp.py` | 727 |
| 主循环 `rollout` | `ZS_Evaluator_mp.py` | 880 |
| 保存历史图 | `ZS_Evaluator_mp.py` | 992-1002 |
| 创建 waypoint | `ZS_Evaluator_mp.py` | 1124-1133 |
| ★ 调用 LA 决策 | `ZS_Evaluator_mp.py` | 1145-1152 |
| STOP 双重确认 | `ZS_Evaluator_mp.py` | 1065-1111 |
| Backtrack + Replan | `ZS_Evaluator_mp.py` | 1167-1279 |
| Agent `__init__` | `agent.py` | 24 |
| `img_to_base64` (PNG) | `agent.py` | 158 |
| ★ `navigate_or_backtrack` | `agent.py` | 238 |
| 构建 history_content | `agent.py` | 247-342 |
| 构建 current_views | `agent.py` | 344-374 |
| 构建文本 prompt | `agent.py` | 418-543 |
| messages 组装 + 调用 | `agent.py` | 547-564 |
| `_build_nav_prompt` | `agent.py` | 142 |
| API `__init__` (OpenAI client) | `api.py` | 126 |
| ★ `generate` (HTTP POST) | `api.py` | 231 |
| payload 大小计算 | `api.py` | 263-264 |
| `client.chat.completions.create` | `api.py` | 287 |
| 错误重试逻辑 | `api.py` | 334-359 |
| `image_to_base64` (JPEG, VA用) | `api.py` | 169 |

---

## 附录 A: `action_list` — 低层动作队列

`action_list` 是 `rollout()` 中的一个 **FIFO 低层动作队列** (`list[int]`)，存的是 Habitat 模拟器的原子动作 ID，在 LA/VA 高层决策和 env.step() 单步执行之间充当缓冲。

### 动作 ID 映射

```python
STOP = 0, MOVE_FORWARD = 1, TURN_LEFT = 2, TURN_RIGHT = 3
```

### 消费规则 (行 1996-2006)

每步只取队首一个动作执行：

```python
if action_list:
    self._action = action_list[0]       # 取队首
    action_list.pop(0)                  # 弹出
    actions = [{"action": self._action}]
    outputs = self.envs.step(actions)   # 执行 1 步
    step += 1
```

### 决策门 (行 1013)

**只有 `action_list` 为空时才做 LA/VA 高层决策:**

```python
if not action_list:
    # 全景采集 → LA 决策 → VA bbox → FMM 规划 → 往 action_list 塞动作
```

### 五种写入场景

#### 场景 1: STOP (行 1077, 1101)

```python
action_list.append(0)  # STOP → 下一步 episode 结束
```

#### 场景 2: LA 方向 → 转向序列 (行 1342-1348)

```python
if direction == 'left':
    action_list.extend([2] * 3)    # TURN_LEFT × 3 = 90°
elif direction == 'right':
    action_list.extend([3] * 3)    # TURN_RIGHT × 3 = 90°
elif direction == 'behind':
    action_list.extend([2] * 6)    # TURN_LEFT × 6 = 180°
# forward → 不需要转向, action_list 保持为空, 直接进入场景 3
```

**例子:** LA 返回 `{"action": "NAVIGATE", "direction": "right"}`  
→ `action_list = [3, 3, 3]`（下 3 步 TURN_RIGHT）

#### 场景 3: VA bbox → FMM 首次规划 (行 1651-1656)

```python
navigation_action = self.policy._get_action(
    current_pose, waypoint, full_map[0], self.traversable,
    self.collision_map, step, ...
)
action_list.append(navigation_action)  # 通常是 1 (MOVE_FORWARD)
```

#### 场景 4: 持续导航 → FMM/NavDP 每步规划 (行 1664+)

`target_map_x/y` 已有值、`action_list` 为空 → 每步重新规划，每次塞 1 个动作。

#### 场景 5: NavDP/iPlanner 轨迹动作 (行 1552, 1602, 1840, 1934)

替代 FMM 时，同样每次塞 1 个动作。

### 完整实例: 一个 waypoint 周期

```
Step N:    action_list=[]                → 全景采集, step+=12
Step N+12: action_list=[]                → LA: "navigate to right"
           action_list.extend([3,3,3])   → [3, 3, 3]

Step N+13: self._action=3, action_list→[3,3]   TURN_RIGHT (转了30°)
Step N+14: self._action=3, action_list→[3]     TURN_RIGHT (转了60°)
Step N+15: self._action=3, action_list→[]      TURN_RIGHT (转完90°)

Step N+16: action_list=[]                → VA bbox, FMM 首次规划
           action_list.append(1)         → [1]

Step N+17: self._action=1, action_list→[]  MOVE_FORWARD (前进了0.25m)
Step N+18: action_list=[]                → FMM 重新规划
           action_list.append(1)         → [1]
...重复直到 target_reached 或 timeout...

Step N+K:  dist < 15 pixels             → target 到达, 回全景采集
```

---

## 附录 B: `target_map_x/y` 与 FMM 协作

### target_map_x/y 的一生

#### 1. 诞生: VA bbox → 深度反投影 → 地图坐标

```
ZS_Evaluator_mp.py:1352-1403

VA 返回 bbox {x1,y1,x2,y2}
  │
  ├─ 取 bbox 中心像素: coords = ((x1+x2)/2, y2)              (行 1384)
  │
  ├─ get_world_xz_from_pixel(coords, depth, pose, intrinsics)
  │   返回世界坐标 target = [world_x, world_z] (米)
  │
  └─ 转为地图坐标:                                            (行 1396-1398)
      target_map_x = int(target[0] * 100.0 / 5.0)             // ×20
      target_map_y = int(target[1] * 100.0 / 5.0)
      target_set_step = step                                   // 记录时间戳
```

**具体例子:**

```
step=42: VA 返回 bbox {x1:220, y1:150, x2:380, y2:400}
         中心像素 = (300, 400), 深度 = 2.5m
         agent 世界位姿 (11.2, 13.5, heading=30°)
         反投影 → 目标世界坐标 (13.0, 14.8)
         → target_map_x = int(13.0 × 20) = 260
         → target_map_y = int(14.8 × 20) = 296
         → target_set_step = 42
```

#### 2. 消费: 传给 FMM 规划器

```python
waypoint = np.array([target_map_y, target_map_x])   # (行 1406)
# ★ 注意: waypoint = [row, col], 这是 ndarray 索引格式

self.policy._get_action(
    full_pose=current_pose,      # 世界坐标 [x, z, heading_deg]
    waypoint=waypoint,            # [target_map_y, target_map_x]
    traversable=traversable,      # 480×480 bool
    ...
)
```

#### 3. 消亡: 到达 / 超时

```python
# 到达检测 (行 1032-1040)
distance = sqrt((target_map_x - agent_map_x)² + (target_map_y - agent_map_y)²)
threshold = 0.75m / 0.05m/pixel = 15 pixels
if distance < 15:
    target_map_x, target_map_y = None, None   # 清零

# 超时检测 (行 1015-1029)
if step - target_set_step >= 15:              # 追了 15 步还没到
    visited_targets.pop()
    target_map_x, target_map_y = None, None   # 清零
```

### FMM 如何将 target_map_x/y 变成动作

```
FusionMapPolicy._get_action()                     Policy.py:46
  │
  ├─ 1. 坐标转换                                    (行 111-113)
  │    x, y = full_pose * 20           # 世界坐标(米) → 地图坐标(pixel)
  │    position = [y, x]               # → ndarray 索引 [row, col]
  │    heading = -full_pose[2]         # 取反 (坐标系差异)
  │
  ├─ 2. 创建 FMMPlanner                             (行 118)
  │    planner = FMMPlanner(config, traversible)
  │    # traversible: 480×480, 1=可通行, 0=障碍物
  │
  ├─ 3. 设置目标 → 计算距离场                         (行 119-124)
  │    planner.set_goal(goal)          # goal = [row, col]
  │    │
  │    │  FMMPlanner.set_goal()       fmm_planner.py:31
  │    │    traversible_ma[goal] = 0   # 目标点标记为 0
  │    │    dd = skfmm.distance(traversible_ma)
  │    │    # ★ 快速行进法: 每个格子存到达 goal 的最短距离
  │    │    self.fmm_dist = dd         # 480×480 float 数组
  │    │
  │    └─ 保存 fmm_fields/eps_X/step-Y.png (可视化)
  │
  ├─ 4. 取短期目标 STG                              (行 125)
  │    stg_x, stg_y, stop = planner.get_short_term_goal(position)
  │    │
  │    │  FMMPlanner.get_short_term_goal()  fmm_planner.py:40
  │    │    在 agent 周围 11×11 窗口内
  │    │    找 fmm_dist 值最小的格子 → STG (5 pixels ahead)
  │    │    if agent 距离 goal < waypoint_threshold*20:
  │    │        stop = True
  │    │
  │    └─ sub_waypoint = (stg_x, stg_y)
  │
  └─ 5. 方向 → 动作                                  (行 128-142)
       heading_vector  = agent 朝向单位向量
       waypoint_vector = STG - agent_position
       relative_angle, action = angle_and_direction(...)
       │
       │  angle < -15°  → action = 2 (TURN_LEFT)
       │  angle > +15°  → action = 3 (TURN_RIGHT)
       │  否则           → action = 1 (MOVE_FORWARD)
       │
       └─ return action
```

### FMM 距离场示意

```
FMM 距离场 (480×480, 数值 = 到 goal 的像素距离):

         agent ●                        goal ★
           |                              |
    [224,270]                         [260,296]
           |                              |
    ┌──────┼──────────────────────────────┼──────┐
    │ 520  500  480  ...  200  160  120   80  40 │  ← 距离值递减 →
    │  .    .    .         .    .    .    .   .  │
    │  .    .    .         .    .    .    .   .  │
    │  . →  . →  .  ...   . →  . →  . →  . → 0  │  ← goal 处 = 0
    └───────────────────────────────────────────┘

get_short_term_goal(): 在 agent 周围 11×11 窗口找最小值
  → 选出最陡下降方向 → 距离 agent 5 pixel 远的格子
  → angle_and_direction() → MOVE_FORWARD / TURN_LEFT / TURN_RIGHT
```

### 关键参数

| 参数 | 位置 | 值 | 含义 |
|------|------|-----|------|
| `MAP_RESOLUTION` | config | 5 cm/pixel | 地图分辨率 |
| `target_reached_threshold` | ZS_Evaluator_mp.py:123 | 75cm / 5 = 15 pixels | 到达阈值 |
| `max_steps_to_target` | ZS_Evaluator_mp.py:938 | 15 步 | 追目标超时 |
| `FMM_WAYPOINT_THRESHOLD` | config | 2.0m = 40 pixels | STG 判定 stop 的距离 |
| `FMM_GOAL_THRESHOLD` | config | 1.0m = 20 pixels | 最终目标 stop 阈值 |
| `step_size` | fmm_planner.py:12 | 5 pixels = 25cm | STG 前瞻距离 |
| `TURN_ANGLE` | config | 30° | 每次转向角度 |
| `FORWARD_STEP_SIZE` | config | 0.25m = 5 pixels | 每步前进距离 |

---

## 附录 C: 世界坐标系 vs 地图坐标系

### 本质

两个坐标系描述的是**同一空间**，区别在于连续 vs 离散。地图坐标系就是把世界坐标系量化为 5cm/pixel 的 grid。

```
Habitat 世界坐标系 (连续, float)           地图坐标系 (离散 int, 480×480 grid)
═══════════════════════════════════        ═══════════════════════════════════
                                          ndarray[row, col], col→右, row→下
       ↑ z (北)                                row ↑
       |                                       |
   24m │  ┌──────────────────┐             480 │  ┌──────────────────┐
       │  │                  │                 │  │                  │
   12m │  │   agent 初始      │             240 │  │   agent 初始      │
       │  │   (12, 12)       │                 │  │   [240, 240]     │
       │  │                  │                 │  │                  │
    0  │  └──────────────────┘               0 │  └──────────────────┘
       └─────────────────────────→ x           0       240       480
           0        12m       24m                        col →

转换公式:
  map_col    = x_world * 20       (100cm / 5cm = 20)
  map_row    = z_world * 20
```

### 为什么 FMM 必须在地图坐标中运行？

**FMM (skfmm.distance) 是离散网格算法，而 `traversable` 恰好是 grid 格式。**

```
traversable: 480×480 bool 数组, 来自语义建图:
  RGB-D → GroundedSAM → Semantic_Mapping → _process_map() → 480×480 ndarray

  ┌─────────────────────┐
  │ 0 0 0 1 1 1 1 1 ... │    0 = 障碍物 / 未知
  │ 0 0 0 1 1 1 1 1 ... │    1 = 可通行
  │ 0 0 0 1 0 0 1 1 ... │          ↑
  │ 1 1 1 1 0 0 1 1 ... │     墙 / 桌子
  │ 1 1 1 1 1 1 1 1 ... │
  └─────────────────────┘

skfmm.distance(traversible_ma) → 同尺寸 distance field
  不支持 float 世界坐标输入, 只接受 2D ndarray
```

放在世界坐标中算 FMM 的代价: 需要把整个 480×480 grid 映射回连续空间, 对任意浮点 (x,z) 对做 FMM —— 没有现成算法, skfmm 不接受连续坐标。

### 坐标系转换链路

```
┌──────────────────────────────────────────────────────────────┐
│                   世界坐标 (Habitat)                          │
│  full_pose = [x, y, z, heading_deg], 单位=米                │
│  heading=0 指向 +x (东)                                      │
│                                                              │
│  VA bbox → get_world_xz_from_pixel() → target [x,z] (米)    │
└────────────────────┬─────────────────────────────────────────┘
                     │ ×20 (100cm / 5cm per pixel)
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                   地图坐标 (ndarray)                          │
│  position = [row, col] = [z*20, x*20]  单位=pixel (5cm)      │
│                                                              │
│  waypoint = [target_map_y, target_map_x] = [row, col]        │
│  traversable = 480×480 bool                                  │
│  fmm_dist = skfmm.distance(traversible, goal=waypoint)       │
│  get_short_term_goal(position) → STG [row, col]              │
└────────────────────┬─────────────────────────────────────────┘
                     │ angle_and_direction()
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                   动作空间 (Habitat)                          │
│  action ∈ {0:STOP, 1:FORWARD, 2:TURN_LEFT, 3:TURN_RIGHT}    │
│  发回 env.step() → agent 在世界坐标中移动                     │
└──────────────────────────────────────────────────────────────┘
```

### 关键变换: Policy.py:111-113

```python
x, y, heading = full_pose        # 世界: x(米), y(米), heading(度)
x = x * (100 / 5)                # = x * 20  → 地图 col
y = y * (100 / 5)                # = y * 20  → 地图 row
position = np.array([y, x])      # ★ [row, col] = [y, x]
heading = -1 * full_pose[-1]     # 取反 (ndarray row 和世界 z 轴方向相反)
```

`position = [y, x]` 是关键 —— ndarray 索引是 `[row, column]`, 世界坐标的 y 映射到 row, x 映射到 column。

### 坐标系统总结

| 属性 | 世界坐标 | 地图坐标 |
|------|----------|----------|
| 类型 | float 连续 | int 离散 (ndarray index) |
| 单位 | 米 | pixel (5cm) |
| 范围 | ~24m × 24m | 480 × 480 |
| 原点 | (0, 0) 左下角 | [0, 0] ndarray 左上角 |
| agent 初始 | (12, 12) | [240, 240] |
| 转换 | — | `col=x*20, row=z*20` |
| 用途 | Habitat 环境交互, 深度反投影 | 语义建图, FMM 规划, 到达检测 |
| FMM 输入 | ❌ 不支持 | ✅ traversable grid + waypoint [row, col] |
| 到达检测 | ❌ 不直接使用 | ✅ `dist < 15 pixels` |

---

## 附录 D: NavDP vs FMM

### 核心差异

| 维度 | FMM | NavDP |
|------|-----|-------|
| **坐标系** | 地图坐标 (480×480 grid, 5cm/pixel) | **机体坐标系** (body-frame, 米) |
| **输入** | `traversable` grid + `waypoint [row, col]` | RGB 图像 + 深度图 + `goal [x_fwd, y_left, z_up]` |
| **算法** | 经典路径规划 (`skfmm.distance`, 快速行进法) | 学习型神经网络 (Diffusion Policy + Transformer) |
| **输出** | 1 个离散动作 (`1/2/3`) | 完整轨迹 (24 个 body-frame waypoints), 取 lookahead → 动作 |
| **依赖地图** | ✅ 必须有 traversable map (来自语义建图) | ❌ 不看地图, 直接从像素端到端预测 |
| **记忆** | 无 | 8 帧 RGB 图像队列 (`memory_queue`) |
| **图像输入** | 无 (只接受 grid) | RGB 640×480 → resize 224×224; Depth → 224×224 |
| **楼梯** | ❌ 无法处理 (楼梯被映射为障碍物) | ✅ 训练数据含楼梯场景 |
| **确定性** | 确定性的 (相同输入 → 相同输出) | 随机采样 (diffusion denoising) |
| **推理速度** | 极快 (纯 numpy/skfmm) | 较慢 (神经网络前向 + diffusion 多步去噪) |
| **需要训练** | 否 | 是 (`navdp-cross-modal.ckpt`) |

### NavDP 的坐标系: 机体坐标系 (Body-frame)

NavDP 完全不碰 480×480 的地图 grid，所有计算在**机器人自身坐标系**中进行:

```
机体坐标系 (body-frame / ego-centric):

              ↑ x_body (forward, 机器人正前方, 单位=米)
              |
        ┌─────●─────┐    ● = 机器人当前位置
        │     |     │
        │     |     │    goal = [x_body, y_body, z_body]
  ←─────┼─────●─────┼─────→ y_body (left, 机器人左侧)
        │     |     │
        │     |     │    例: [2.5, 0.3, 0]
        └─────┼─────┘         前方 2.5m, 左侧 0.3m
              │
              ↓ (-y_body = right)
```

### goal 构造: 地图坐标 → 世界坐标 → 机体坐标

```
ZS_Evaluator_mp.py:1556-1569

# 1. 地图坐标 → 世界坐标 (米)
tx = target_map_x * 0.05              # pixel × 5cm = 米
ty = target_map_y * 0.05

# 2. 世界坐标 → 机体坐标 (旋转到机器人朝向)
ax, ay = current_pose[0], current_pose[1]   # agent 世界位置
ah = current_pose[2]                         # agent 朝向 (弧度)

dx = tx - ax           # 世界系中的 δx
dy = ty - ay           # 世界系中的 δy

x_body =  dx * cos(ah) + dy * sin(ah)    # 投影到机器人前方
y_body = -dx * sin(ah) + dy * cos(ah)    # 投影到机器人左侧
z_body = 0.0                             # 高度差 (平地上=0)

goals = np.array([[x_body, y_body, z_body]])
```

### NavDP 的完整输入

```python
# navdp/policy_agent.py:157-184

NavDP_Agent.step_pointgoal(
    goals=[[x_body, y_body, z_body]],    # 机体坐标目标 (米), clip 到 [-10,10]
    images=rgb[None, ...],               # (1, 640, 480, 3) uint8
    depths=depth[None, ...],             # (1, 640, 480, 1) float (米)
)

# 内部处理:
#   process_image()  → resize + pad → (1, 224, 224, 3) float32 [0,1]
#   process_depth()  → resize + pad → (1, 224, 224, 1)
#   memory_queue: 最近 8 帧 → (1, 8, 224, 224, 3)
#   process_pointgoal() → clip(goals, -10, 10)
#
# → NavDP_Policy.predict_pointgoal_action(goals, images, depths)
# → all_trajectory: (B, N_modes, 24, 2)  24步 × 2D (x,y), 单位=米
# → all_values:     (B, N_modes)         critic 评分
# → good_trajectory: 最高分的那条轨迹
```

### NavDP 输出: 轨迹也在机体坐标中

```python
# 输出轨迹 → 取 lookahead → 判断动作
# good_trajectory[:, 0] shape = (Batch, 24, 3) or (Batch, 24, 2)
# 每个 waypoint = [x_body, y_body]  单位=米 (相对当前机器人位置)

local_traj = next_point[0]     # (24, 2) or (24, 3)

# 取 lookahead 点 (0.5m 前方)
for pt in local_traj:
    dist = sqrt(pt[0]² + pt[1]²)
    if dist > 0.5:
        target_pt = pt
        break

x, y = target_pt[:2]
angle = arctan2(y, x)          # 轨迹方向相对机器人前方的角度
if angle > 0.26:                # 15°
    action = TURN_LEFT
elif angle < -0.26:
    action = TURN_RIGHT
else:
    action = MOVE_FORWARD
```

### 坐标系对比图

```
FMM (地图坐标系, top-down grid):         NavDP (机体坐标系, ego-centric):

        480×480 grid                            RGB-D 视角
   ┌─────────────────┐                          ┌──────────┐
   │                  │                          │          │
   │     goal ★       │                          │    ★ goal│
   │      |           │                          │    |     │  goal = [2.5, 0.3, 0]
   │      | (距离场)   │                          │    |     │  前方2.5m, 左侧0.3m
   │      |           │                          │    |     │
   │   agent ●        │                          │    ●     │  机器人看前方
   │                  │                          │  camera  │
   └─────────────────┘                          └──────────┘

   输入: traversable[480,480]                    输入: rgb[640,480,3]
         waypoint = [row, col]                          depth[640,480,1]
                                                         goal = [2.5, 0.3, 0]

   算法: skfmm.distance()                          算法: NavDP_Policy
         (数值方法, 零样本)                              (transformer + diffusion,
                                                         预训练权重)

   输出: 1个动作 ID                                  输出: 24步轨迹 → lookahead → 动作
```

### rollout 中的切换逻辑

```
ZS_Evaluator_mp.py:1408-1433

决策: 用 FMM 还是 NavDP?

  if not self.use_fmm:
      → 必须用 NavDP (或 iPlanner), FMM 被禁用           (行 1419-1423)

  elif self.agent.stair and (self.use_navdp or self.use_iplanner):
      → FMM 启用, 但检测到楼梯 → 临时切换到 NavDP        (行 1426-1430)
        原因: 楼梯在 traversable map 上像障碍物, FMM 会绕开

  else:
      → 正常情况 → FMM                                     (行 1431-1433)
```

### 各自优势场景

```
FMM 适合:                              NavDP 适合:
┌────────────────────────────┐        ┌────────────────────────────┐
│ • 平地上的自由导航           │        │ • 楼梯 / 斜坡               │
│ • 已建好 traversable map    │        │ • 复杂地形需像素级推理       │
│ • 零样本, 不需训练            │        │ • --no-fmm 时作为主力规划器  │
│ • 可解释 (距离场可可视化)     │        │ • 狭窄通道 / 精确轨迹跟踪    │
│ • 速度快                     │        │ • 需要端到端隐式推理的场景    │
└────────────────────────────┘        └────────────────────────────┘

两者共享同一个 target_map_x/y (都从 VA bbox 反投影得到),
但 NavDP 在调用前先把地图坐标 → 世界坐标 → 机体坐标,
而 FMM 直接用地图坐标 [row, col]。

### NavDP 的 z_body 与楼梯处理

#### goal 的 z 始终为 0

```python
# ZS_Evaluator_mp.py:1565-1567
x_body =  dx * np.cos(ah) + dy * np.sin(ah)    # 前方 (米)
y_body = -dx * np.sin(ah) + dy * np.cos(ah)    # 左侧 (米)
z_body = 0.0                                     # ★ 始终为 0
```

goal 来自 VA bbox → 深度反投影 → 2D 世界地面位置，没有高度信息，所以 `z_body=0` 是必然的。

#### NavDP 网络内部: 确实预测了 3D 轨迹

```python
# policy_network.py:110-121
noisy_action = torch.randn((..., 24, 3))          # ★ 24步, 每步 [Δx, Δy, Δz]
# ... diffusion 去噪 ...
all_trajectory = torch.cumsum(naction / 4.0, dim=1)  # 积分为绝对位置
# shape: (B, N_samples, 24, 3)                     # 24个waypoints, 每个有 [x,y,z]
```

网络输出 `(24, 3)`——z 维度的确被预测了。

#### rollout 只用 x, y 做决策，z 被丢弃

```python
# ZS_Evaluator_mp.py:1593-1600 (NavDP 调用点)
x, y = local_traj[0][:2]           # ★ 只取 [:2]
angle = np.arctan2(y, x)
if angle > 0.26:   action = TURN_LEFT
elif angle < -0.26: action = TURN_RIGHT
else:               action = MOVE_FORWARD
```

```python
# 可视化投影 (policy_agent.py:48-49): z 被硬编码
input_points[:, 0:2] = waypoints      # x, y 来自网络输出
# input_points[:, 2] = -0.2           # z 被硬编码为 -0.2, 不读取网络输出的 z!
```

#### z 维度在训练中的实际作用

```python
# policy_network.py:122-123 — 唯一的 z 使用处
trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)   # 最终点的 x,y 距离
all_trajectory[trajectory_length < 0.5] = \
    all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0, 0, 1.0]]])
#   当轨迹太短 (< 0.5m): 清零 x,y, 只保留 z → 网络被鼓励至少输出有意义的方向
#   这可能是训练 loss 中防止退化到原地不动的正则化技巧
```

#### 楼梯处理的真正机制: 不是靠 z，是靠"不把楼梯当障碍物"

```
FMM 视角 (traversable map):               NavDP 视角 (RGB-D 像素):

  楼梯在 map 上 = 一片 0 (障碍物)           深度图中楼梯有连续的几何纹理
  ┌────────────────────┐                  ┌──────────────────┐
  │ 1 1 1 1 0 0 1 1   │  traversable      │ ░░░░░░░░░░░░░░░░ │
  │ 1 1 1 1 0 0 1 1   │  = 0 (不可通行)    │  ../../../...   │ ← 神经网络从
  │ 1 1 1 1 0 0 1 1   │  → FMM 绕路       │ ../../....      │   训练数据中学会
  │ 1 1 1 1 ● → → ?   │  无路可走!          │ ../.....        │   这是"可以走的路"
  └────────────────────┘                  │ ●→→→→→→→→→→→→→ │   预测轨迹直穿
                                          └──────────────────┘
```

**三层分工:**

```
1. NavDP 网络         →  从 RGB-D 像素"看懂"楼梯不是死路, 预测轨迹直穿过去
                        z 分量训练中存在但 rollout 不显式使用

2. Habitat 物理引擎   →  MOVE_FORWARD (0.25m步长) 自动沿楼梯表面攀爬
                        碰撞检测 + 斜坡滑动 → 物理层面完成高度变化

3. rollout 切换逻辑    →  agent.stair == True 时选择 NavDP 而非 FMM
                        原因: FMM 的 traversable map 把楼梯当障碍物,
                              NavDP 从像素理解场景, 不受此限制
```

**一句话:** goal 的 `z_body=0` 是目标点的 2D 特性；NavDP 网络内部预测了 z 但 rollout 不用；真正"爬楼梯"靠的是 NavDP 从 RGB-D 中学会不走弯路 + Habitat 物理引擎自动处理高度变化。
