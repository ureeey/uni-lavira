# Prompt 模板详解 — OBJNAV 任务

> 代码位置: `vlnce_baselines/prompts/prompts_objnav.py`  
> 加载机制: `get_prompts("OBJNAV")` → `self.P` (agent.py:75)  
> 补充上下文: 图像、history、当前视图等在 agent.py 中拼装，不在模板内

---

## 实际使用的 7 个 Prompt

### 1. `LA_PROMPT_BACKTRACK` — LA 主决策（有 backtrack 选项）

**调用:** `agent.py:154` → `_build_nav_prompt(use_backtrack_prompt=True)`  
**触发条件:** `should_consider_backtrack and num_waypoints > 0 and available_ids` 非空  
**位置:** prompts_objnav.py:1-52

#### 模板结构

```
Based on the navigation history, current 4-directional views, and the TODO list,
decide the next action to find the target object.

Current TODO List:
{todo_list}              ← self.agent._format_todo_for_prompt()

{feedback}               ← stop_feedback + todo_verification_feedback

Task:
1. FIRST, write `reasoning_todo`: review ALL TODO items ...
   - completed with result / rewrite / add / remove
2. THEN, write `reasoning_action` and choose the next action
   Choose one of these actions:
{action_list}             ← 动态生成的可用动作, 例如:
                            "   - navigate to forward - continue straight ahead"
                            "   - navigate to left - turn left and go forward"
                            "   - backtrack to <waypoint_id> - ... (Available IDs: 0,1,2)"

Response format (JSON):
{
    "progress_analysis": "<brief assessment of what you've observed and done so far>",
    "reasoning_todo": "<reasoning behind the TODO updates>",
    "todo_updates": [
        {"index": <int>, "status": "completed", "result": "<REQUIRED observation>"},
        {"op": "rewrite", "index": <int>, "content": "<refined description>"},
        {"op": "add", "content": "<new sub-goal>", "status": "pending"},
        {"op": "remove", "index": <int>}
    ],
    "reasoning_action": "<reasoning for the chosen action>",
    "action": "{action_desc}",      ← 例如 "navigate to forward|...|backtrack to <waypoint_id>"
    "stop": True/False,
    "stair": False|"up"|"down"
}
```

#### 关键字段

| 字段 | 含义 |
|------|------|
| `progress_analysis` | 当前搜索进度总结 |
| `todo_updates` | TODO list 增量更新 (complete/rewrite/add/remove) |
| `action` | 动作字符串, 直接映射到 `action_list` (append 转向动作) |
| `stop` | 是否认为已到达目标 (触发 STOP 流程) |
| `stair` | 是否在楼梯上 (触发 NavDP 替代 FMM) |

---

### 2. `LA_PROMPT_NO_BACKTRACK` — LA 主决策（无 backtrack 选项）

**调用:** `agent.py:154` → `_build_nav_prompt(use_backtrack_prompt=False)`  
**触发条件:** 无可用 backtrack waypoint（第一个 waypoint 或所有历史点距离 > 6m）  
**位置:** prompts_objnav.py:110-152

#### 与 BACKTRACK 版的区别

| 区别 | BACKTRACK 版 | NO_BACKTRACK 版 |
|------|-------------|-----------------|
| `action_list` 内容 | 含 `backtrack to <waypoint_id>` | 只有 `navigate to forward/left/right/behind` |
| `stop` 字段 | `True/False` (允许停) | 硬编码 `False` (第一waypoint不允许停) |
| 其余结构 | 完全相同 | 完全相同 |

---

### 3. `LA_PROMPT_BACKTRACK_REPLAN` — Backtrack 后重规划

**调用:** `agent.py:1482` → `replan_at_backtrack()`  
**触发条件:** LA 决定 backtrack + `self.backtrack_second_chance=True`  
**位置:** prompts_objnav.py:55-91

#### 模板结构

```
You are a navigation agent. You have backtracked to a previous waypoint
to have a second chance to choose action.

Instruction: "{instruction}"

Navigation History (up to this waypoint):
(Images provided above)

Current 4-directional views at this waypoint:
(Images provided above)

Previous Action: **navigate to {previous_action}** from here.   ← 上次从这里走的方向

Previous Trajectory (Path taken from here):
(Images provided above)                                        ← 失败路径的图像

Task:
- Review the Previous Trajectory to understand the outcome of the previous choice.
- Analyze the Current 4-directional views to give a second-chance choice.
- The previous choice can be reconsidered if necessary.

Available Actions:
{action_list}              ← 不含 backtrack (已经在backtrack点了)

Response format (JSON):
{
    "reasoning": "<analysis of the previous path and why the new direction is chosen>",
    "action": "{action_desc}",
    "stop": True/False,
    "stair": False|"up"|"down"
}
```

#### 与主 LA prompt 的区别

- **不含 TODO list** — backtrack 是紧急纠正，不需要更新计划
- **不含 `progress_analysis` 和 `todo_updates`** — JSON 结构更简单
- **多了 `previous_action`** — 告诉模型"上次往这边走失败了"
- **多了失败路径图像** — agent.py:1490-1493 中附上了 `failed_path_content`

#### agent.py 中的完整拼装

```python
# agent.py:1490-1496
content = [{"type": "text", "text": "Navigation History:"}]
content.extend(history_content)                           # 历史成功路径图像
content.append({"type": "text", "text": "Previous Trajectory:"})
content.extend(failed_path_content)                       # 失败路径图像
content.append({"type": "text", "text": "Current 4-directional views at Backtrack Waypoint:"})
content.extend(current_views)                             # 当前全景
content.append({"type": "text", "text": prompt})          # 文本模板
```

---

### 4. `LA_PROMPT_TODO_GENERATOR` — 初始 TODO 生成

**调用:** `agent.py:699` → `generate_todo_list()`  
**触发条件:** 每个 episode 第一次 LA 决策前 (`self.todo_list is None`)  
**位置:** prompts_objnav.py:93-108

#### 模板

```
You are a navigation agent.
Instruction: "{instruction}"

Create a TODO list to complete this instruction.

Response format (JSON ONLY):
{
    "todos": [
        {"content": "<step description>", "status": "pending"},
        {"content": "<step description>", "status": "pending"},
        ...
    ]
}

Return ONLY the JSON object.
```

#### 特点

- **最简单的模板** — 没有任何图像输入
- **输入只有 instruction** — 例如 `"Find the footrest"`
- **输出是初始 TODO list** — 之后每步 LA 决策时可以增量更新
- **调用后会接一个验证步骤** — 用单独的 LLM 调用来验证 TODO 内部一致性

---

### 5. `VA_PROMPT` — VA 获取 bbox

**调用:** `agent.py:929` → `query_llm()`  
**触发条件:** 每次全景采集 + LA 方向决策后，需要从 RGB-D 中识别目标并获取 bbox  
**位置:** prompts_objnav.py:155-178

#### 模板结构

```
Navigation Task: "{instruction}"

Current situation:
- Step: {current_step}
- Image size: {width}x{height} pixels
- Current TODO List:
{todo_list}
- Progress Info: {progress_info}       ← LA 输出的 progress_analysis

Your task:
1. Identify the most relevant target object/area for what you should do next.
   Specify ONLY ONE. And it should not be too close to you.
2. Provide the bounding box of the target.

Response format (JSON):
{
    "reasoning": "<brief explanation of decision>",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<description of target object>"
}

Guidelines:
- Target description should be specific and clear.
- If deciding to go up or down stairs, the bbox MUST select the ENTRY of the stairs.
- Provide bounding box of next target.
```

#### agent.py 中的完整拼装

```python
# agent.py:889-939
content = [{
    'type': 'text',
    'text': self.P.ROBOT_NAVIGATION_SYSTEM_PROMPT,   # ← system prompt
}]
content.append({"type": "image_url", "url": rgb_base64})   # RGB 图像
content.append({"type": "image_url", "url": depth_base64}) # 深度图
content.append({"type": "text", "text": VA_PROMPT文本})    # 上面的模板
```

#### 输出用途

VA 返回的 bbox 被用于:
1. `bbox_2d` → 深度反投影 (`get_world_xz_from_pixel`) → `target_map_x/y`
2. `target` 字符串 → 存入 `visited_targets[-1]['description']`
3. `reasoning` → 存入 `visited_targets[-1]['llm_reasoning']`

---

### 6. `STOP_CHECK_PROMPT` — STOP 双重确认

**调用:** `agent.py:1264` → `double_check_stop()`  
**触发条件:** LA 在主决策中设了 `stop=True`，且 agent 到达了该 target  
**位置:** prompts_objnav.py:303-325（文件末尾第二个定义，覆盖了第一个）

#### 模板结构

```
You are an intelligent navigation agent.
Task: "{instruction}"

You have decided to STOP, believing you have reached the target.
Now, please double-check your decision based on the current 4-directional views.

Current Views:
{current_views}          ← 通过 split 插入 4 张全景图像 + 标签

Requirements:
1. Check if the target object is clearly visible in any of the views.
2. Estimate if the distance to the target is less than 1 meter.

Decision Rules:
- If target is visible AND distance < 1m: CONFIRM STOP.
- If target is NOT visible OR distance >= 1m: CONTINUE NAVIGATION.

Response format (JSON):
{
    "analysis": "<analysis of visibility and distance>",
    "decision": "STOP" or "CONTINUE"
}
```

#### 特殊拼装方式

与其他 prompt 不同，`STOP_CHECK_PROMPT` 使用 `split("{current_views}")` 将图像插入中间:

```python
# agent.py:1264-1270
parts = STOP_CHECK_PROMPT.split("{current_views}")
part1 = parts[0].format(instruction=instruction)
rest = parts[1]

content = [{"type": "text", "text": part1}]
content.extend(current_views)     # 4 张全景图像插在模板中间
content.append({"type": "text", "text": rest})
```

#### 确认/拒绝后的行为

```
should_stop == True:
  → action_list.append(0)  → 下一步 STOP
  → 对于 EQA, 额外调用 query_llm_oracle 获取答案

should_stop == False:
  → going_to_stop = False  → 重新进入正常导航流程
  → consecutive_stop_failures += 1
  → stop_feedback 记录拒绝原因 → 传给下一次 LA 调用
  → 拒绝 3 次后强制 STOP
```

---

### 7. `ROBOT_NAVIGATION_SYSTEM_PROMPT` — VA 的 System Prompt

**调用:** `agent.py:891` → `query_llm()`  
**触发条件:** 每次 VA 调用  
**位置:** prompts_objnav.py:379-382

```
You are a robot performing navigation task.
Look at this image and the corresponding depth gray image and help the robot navigate.
```

放在 VA messages 的 `content[0]` 位置作为开路文本，在图像之前。

---

## 未使用的 5 个 Prompt (死代码)

### `LA_PROMPT_OBJECTNAV` (line 211-243)

旧版 ObjectNav LA prompt，不含 TODO list 机制。被 `LA_PROMPT_BACKTRACK` / `LA_PROMPT_NO_BACKTRACK` 替代。

### `VA_PROMPT_OBJECTNAV` (line 277-301)

旧版 ObjectNav VA prompt，含 `{target_object}` 和 `visited_targets_str`。被统一的 `VA_PROMPT` 替代。

### `VA_PROMPT_VISITED_TARGETS` (line 249-275)

含 `{visited_targets_str}` 和 `action: "NAVIGATE" or "STOP"` 字段的 VA prompt 变体。注释说"保留供参考"。

### `GUIDELINES_GENERATOR` (line 327-357) / `GUIDELINES_GENERATOR_OBJECTNAV` (line 359-374)

用于从 instruction 生成持久化导航指南，但 `agent.py:68` 硬编码 `self.use_guideline = False`，且 `agent.py:385-386` 中对应的 content 行被注释掉了。

---

## Prompt 调用链总览

```
Episode 开始
  │
  ├─ generate_todo_list()                  agent.py:699
  │   └─ LA_PROMPT_TODO_GENERATOR          → 初始 TODO list
  │
  └─ rollout 循环 ──────────────────────────────────────────┐
      │                                                      │
      ├─ [每次全景采集后] navigate_or_backtrack()              │
      │   ├─ LA_PROMPT_BACKTRACK         有历史waypoint       │
      │   └─ LA_PROMPT_NO_BACKTRACK      无历史waypoint       │
      │        → action + direction + todo_updates            │
      │                                                      │
      ├─ [LA 决定方向后] query_llm()                          │
      │   ├─ ROBOT_NAVIGATION_SYSTEM_PROMPT  (content[0])    │
      │   └─ VA_PROMPT                       (content[末尾])  │
      │        → bbox → 深度反投影 → target_map_x/y          │
      │                                                      │
      ├─ [LA 说 STOP 且到达后] double_check_stop()            │
      │   └─ STOP_CHECK_PROMPT                                │
      │        → STOP / CONTINUE                              │
      │                                                      │
      └─ [backtrack 时] replan_at_backtrack()                 │
          └─ LA_PROMPT_BACKTRACK_REPLAN                       │
               → 新方向 + 新 bbox                             │
```

## 各 Prompt 输入源汇总

| Prompt | 图像输入 | 文本 placeholder | 来源 |
|--------|---------|-----------------|------|
| `LA_PROMPT_BACKTRACK` | history_images (n张) + current_views (4张) | `{todo_list}`, `{feedback}`, `{action_list}`, `{action_desc}`, `{negative_constraints}` | agent.py:143-156 |
| `LA_PROMPT_NO_BACKTRACK` | 同上 | 同上 | agent.py:143-156 |
| `LA_PROMPT_BACKTRACK_REPLAN` | history_images + failed_path + current_views (4张) | `{instruction}`, `{action_list}`, `{action_desc}`, `{previous_action}`, `{negative_constraints}` | agent.py:1482-1488 |
| `LA_PROMPT_TODO_GENERATOR` | 无 | `{instruction}` | agent.py:699 |
| `VA_PROMPT` | RGB (1张) + Depth (1张) | `{instruction}`, `{current_step}`, `{width}`, `{height}`, `{todo_list}`, `{progress_info}` | agent.py:929-937 |
| `STOP_CHECK_PROMPT` | current_views (4张, 插在中间) | `{instruction}` | agent.py:1264-1266 |
| `ROBOT_NAVIGATION_SYSTEM_PROMPT` | 无 placeholder | — | agent.py:891 |
