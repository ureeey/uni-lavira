# 个人笔记

*心得后面是实践记录，尝试实现实时性的 rollout v2/v3/v4 在最后。*

## 心得

### benchmark 误差

- 仅靠结束时与目标距离来判定是不够的，成功率中有一部分是蒙对的，没有校验与排除机制。
- HM3D-OVON 开放词汇标注不完美，例如 picture、window glass 等目标，有的 episode 标注有遗漏、或者是标注者与测试者理解不一致，导致失败。
- 破损的墙面、留着缝的门、能够看到室外场景的玻璃门或窗 都会造成 可见但不可达，这些可以说是挑战，也可以说是 benchmark 本身需要改进的地方。
- 曾经遇到过，机器人看到一个边缘房间里的情况、一直想进去，但是这个房间恰恰只能看见里面但进不去，就一直僵在那里，最后甚至超出地图边界。

### VLM 与 prompt

- VLM API 设置为 temperature = 0 且 禁用思考。

- 同一个 episode 多次测试由多个路线完成，反映了 VLM 的随机性。

- 模型确实需要提示来遵循指令，看上去是提示得越多、遵循得越好。

    比如要通过加系统提示词“Output only the result. No analysis.”和用户要求“Respond with exactly one line”，才能让模型不长篇大论，按格式输出。

    还有 误报目标在图像中、认不出可以探索的区域、容易选取墙面中无意义的区域 可以通过提示来改善。

### FMM

- FMM 的 到达判定、卡住判定、最大步数 需调优，尝试了用更优雅的策略，但还是不完美。

- episode 1297 发生过“穿墙”现象，具体而言是由于噪声目标定在了一个从未去过的房间、即当前区域可见墙面的后面，结果 FMM 就像是提前知道那里有个房间似的，绕过墙面、边走边看，真的过去了。

- 受限于机器人的高度和视野，经常会发生远处看见目标，凑近了又看不到目标（例如高处的画、低处的凳子）而傻傻错过的情况。不过这不是什么大问题，真机配备恰当的传感器即可解决，或者哪怕加个 PTZ 也行。

### rollout v4 之后该做什么？

    原项目的 rollout 即 v1，每个 waypoint 决策需要调用两次 API (LA 和 VA)，LA 10秒起步、VA 5秒起步，我觉得这太慢了，所以尝试改进。先试下 ObjNav 任务。

    v2 的贡献是简化了 prompt、约定了模型只要判断情形、给出区域即可。但由于对 API 调用完整流程不了解，错误的采用多轮对话机制、且没有沿用全景扫描这个好的策略。

    现在看来，API 耗时主要由三部分构成：网络延迟，prefill，decode，其中，prefill 耗时对于输入 token 的数量不敏感、但有一个 2 秒左右的基础开销，但 decode 耗时对于输出 token 的数量非常敏感，所以，一方面要减少 API 调用次数，另一方面要减少输出 token 的数量。

    v3 的贡献是将调用次数合并为 1 至 2 次，对于需要探索的情况，由第二次 API 调用判断是否重复、以及最优选择，并沿用了全景扫描。

    v4 做得更加彻底，直接把 v3 的 1 至 2 次合并为 1 次，并且优化了 FMM。但是，当我做完这些的时候，逐渐意识到，记忆问题到底该如何解决？仅仅用带bbox的历史图像应该是不行的。

    由近及远的去设想：
    
    1.v4 实时性可以做到平均 3 秒 1 次 waypoint 决策，现在就差记忆能力了，抛开 VLM 和 prompt，用传统方法辅助一下，比如基于 waypoint 建立拓扑图，再基于拓扑图去实现高效探索、纠错回溯。

    2.VLM 本身能不能处理 拓扑图？如果可以，那是不是在输入图像和自然语言文本之外，再输入可用于构建拓扑图（可能是结构化文本形式）的基本信息给 VLM，让 VLM 自己构建拓扑图并高效探索、纠错回溯。

    3.设想一个机器人与人视频通话，机器人把环境画面实时传给人，人是可以根据画面和目标实时指挥机器人该做什么的，包括停下或者朝画面中某个区域移动，这个过程放在 VLM 框架下是一种多轮对话，user（刚才的机器人）发图像，assistant（刚才的人类）回答动作，反复进行，直到停止。VLM 如果可以 in-context learn 导航对话示例，那事情就变得简单了。一方面这样可以利用 kv cache 命中，提升响应速度，另一方面有清晰的路径来提升成功率，路径包含提升预训练 VLM 模型本身的能力 和 设计更好的导航对话示例。

## API 准备

- 先申请好 API 密钥，保存到环境变量中。
- 网络代理可能影响远程调用模型 API，注意排查。
- 如果访问阿里云 API 有问题，用以下命令对比测试（遇到过 IPv6 的坑）：

  ```bash
  python hello_qwen.py
  python hello_qwen.py --force-ipv4
  ```

- API 请求体 20MB 限制问题 通过 DashScope Public 方式解决，run_mp.py 加上 --api-format dashscope 选项。
- `test_api.py` 可以测试更多厂商和模型的 API，例如 DeepSeek 的 `deepseek-V4-pro`。
- 阿里云可以在服务器侧配置日志，用于分析组装的请求是否符合预期、API 耗时等，还可以观察到 DashScope Public 是用 OSS 传图的。

## 可视化

> 运行前请先确保目录下（例如saved_rgb_images/test-ovon/2518）没有以前测试残留的图片。
```bash
python watch_viz.py --auto
```

## 单条执行

### HM3D-v2

> HM3D 数据集较容易申请到。

```bash
source .env.local && source env.sh
python run_mp.py \
  --exp-name test \
  --run-type eval \
  --exp-config vlnce_baselines/config/objectnav_v2.yaml \
  --nprocesses 1 \
  --debug-episodes 0 \
  TRAINER_NAME ZS-Evaluator-mp \
  TORCH_GPU_IDS [0] \
  NUM_ENVIRONMENTS 1
```

### HM3D-OVON

```bash
source .env.local && source env.sh
python run_mp.py \
  --exp-name test-ovon \
  --run-type eval \
  --exp-config vlnce_baselines/config/objectnav_ovon.yaml \
  --nprocesses 1 \
  --debug-episodes 2469 \
  TRAINER_NAME ZS-Evaluator-mp \
  TORCH_GPU_IDS [0] \
  NUM_ENVIRONMENTS 1
```

## 测试记录

### HM3D-v2（单 episode）

| Episode | 结果 | 备注 |
|---------|------|------|
| 0       | ✅ ok | 多次测试，路线有多样性 |

### HM3D-OVON（单 episode）

| Episode | 结果 | 备注 |
|---------|------|------|
| 53      | ❌ fail | 探索效率低，在第一个房间有点打转；最终把用别的东西装的一簇花当成花瓶了 |
| 2469    | ✅ ok | 看上去正常 |
| 1297    | 时好时坏 | [dashscope] |
| 2518    | 时好时坏 | 失败的时候是因为老被沙发挡着，看到目标了但是过不去 |

### HM3D-OVON 100-episode 全量测试

**日期**: 2026-07-16 13:42 ~ 23:35 (约 9h53m)
**命令**: `bash eval_scripts/hm3d_ovon.sh`
**配置**: LA=qwen3.6-plus, VA=qwen3.6-plus, DashScope Public 模式 (`dashscope_maas=False`, `base_url=https://dashscope.aliyuncs.com/api/v1`), 1 worker (GPU 0)
**批次**: `data/datasets/stratified_samples/hm3d_ovon.json` (100 episodes)

#### 结果汇总

| 指标 | 值 |
|------|-----|
| 总 episode | 100 |
| 成功 | **54 (54.0%)** |
| 失败 | 46 (46.0%) |
| 总步数 | 22,303 |
| 平均步数/ep | 223.0 |
| 成功平均 SPL | **0.600** |
| 总耗时 | 9h53m |

#### 模型用量

| | LA | VA | 合计 |
|------|-----|------|------|
| 调用次数 | 1,178 | 935 | 2,113 |
| 总耗时 | 3.4h | 1.2h | 4.6h |
| Input tokens | 13.9M | 0.97M | 14.9M |
| Output tokens | 328K | 122K | 450K |
| 平均响应 | 10.3s | 4.8s | 7.8s |
| 最大请求体 | 70 MB | 0.5 MB | — |

#### 按场景成功率

| 场景 | Ep数 | 成功 | 成功率 |
|------|------|------|--------|
| 00802-wcojb4TFT35 | 10 | 5 | 50% |
| 00891-cvZr5TUy5C5 | 7 | 2 | 29% |
| 00873-bxsVRursffK | 7 | 6 | **86%** |
| 00877-4ok3usBNeis | 6 | 2 | 33% |
| 00844-q5QZSEeHe5g | 6 | 4 | 67% |
| 00814-p53SfW6mjZe | 5 | 2 | 40% |
| 00862-LT9Jq6dN3Ea | 5 | 2 | 40% |
| 00839-zt1RVoi7PcG | 5 | 2 | 40% |
| 00890-6s7QHgap2fW | 5 | 3 | 60% |
| 00869-MHPLjHsuG27 | 5 | 3 | 60% |
| 其他小桶 (<5ep) | 39 | 23 | 59% |

#### 异常记录

| 类型 | 次数 | 说明 |
|------|------|------|
| DashScope 连接超时 | 15 (LA 8 + VA 7) | 全部自动重试成功，不影响数据完整性 |
| JSON 解析失败 | 7 | STOP double-check 阶段 LLM 返回 `{{...}}` 双花括号格式不规范 |
| LA 最大请求体 | 70 MB | 长 episode 末尾多轮全景+历史累积导致，未触发 API 限流 |
| GPU 显存 | 稳定 2.5–3.1 GB | 无泄漏 |
| 进程内存 (RSS) | 720→105 MB | 正常，场景释放后回落 |

#### 关键观察

- 场景间成功率差异很大（29% ~ 86%），说明场景布局对 agent 性能影响显著。
- DashScope qwen3.6-plus 公共端点约每 20-25 分钟波动一次，但重试机制可靠。
- LA 请求体在 episode 内线性增长（从 ~1MB 到 30-70MB），需要关注长 episode 是否可能触及 token 上限。
- JSON 解析失败集中在 STOP double-check，根因是 LLM 输出了 `{{` 而非 `{`，可在 prompt 或解析侧加固。

#### 随机性分析

##### SEED 传递链路

```
Shell                           YAML（实验级）                  YAML（任务级）
eval_scripts/hm3d_ovon.sh       objectnav_ovon.yaml             habitat_extensions/config/objectnav_ovon.yaml
  --exp-config ──────────────→  BASE_TASK_CONFIG_PATH ────────→ SEED: 0
  vlnce_baselines/config/        habitat_extensions/config/       ↑
  objectnav_ovon.yaml            objectnav_ovon.yaml              │
                                         │                        │
                                  get_config() 读取 BASE_TASK_CONFIG_PATH
                                  调用 get_task_config() 加载此文件
                                         │                        │
                                  run_mp.py:425 ─────────────────┘
                                  seed_everything(config.TASK_CONFIG.SEED)
```

##### SEED 控制的随机数生成器

`seed_everything()` (`vlnce_baselines/utils/misc.py:8-13`) 设置：

| 生成器 | 调用 | 影响范围 |
|--------|------|----------|
| Python `random` | `random.seed(seed)` | Episode shuffle 顺序 (`habitat_extensions/task.py:143`) |
| NumPy | `np.random.seed(seed)` | 地图构建、数据处理 |
| PyTorch CPU | `torch.manual_seed(seed)` | GroundedSAM / RepViTSAM mask 生成 |
| PyTorch CUDA | `torch.cuda.manual_seed_all(seed)` | GPU 上的模型推理 |

此外，模块导入时就有 `random.seed(0)` 硬编码：
- `habitat_extensions/task.py:23`
- `vlnce_baselines/env/env_utils.py:10`

多进程运行时：`env_utils.py:142` 会对每个 worker 执行 `task_config.SEED += proc_id`，即 worker N 的 seed = 0 + N。

##### SEED **不**控制 MLLM 回答

LA 和 VA 的 API 调用 (`agent.py`) **没有 seed 参数**：

| 角色 | temperature | 说明 |
|------|-------------|------|
| LA（导航决策、停止判断、TODO-list） | **0.7** | 有随机采样，输出有多样性 |
| VA（bbox 坐标输出） | **0.0** (贪心解码) | 选最高概率 token，理论上确定性，但远程 API 服务端仍可能有微小非确定性 |

所有 `generate()` 调用（`api_openai.py:314`、`api_dashscope.py:344`）只接受 `temperature`，不接受 `seed`。

##### "mean ± std over three seeds" 实际捕捉的变异性来源

论文用不同 SEED (0/1/2) 跑三次，捕获的是：

1. **Episode 评估顺序** — `random.shuffle` 的差异
2. **GroundedSAM 分割** — mask 质量的微小差异，影响地图构建和导航路径
3. **地图 + 局部规划** — NumPy/PyTorch 随机性
4. **MLLM 非确定性** — `temperature=0.7` 导致 LA 决策有多样性；即使是 `temperature=0` 的 VA 调用，远程 API 的浮点计算、量化误差也可能产生微小差异

注意：MLLM 的非确定性**独立于** SEED。即使 SEED 相同，两次运行的结果也可能不完全一致——SEED 只固定了环境端，模型端的随机性不受控。

##### 复现

该仓库内**没有**自动化跨 seed 聚合脚本。要复现 "mean ± std over three seeds"：

```bash
# 分别修改 habitat_extensions/config/objectnav_ovon.yaml 中 SEED 为 0/1/2
# 或用命令行覆盖（如果 habitat 框架支持 opts 覆盖 TASK_CONFIG.SEED）
SEED=0 bash eval_scripts/hm3d_ovon.sh
SEED=1 bash eval_scripts/hm3d_ovon.sh
SEED=2 bash eval_scripts/hm3d_ovon.sh
# 然后手工聚合三次结果，计算每个 episode 的均值 ± 标准差
```

## Rollout V2

出于提升速度、简化prompt设计的考虑，提出新的 ObjNav 导航策略与相应的prompt模板。用 VLM 多轮对话链替代四步全景图→LA→VA→planner 管线。

### 架构

| | V1 | V2 |
|---|---|---|
| 感知 | 全景图 (12步旋转, 4帧) | 单帧 RGB |
| 决策 | LA (全景→方向) + VA (RGB-D→bbox) | VLM 多轮对话链 (6个方法) |
| 规划 | FMM / NavDP / iPlanner | 仅 FMM |
| 停止 | STOP double-check (额外 LA 调用) | is_target_near 直接判断 |

### V2 状态机

```
Loop:
  ├─ 超时/距离/丢失检查 → 清 target
  ├─ 有 target 且未超时 → NAV: FMM 持续导航，不调 VLM
  └─ 无 target → DECIDE:
       ├─ visible → NEAR → STOP
       ├─ visible → FAR → bbox → 存 target → FMM (nav_to_visible=True)
       ├─ possible → NEW → bbox → 存 target → FMM (nav_to_visible=False)
       ├─ possible → REPEAT → TURN_RIGHT×3 (90°)
       └─ not visible/possible → TURN_RIGHT×3 (90°)

ACT 后 guard (仅 nav_to_visible=True 且 ≥1 步后):
  ├─ target lost → 清空 → save debug img → 回 DECIDE
  └─ target near → STOP
```

### 关键设计决策

1. **仅 FMM** — 移除 NavDP/iPlanner
2. **无全景图** — 从单帧 RGB 决策
3. **持久导航** — VLM 设定 waypoint 后 FMM 持续走 (max 15步)
4. **深度回退** — `_v2_bbox_to_target` 投影点不可通行时递减重试
5. **Guard 仅可见目标** — `nav_to_visible=True` 时才检查，possible 探索不浪费 API
6. **90° 旋转** — 单次 TURN_RIGHT×3 (30°×3=90°)，和 v1 LA 转向粒度一致
7. **Bbox 历史** — annotated frame 用于 `is_repeat` 走圈检测
8. **Target name 提取** — "Find the pillow" → "pillow"

### 日志控制（v3 统一预设系统）

从 `env.sh` 统一配置，三层分层：Evaluator / Agent / API。

```bash
# 预设（一键设置所有层级）
export LAVIRA_LOG=quiet    # 仅结果（紧凑进度条）
export LAVIRA_LOG=normal   # prompt + decision（日常 eval）
export LAVIRA_LOG=debug    # 全部输出（网络、body、FMM、完整请求历史）

# 按分类覆盖（优先级高于预设）：
export LAVIRA_LOG_PLAN=1   # Evaluator: branch decisions & NAV steps      [0/1]
export LAVIRA_LOG_FMM=1    # Evaluator: FMM planner output                 [0/1]
export LAVIRA_LOG_ACT=1    # Evaluator: action execution                   [0/1]
export LAVIRA_LOG_REQ=2    # Agent: API request content [0=off,1=增量,2=全量]
export LAVIRA_LOG_RESP=1   # Agent: prompt & response content              [0/1]
export LAVIRA_LOG_BODY=1   # API: HTTP body (image sizes, tokens)          [0/1]
export LAVIRA_LOG_NETWORK=1 # API: HTTP metadata (latency, status)         [0/1]
```

旧变量兼容：`LAVIRA_V2_LOG_*`、`LAVIRA_LOG_PROMPT_OUT`、`LAVIRA_LOG_VERBOSE` 仍可使用，自动映射到新系统并输出 deprecation warning。实现见 `vlnce_baselines/utils/logging.py`。

### 实验结果

#### ep2513 (HM3D-OVON, target: pillow)

| | V1 | V2 |
|---|---|---|
| API 调用 | 8 (5 LA + 3 VA) | 14 |
| API 耗时 | 44s | 31s |
| 总耗时 | 76s | 41s |
| Tokens | 15,929 | 5,936 (-63%) |
| Steps | 58 | 11 |
| Path | 0.9m | 1.5m |
| SPL | 1.0 | 0.94 |
| Success | ✓ | ✓ |

#### ep2469 (HM3D-OVON, target: picture)

| | V1 | V2 |
|---|---|---|
| API 调用 | 18 (10 LA + 8 VA) | 68 |
| API 耗时 | 130s | 144s |
| 总耗时 | 225s | 229s |
| Tokens | 75,654 | 54,830 (-28%) |
| Steps | 166 | 128 |
| Path | 8.2m | 15.3m (+87%) |
| SPL | 0.19 | 0.10 |
| Success | ✓ | ✓ |

V2 行为分解 (68 调用 / 108 DECIDE+NAV 周期):
- `visible → FAR → FMM`: 1 次
- `possible → NEW → FMM`: 12 次 (博运气探索)
- `not visible/possible → TURN_RIGHT`: 9 次 (全盲转 90°)
- REPEAT: 1 次

### V2 速度未达预期的根因分析

#### 核心问题: 调用次数太多，每次都要付网络固定成本

一次 API 调用的完整延迟构成:

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 网络往返 (RTT + TLS + API 网关) | ~1-2s | 每次调用的**固定成本**，与 prompt 大小几乎无关 |
| Prefill (处理输入 token) | ~0.1ms/token | V1 3500 tokens ≈ 350ms, V2 330 tokens ≈ 33ms |
| Decode (生成输出 token) | ~10ms/token | V1 200 tokens ≈ 2s, V2 2 tokens ≈ 20ms |
| 排队 | 不定 | 取决于 API 负载 |

```
V1 单次调用:  ███ 网络 1-2s ██████ prefill 0.35s ██████████ decode 2s ████ 排队
V2 单次调用:  █████████████████████████ 网络 1-2s █ prefill 0.03s █ decode
               ↑ 固定成本占 >95%
```

**V1 的 7.8s 平均耗时不是固定开销，而是重 infer（长 prompt + 长输出）的合理耗时**。V2 把 infer 从 7s 压到了 50ms，但每次调用仍然要付 ~1-2s 的网络往返成本。

#### ep2469 对比拆解

| | V1 | V2 |
|---|---|---|
| 调用次数 | **18** | **68** |
| 每次 infer (prefill+decode) | ~2.5s (3500in + 200out) | ~0.05s (330in + 2out) |
| 每次网络固定成本 | ~1.5s | ~1.5s |
| infer 合计 | 18 × 2.5s ≈ **45s** | 68 × 0.05s ≈ **3s** |
| 网络固定成本合计 | 18 × 1.5s ≈ **27s** | 68 × 1.5s ≈ **102s** |
| 排队/波动 | ~58s | ~39s |
| **API 总耗时** | **130s** | **144s** |

**结论**: V2 把模型推理时间从 45s 压到了 3s（效率提升 15x），但代价是多付了 75s 的网络固定成本（68 次 vs 18 次），净效果是反而慢了 14s。

#### 多轮对话链并未省时

V2 的核心设计假设: 在同一个 conversation 中追加文本，避免重复上传图片，可以节省时间。

但实测表明这个假设不成立:

1. **图像 KV Cache 不跨调用复用**: 根据 SGLang issue #11785 和华为 Ascend-vLLM 文档确认，VLM 多轮对话中**图片的 KV Cache 不会被后续调用复用**。每次新调用都必须从头 prefill 图片 token（~300 tokens），即使图片在首轮已发送过。

2. **Base64 图片在对话历史中累积**: 每次调用时 `self._messages` 包含整个对话链。图片的 base64 只在第一轮发送，但 API 后端仍需重新编码图像。从 V2 日志可见，输入 token 数从 327 逐步增长到 427——增长的是文字内容（assistant 回复 + 新 prompt），但图片 token 每轮都重新计算。

3. **因此多轮链唯一的节省是省去了 ~200KB 图片的网络传输**，但相比 ~2s 的延迟开销，这点传输节省（~100ms）可以忽略。

#### V2 真正的优势

| 优势 | 验证 |
|------|------|
| **Token 消耗大幅降低** | ep2469: 54,830 vs 75,654 (-28%)，ep2513: 5,936 vs 15,929 (-63%) |
| **Prompt 简洁** | 单句 yes/no 或极简 JSON，不需复杂结构化输出 |
| **简单场景极快** | ep2513: 41s vs 76s (-46%)，目标可见时导航效率高 |
| **API 调用轻量** | avg output 4-8 tokens vs 233 tokens |
| **无全景图旋转** | v1 每轮决策需要 12 steps 全景采集，v2 每步都是决策步 |

#### V2 真正的劣势

| 劣势 | 根因 | 是否可修复 |
|------|------|-----------|
| **调用频率过高** | 单帧视野窄，大量步数浪费在旋转探索上 | **是** — 合并 prompt、增加"环视"行为 |
| **调用次数多重复交网络成本** | 每次 ~1.5s 固定网络成本，68 次即 102s | **是** — 需从架构层面减少调用次数 |
| **多轮链无 KV Cache 收益** | VLM 图片 cache 不跨轮复用 | **否** — API 后端限制，不可控 |
| **复杂场景路径长** | 单帧无 360° 感知，TURN_RIGHT 盲转 12-21 steps | **可缓解** — 环视或多帧拼接 |
| **`possible` 分支不可靠** | 模型凭单帧猜测探索方向，准确率有限 | **是** — prompt 或策略可调 |

#### 改进方向

1. **Prompt 合并** (最大收益): 将 is_target_visible + is_target_near + target_bbox 三个调用合并为一个调用。减少 3 次 → 1 次，消除 2 次固定开销。

2. **减少 DECIDE 频率**: 当 `possible → NEW → FMM` 时，nav_to_visible=False 不应该每次都 guard。当前 guard 只对 visible 目标生效，这部分已经做了。下一步可考虑 NAV 阶段不做任何 API 调用。

3. **系统化环视**: 代替盲转 TURN_RIGHT×3（90°），做一次 360° 环视（4 方向各 1 帧），合并 4 帧一次性发给 VLM 判断哪边最可能。这用 1 次多图 API 调用替代多次单帧调用。

4. **提高单次导航步数**: max_steps_to_target 从 15 提到更高（v1 有时单轮 20+ steps），减少 DECIDE 频率。

## Rollout V3

将 V2 改进方向中的 **"Prompt 合并"** 和 **"系统化环视"** 落地实现。核心思路：一次多图 API 调用同时完成场景判断 + bbox 输出，用 4 方向视图替代单帧盲猜，彻底消除 V2 的多轮对话链。

### 架构

| | V2 | V3 |
|---|---|---|
| 感知 | 单帧 RGB | 4 方向视图 (前/右/后/左)，每方向 1 帧 |
| 决策 | 多轮对话链 (6 个 stub 方法，每步最多 6 次调用) | 单次 judge() 调用 → plan + regions |
| 探索 | TURN_RIGHT×3 盲转 90° | judge 识别所有候选区域 → select_one 挑选最佳 |
| 规划 | 仅 FMM | 仅 FMM |
| 停止 | is_target_near 直接判断 | judge 的 A 分支（可见 + 够近） |
| 防重复 | bbox_history_images → is_repeat | came_from_direction 提示 + bbox_history 去重 |

### V3 状态机

```
Loop:
  ├─ 超时/距离检查 → 清 target
  ├─ 有 target → NAV: FMM 持续导航 (max 15 steps)，不调 API
  └─ 无 target → DECIDE:
       ├─ 采集 4 方向视图 (4×TURN_RIGHT + 4×TURN_LEFT + 回到原位)
       ├─ 计算 came_from_direction（避免走回头路）
       ├─ agent_v3.judge(front,right,back,left, came_from) → plan + regions
       ├─ A (STOP): 目标可见且够近 → 保存 bbox → STOP
       ├─ B (APPROACH): 目标可见但远 → 保存 bbox → 设 target → FMM
       ├─ C (EXPLORE):
       │   ├─ 为所有候选区域画 bbox → f_ann
       │   ├─ agent_v3.select_one(f_ann, history) → 挑最佳
       │   ├─ 全已探索 → fail
       │   └─ 选中 → 设 target → FMM
       └─ D (OTHER): 死胡同/无探索区域 → fail

ACT: 执行 action_list 中的动作（FMM / STOP），更新地图
```

### 关键设计决策

1. **单次多图调用** — 4 帧一次性发给 VLM，同时完成场景分类 + bbox 定位。消除 V2 的 3-6 次串行调用，每次 DECIDE 只需 1-2 次 API 调用（judge + 可选 select_one）。

2. **4 方向感知** — 替代 V2 的单帧盲猜。4 张 bbox-annotated 图片让模型能对比选择最优方向，而非随机 TURN_RIGHT。

3. **came_from_direction** — 在 prompt 中告诉模型"Front 方向是来路，避免走回去"，减少原地打转。

4. **两级决策（judge → select_one）** — judge 负责"看见什么"，select_one 负责"去哪"。两者独立调用的好处是：select_one 不需要再看原始帧，只需比较候选 bbox 图片，token 消耗更小。

5. **per-mille 坐标** — bbox 坐标用 0-1000 归一化而非像素值，让 prompt 更简洁。

6. **bbox 历史** — 只保留被选中的 bbox（`_selected`）和 STOP/APPROACH 的 bbox，不被选中的候选不会进入历史（下次仍可考虑）。

7. **仅 FMM** — 和 V2 一样，不做全景图旋转，4 方向采集用 simulator step。

8. **无 STOP double-check** — judge 的 A 分支要求"visible + close enough to reach"，单个模型调用直接决定停止。

### V2 vs V3 对比

| | V2 | V3 |
|---|---|---|
| 每步 DECIDE 调用次数 | 1-6 次 (visible/near/bbox/possible/possible_bbox/repeat) | 1-2 次 (judge + 可选 select_one) |
| 每次调用图片数 | 1 帧 | 4 帧 (judge) / N 候选 bbox (select_one) |
| 探索策略 | 盲转 90° + 单帧猜测 | 4 方向对比 + 模型主动选区域 |
| 防回头 | is_repeat 走圈检测 | came_from_direction 提示 |
| 决策类型 | 6 个 yes/no + bbox stub | 1 个 A/B/C/D 分类 + bbox JSON |
| 停止判断 | is_target_near | judge A 分支 |
| 失败处理 | 盲转到底 | D 分支 / select_one 全探索 → 直接 fail |

## Rollout V4

将 V3 的 judge + select_one 两步合并为单步 decide 调用，模型直接输出最终决策 + 单个最优 bbox，极致精简 prompt。

### 架构

| | V3 | V4 |
|---|---|---|
| 决策调用 | 1-2 次 (judge + 可选 select_one) | **1 次** decide() |
| 输出格式 | letter + JSON regions | **单行 CSV** |
| 请求体 | 4 图 + JSON prompt (~40 行) | 4 图 + 简洁 prompt (~10 行) |
| 探索策略 | judge 返回所有候选 → select_one 挑最优 | **模型直接挑最优** |
| 探索历史 | 传给 select_one 去重 | **作为消息上下文嵌入** |

### V4 状态机

```
Loop:
  ├─ 超时/距离/卡住检查 → 清 target
  ├─ 有 target → NAV: FMM 持续导航
  └─ 无 target → DECIDE:
       ├─ 采集 4 方向视图 (前/右/后/左，12×TURN_RIGHT)
       ├─ agent_v4.decide() → plan + frame_idx + bbox_px
       ├─ A (STOP): 目标可见且够近 → STOP
       ├─ B (APPROACH): 目标可见但远 → 设 target → FMM
       ├─ C (EXPLORE): 有未探索区域 → 设 target → FMM
       └─ D (OTHER): 死胡同/重复探索 → fail

ACT: FMM → 更新地图
```

### 返回格式

```
A,front,100,200,300,400  → STOP，前方有目标
B,right,150,0,267,133    → APPROACH，右方有目标
C,left,0,240,234,480     → EXPLORE，左方有探索区域
D                        → OTHER
D,1                       → OTHER (重复历史图 #1)
```

### FMM 导航优化

| 机制 | 逻辑 |
|------|------|
| **最大步数** | `max(10, min(fmm_d / step_px × 3, 80))`，由 FMM 测地距离动态决定 |
| **抵达判定** | APPROACH: `max(欧氏, FMM) < 0.8m`；EXPLORE: `fmm ≤ max(3, original×10%)` |
| **卡住检测** | FMM 连续 `max(12, max_steps//3)` 步改善不足 1 FWD 步 → 放弃 |
| **目标回退** | bbox 反投影点不可通行时，未探索区域乐观接受，已知障碍拉近深度重试 |

### 可视化

- combined 图底部由 V1 文本分析区改为 **bbox 标注图**（保持宽高比）
- panorama 阶段 bbox 区域显示 "scanning"
- panorama 时机器人圆心空心，历史目标点缩小
- bbox 图上叠加类型标签（STOP/APPROACH/EXPLORE）