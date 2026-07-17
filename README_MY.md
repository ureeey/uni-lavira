# 个人操作笔记

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

## 可视化

> 运行前请先确保目录下（例如saved_rgb_images/test-ovon/2518）没有以前测试残留的图片。
```bash
python watch_viz.py --auto
```

## 单条测试

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