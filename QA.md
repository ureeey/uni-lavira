# Q&A

## Habitat-Sim 中是否指定了不同本体类型？

**问**: 这个项目中使用 habitat-sim 时有没有指定不同本体类型？

**答**: 没有。所有 5 个 Habitat 仿真任务（VLN R2R、VLN RxR、HM3D-v2、HM3D-OVON、MP3D-EQA）使用完全相同的 agent 配置：

| 参数 | 值 |
|------|-----|
| `AGENT_0.HEIGHT` | 0.88 m |
| `AGENT_0.RADIUS` | 0.1 m |
| `FORWARD_STEP_SIZE` | 0.25 m |
| `TURN_ANGLE` | 30° |
| `ACTION_SPACE_CONFIG` | v1 |

5 个 `habitat_extensions/config/*.yaml` 文件中这些参数完全一致，代码中也只有 `AGENT_0`，没有任何按任务类型动态切换 agent 参数的逻辑。`AGENT_HEIGHT` 只在 `ZS_Evaluator_mp.py:136` 被读取一次，传给 `Semantic_Mapping` 模块做坐标变换，所有任务共用。

**Habitat 仿真侧统一本体，真实世界侧多本体适配。** 论文的核心论点"一套架构覆盖四个异构机器人"主要通过 `real-world-code/` 中四个截然不同的机器人平台来体现（轮式双臂 Cobot Magic、人形 Unitree G1、四足 Unitree Go1、四旋翼自研 UAV），而非在 Habitat 仿真中切换 agent 参数。真实机器人只需换底层控制器，LA/VA 推理流水线完全共用。

---

## VA API 如何使用 Qwen-VL？

**问**: VA API 用 Qwen-VL，具体如何申请和配置？

**答**: Qwen-VL 通过阿里云 DashScope（百炼平台）提供 OpenAI 兼容 API，本项目代码无需修改即可对接。

### 申请步骤

1. 访问 https://bailian.console.alibabacloud.com/
2. 登录阿里云账号，开通"百炼"模型服务
3. 左侧导航 → **API 密钥管理** → 创建密钥 → 得到 `sk-xxxxxxxxxxxxxx`
4. 新用户有免费额度：100 万输入 + 100 万输出 tokens（90 天有效）

### Base URL

| 区域 | URL |
|------|-----|
| 国内（北京） | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 国际（新加坡） | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |

> API Key 和 Base URL 区域必须一致。

### .env 配置

```bash
# LA — DeepSeek（纯文本推理）
LA_API_KEY=sk-<deepseek-key>
LA_BASE_URL=https://api.deepseek.com
LA_MODEL_NAME=deepseek-v4-pro

# VA — Qwen-VL（多模态视觉定位）
VA_API_KEY=sk-<dashscope-key>
VA_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VA_MODEL_NAME=qwen3.6-plus
```

### 兼容性

本项目 `LaViRA_API` 使用 `openai.OpenAI()` 标准客户端，DashScope 完全兼容。代码中 `enable_thinking=False` 是 Qwen 原生支持的参数；`reasoning_effort='low'` Qwen-VL 不识别但会静默忽略，不影响运行。

### 代理兼容性问题

**问**: 运行时 httpx/openai 报错 `Unknown scheme for proxy URL URL('socks://...')`？

**答**: httpx 库不支持 socks 代理。运行评估脚本前需要：

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
```

### 数据存放位置

**问**: 数据应该放在哪里？

**答**: 当前配置：外接硬盘 `/media/jcy/6f8bd02b-5080-9f4b-ad42-598fc747eda6/lavira-data/`，通过软链接 `sim-code/habitat/data -> 外接硬盘` 访问。577GB 总量，可用 269GB。

### 运行 VLN-CE R2R 需要 MP3D 还是 HM3D？

**问**: 跑 vln-ce r2r 需要 MP3D 还是 HM3D？

**答**: **需要 MP3D。** R2R 和 RxR 的 episode 都是在 Matterport3D 场景上定义的（90 个室内场景）。HM3D 仅用于 ObjectNav 任务。

### MP3D 有没有公开镜像？

**问**: 网络上有没有现成的分享的 MP3D 镜像？

**答**: **没有。** Matterport 许可协议禁止第三方再分发。唯一的获取途径是向 https://niessner.github.io/Matterport/ 申请授权后通过 `download_mp.py` 下载。替代方案：先跑 HM3D-v2 ObjectNav（仅需注册 API token，即时通过，~20GB）。
