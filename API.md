# API 信息

本项目需要以下 API，均通过环境变量 `.env` 配置。

## LA — DeepSeek（语言推理）

| 项目 | 值 |
|------|-----|
| 用途 | 语言代理，分析全景图描述、维护 TODO 列表、决定导航方向 |
| 提供商 | DeepSeek |
| Base URL | `https://api.deepseek.com` |
| 模型 | `deepseek-v4-pro`（当前使用） |
| 备选模型 | `deepseek-v4-flash`（更快更便宜） |
| 申请地址 | https://platform.deepseek.com/ |
| 环境变量 | `LA_API_KEY` / `LA_BASE_URL` / `LA_MODEL_NAME` |
| 兼容性 | OpenAI 兼容接口，代码无需修改 |

## VA — Qwen（视觉定位）

| 项目 | 值 |
|------|-----|
| 用途 | 视觉代理，接收 RGB 图像、返回下一路径点的边界框坐标 |
| 提供商 | 阿里云 Model-as-a-Service (MaaS) |
| Base URL | `https://ws-gs1ofh9fdpuo75kq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` |
| 模型 | `qwen3.6-plus`（当前使用） |
| 环境变量 | `VA_API_KEY` / `VA_BASE_URL` / `VA_MODEL_NAME` |
| 兼容性 | OpenAI 兼容接口，代码无需修改 |

### ⚠️ 代理注意事项

如果系统设置了 socks 代理（如 `socks://127.0.0.1:7890`），httpx 库不支持 socks 协议，运行评估前需：

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
```

## HM3D / Matterport

| 项目 | 值 |
|------|-----|
| 用途 | 下载 HM3D-Semantics 场景数据集（ObjectNav 用） |
| Token 获取 | `my.matterport.com/settings/account/devtools` → Generate API Token |
| Token 格式 | Token ID（作为 username）+ Token Secret（作为 password） |
| 配置方式 | ⚠️ **不写入 .env**，仅在下方的下载命令中使用一次 |
| ⚠️ 已知 bug | "Access the Dataset" 按钮可能循环重定向到 GitHub，需发邮件给 `developer@matterport.com` 申请手动开通 |
| 申请地址 | https://aihabitat.org/datasets/hm3d/ |

### 下载命令

在 `sim-code/habitat/` 目录下执行（需先 `conda activate lavira`）：

```bash
python -m habitat_sim.utils.datasets_download \
    --username <TOKEN_ID> \
    --password <TOKEN_SECRET> \
    --uids hm3d_val_v0.2 \
    --data-path data/scene_datasets/hm3d
```

可选 UID：

| UID | 说明 |
|-----|------|
| `hm3d_val_v0.2` | 验证集（100 个场景，~20GB） |
| `hm3d_train_v0.2` | 训练集 |
| `hm3d_example_v0.2` | 示例场景 |

## MP3D / Matterport

| 项目 | 值 |
|------|-----|
| 用途 | 下载 Matterport3D 场景数据集（VLN R2R/RxR、EQA 用） |
| 获取方式 | 访问 https://niessner.github.io/Matterport/ 填写申请表，审核通过后获取 `download_mp.py` |
| 配置方式 | ⚠️ **不写入 .env**，`download_mp.py` 内置认证流程 |
| ⚠️ 注意 | 审核需数日；数据集约 200GB；无公开镜像 |

### 下载命令（审核通过后）

```bash
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

## .env 模板

```bash
# LA — DeepSeek
LA_API_KEY=sk-<your-deepseek-key>
LA_BASE_URL=https://api.deepseek.com
LA_MODEL_NAME=deepseek-v4-pro

# VA — Aliyun MaaS
VA_API_KEY=sk-<your-dashscope-key>
VA_BASE_URL=https://ws-gs1ofh9fdpuo75kq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
VA_MODEL_NAME=qwen3.6-plus

# 离线模式
HF_HUB_OFFLINE=1
BERT_LOCAL_PATH=data/grounded_sam/bert-base-uncased
```
