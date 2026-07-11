# TODO — Uni-LaViRA 复现待办事项

> 硬件: Dell G15 5530 (RTX 4060 8GB, 16GB RAM)
> 数据盘: `/media/jcy/6f8bd02b-5080-9f4b-ad42-598fc747eda6/` (269GB 可用)

---

## 阶段 0: 并行等待项（不阻塞主流程，可随时做）

### 0.1 MP3D 数据集访问权限

- [ ] 访问 https://niessner.github.io/Matterport/ 填写申请表
  - 审核需数日，尽早提交
  - 用途：VLN-CE R2R/RxR + MP3D-EQA（~200GB）
  - 审核通过后获取 `download_mp.py`

### 0.2 LLM API Key

- [x] LA — DeepSeek（`https://api.deepseek.com`，模型 `deepseek-v4-pro`）✅
- [x] VA — 阿里云 MaaS Qwen（`qwen3.6-plus`）✅
- [x] API key 已填入 `sim-code/habitat/.env` ✅
  - LA: deepseek-v4-pro @ api.deepseek.com
  - VA: qwen3.6-plus @ Aliyun MaaS (北京)
  - 详见 [[API.md]]

---

## 阶段 1: 编译环境搭建 ✅ 已完成

> 实际安装方式与计划略有不同：habitat-sim 使用 conda (aihabitat channel) 预编译包，GroundingDINO 使用 conda-forge 预编译包，避免源码编译的依赖问题。

### 1.1 conda 环境 ✅

- [x] `lavira` conda 环境 (Python 3.8.20)
- [x] CUDA 11.8 toolkit (conda: `cudatoolkit=11.8` + `cuda-nvcc=11.8.89`)

### 1.2 habitat-sim 0.1.7 ✅

- [x] conda install from `aihabitat` channel: `habitat-sim=0.1.7` (headless, bullet, with-cuda)
- [x] 预编译包，无需源码编译

### 1.3 PyTorch ✅

- [x] PyTorch 2.4.1+cu118 + torchvision 0.19.1+cu118 + torchaudio 2.4.1+cu118

### 1.4 habitat-lab 0.1.7 ✅

- [x] 从源码 `setup.py develop --all` 安装（v0.1.7 tag）
- [x] 已移除 `habitat_baselines/rl/requirements.txt` 中的 tensorflow==1.13.1

### 1.5 GroundingDINO + SAM ✅

- [x] groundingdino-py 0.4.0 (conda-forge)
- [x] segment-anything 1.0 (conda-forge)
- [x] `phrases2classes` 已替换为编辑距离版本
- [x] nltk 已安装

### 1.6 pip 依赖 ✅

- [x] `pip install -r requirements.txt`（scikit-fmm 通过 conda 安装）
- [x] NavDP 依赖: diffusers==0.33.1

### 关键包版本

| 包 | 版本 | 来源 |
|------|------|------|
| torch | 2.4.1+cu118 | pip |
| habitat-sim | 0.1.7 | conda (aihabitat) |
| habitat-lab | 0.1.7 | 源码 develop |
| groundingdino-py | 0.4.0 | conda-forge |
| segment-anything | 1.0 | conda-forge |
| transformers | 4.46.3 | pip |
| openai | 2.2.0 | pip |
| opencv-python | 4.12.0 | pip |
| numpy | 1.23.5 | pip |
| diffusers | 0.33.1 | pip (NavDP) |

### ⚠️ 已知问题

- GroundingDINO C++ CUDA 扩展未编译（conda-forge 包不含），CPU fallback 运行
- `tb-nightly` 缺失（不影响核心导航，仅 TensorBoard 可视化需要）

---

## 阶段 2: 数据准备（依赖阶段 1 完成）

> 数据统一存放到外接硬盘: `/media/jcy/6f8bd02b-5080-9f4b-ad42-598fc747eda6/`

### 2.0 创建数据目录 + 软链接

- [x] 在外接硬盘创建目录结构 ✅
  ```bash
  DATA_ROOT=/media/jcy/6f8bd02b-5080-9f4b-ad42-598fc747eda6/lavira-data
  mkdir -p $DATA_ROOT/{scene_datasets,grounded_sam,datasets/episodes}
  ```
- [x] `sim-code/habitat/data` → 指向外接硬盘的软链接 ✅
- [x] 已有数据（GroundedSAM 权重、bert、episodes）已迁移到外接硬盘 ✅

### 2.1 场景数据集

- [x] **HM3D API Token** — 已获取 ✅
- [x] **HM3D val split** — 102 场景, 5.3GB，下载+解压已完成 ✅
  ```bash
  cd sim-code/habitat
  conda activate lavira
  python -m habitat_sim.utils.datasets_download \
      --username <TOKEN_ID> \
      --password <TOKEN_SECRET> \
      --uids hm3d_val_v0.2 \
      --data-path data/scene_datasets/hm3d
  ```
- [ ] **MP3D 场景**（等阶段 0.1 审核通过，~200GB）
  ```bash
  python download_mp.py --task habitat -o data/scene_datasets/mp3d/
  ```

### 2.2 模型权重

- [x] **GroundedSAM 权重** (~3.2GB) ✅
  - `groundingdino_swint_ogc.pth` (662MB)
  - `GroundingDINO_SwinT_OGC.py` (1KB)
  - `repvit_sam.pt` (105MB)
  - `sam_vit_h_4b8939.pth` (2.4GB)
  - 存放位置: `data/grounded_sam/`

- [x] **BERT 模型** ✅
  - `data/grounded_sam/bert-base-uncased/` 已通过 huggingface_hub 下载

- [ ] **NavDP checkpoint** ⚠️ 需手动填表
  - 填表地址: https://docs.google.com/forms/d/e/1FAIpQLSdl3RvajO5AohwWZL5C0yM-gkSqrNaLGp1OzN9oF24oNLfikw/viewform
  - 放入 `sim-code/habitat/navdp/navdp-cross-modal.ckpt`

### 2.3 Episode 数据

- [x] Episode JSON 包 ✅（7 个文件已解压到 `data/datasets/episodes/`）
- [x] `stratified_samples/`（100-episode ID 列表，随代码发布）

---

## 阶段 3: 配置与验证（依赖阶段 1 + 2 全部完成）

### 3.1 创建 .env 文件

- [x] `sim-code/habitat/.env` 已配置 ✅
  ```bash
  # LA — DeepSeek
  LA_API_KEY=sk-xxx
  LA_BASE_URL=https://api.deepseek.com
  LA_MODEL_NAME=deepseek-v4-pro

  # VA — Aliyun MaaS (Qwen)
  VA_API_KEY=sk-xxx
  VA_BASE_URL=https://ws-gs1ofh9fdpuo75kq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
  VA_MODEL_NAME=qwen3.6-plus
  ```
  - ⚠️ 运行前需 `unset HTTP_PROXY HTTPS_PROXY`，httpx 库不支持 socks 代理

### 3.2 最小测试（1 个 episode）

```bash
cd sim-code/habitat
conda activate lavira
export NPROC=1
python run_mp.py \
  --exp-name test \
  --run-type eval \
  --exp-config vlnce_baselines/config/objectnav_v2.yaml \
  --nprocesses 1 \
  --debug-episodes "<HM3D_EPISODE_ID>" \
  TRAINER_NAME ZS-Evaluator-mp \
  TORCH_GPU_IDS [0] NUM_ENVIRONMENTS 1
```

### 3.3 完整测试

- [ ] **HM3D-v2 ObjectNav（100 集，约 1 小时）**
  ```bash
  bash eval_scripts/hm3d_v2.sh
  ```
- [ ] **HM3D-OVON**
  ```bash
  bash eval_scripts/hm3d_ovon.sh
  ```
- [ ] **VLN-CE R2R**（等 MP3D 就绪）
  ```bash
  bash eval_scripts/vlnce_r2r.sh
  ```
- [ ] **VLN-CE RxR**（等 MP3D 就绪）
  ```bash
  bash eval_scripts/vlnce_rxr.sh
  ```
- [ ] **MP3D-EQA**（等 MP3D 就绪）
  ```bash
  bash eval_scripts/mp3d_eqa.sh
  ```

---

## 阶段 4: 结果分析

- [ ] 查看统计文件 `data/checkpoints/<exp_name>/stats_ep_ckpt_*.json`
- [ ] 运行可视化服务 `python server.py` → 浏览 `http://localhost:9999`
- [ ] 对比论文上报指标

---

## 依赖关系

```
阶段 0 (并行等待) ──→ 不影响主流程，做完随时插入
    MP3D 申请 ─────────→ 2.1 MP3D 下载 ──→ 3.3 VLN/EQA 测试
    API key 获取 ──────→ 3.1 .env 配置

阶段 1 (编译环境) ★ 必须先做
    1.1 conda env → 1.2 系统依赖 → 1.3 habitat-sim → 1.4 PyTorch
    → 1.5 habitat-lab → 1.6 GroundingDINO+SAM → 1.7 pip deps
        │
        ▼
阶段 2 (数据下载) 依赖 habitat-sim
    2.0 目录+软链接 → 2.1 HM3D 场景 → 2.2 模型权重 → 2.3 episode JSON
        │
        ▼
阶段 3 (运行) 需要环境 + 数据 + API 全部就绪
    3.1 .env → 3.2 单集测试 → 3.3 全量测试 → 4 结果分析
```

> **关键变更**: HM3D 下载命令 `python -m habitat_sim.utils.datasets_download` 需要 habitat-sim 已安装，所以阶段 1 必须排在阶段 2 之前。MP3D 申请和 API key 获取可与阶段 1 并行进行。
