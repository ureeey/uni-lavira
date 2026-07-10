# 硬件环境

本项目实际运行在个人笔记本电脑 Dell G15 5530 上。

## 主机

| 项目 | 实际规格 |
|------|------|
| 型号 | Dell G15 5530 |
| CPU | Intel Core i7-13650HX (13th Gen Raptor Lake-HX) |
| CPU 架构 | 14 核 20 线程 (6P + 8E)，最高 4.90 GHz |
| 内存 | 16 GB DDR5 4800 MHz |
| GPU | NVIDIA GeForce RTX 4060 Laptop |
| 显存 | 8 GB GDDR6 (8188 MiB) |
| CUDA 能力 | sm_89 (Ada Lovelace) |
| 硬盘 | 457 GB NVMe SSD (PCIe 4.0) |
| 操作系统 | Ubuntu-based Linux, kernel 6.0.0-1020-oem |

## 端口与外设

- USB-C 3.2 Gen 2 (支持 DisplayPort Alt-Mode)
- 3 × USB 3.2 Gen 1 Type-A
- HDMI 2.1
- RJ-45 千兆以太网
- Wi-Fi 6 AX201 + Bluetooth 5.2

## 对开发的影响

- **显存 8GB 是强约束**: 无法同时加载大模型和 Habitat 仿真。Habitat-Sim 渲染 + GroundedSAM + NavDP 同时驻留显存时极易 OOM。
- **仿真评估时不能跑本地模型**: 在 Docker/conda 中运行 Habitat 评估脚本时，LA/VA 模型必须走远程 API（`LA_API_KEY` / `VA_API_KEY`），不能在本地同时跑 `llama-server`。
- **真实机器人部署时可以本地跑模型**: `cobot_magic` / `unitree_g1` / `unitree_go1` 不运行 Habitat-Sim，8GB 显存可以加载 `Qwen3.5-27B-Q4_K_M` GGUF（量化后约 16GB 系统内存占用 + ~6GB 显存），但 `--n-gpu-layers` 需根据实际显存余量调整，避免 OOM。
- **Docker 构建**: `Dockerfile` 中 `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"` 已覆盖 sm_89，本地编译 habitat-sim CUDA 扩展没问题。
- **不适用于大规模训练**: 该 GPU 仅用于推理/评估，任何需要训练的组件应提交到远程集群。
- **内存 16GB 偏紧**: 同时运行 Habitat 仿真 + Python 推理 + 系统开销可能接近上限。量化模型（GGUF）使用时需要权衡 GPU offload 层数和系统内存占用。
