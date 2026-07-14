# 个人操作笔记

## API 准备

- 先申请好 API 密钥，保存到环境变量中。
- 网络代理可能影响远程调用模型 API，注意排查。
- 如果访问阿里云 API 有问题，用以下命令对比测试（遇到过 IPv6 的坑）：

  ```bash
  python hello_qwen.py
  python hello_qwen.py --force-ipv4
  ```

- `test_api.py` 可以测试更多厂商和模型的 API，例如 DeepSeek 的 `deepseek-V4-pro`。

## 可视化

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

### HM3D-v2

| Episode | 结果 | 备注 |
|---------|------|------|
| 0       | ✅ ok | 多次测试，路线有多样性 |

### HM3D-OVON

| Episode | 结果 | 备注 |
|---------|------|------|
| 53      | ❌ fail | 探索效率低，在第一个房间有点打转；最终把用别的东西装的一簇花当成花瓶了 |
| 2469    | ✅ ok | 看上去正常 |
