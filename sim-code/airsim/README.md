# Uni-LaViRA for TravelUAV

Aerial-VLN evaluation of **Uni-LaViRA** on the TravelUAV / AirSim simulator: the same training-free framework that *translates* instructions **language → vision → robot action**, here flying a UAV through 3D navigation with no trajectory training.

Only the Uni-LaViRA evaluation path is kept in this release:

- `scripts/unilavira_eval.sh`
- `unilavira_evaluator.py`
- `src/model_wrapper/unilavira_model.py`

## Environment and Dataset

For environment setup, dependency installation, simulator preparation, and dataset download, please refer to the official TravelUAV repository:

- https://github.com/buaa-colalab/TravelUAV

In particular, please follow the corresponding instructions in the official repository for:

- dependency installation
- simulator environment setup
- dataset preparation and download
- grounding model assets and related paths

## Run

After completing the TravelUAV environment and dataset setup, set the model credentials (Vision-Action and Language-Action models; see `.env.example`) and run:

```bash
export VA_API_KEY=... VA_BASE_URL=... VA_MODEL_NAME=...   # vision grounding (e.g. qwen3.5-27b)
export LA_API_KEY=... LA_BASE_URL=... LA_MODEL_NAME=...   # language reasoning (e.g. gemini-3.5-flash)
bash scripts/unilavira_eval.sh
```

## Notes

- This repository is a trimmed release and does not include other previously used baselines or auxiliary scripts.
- Before running, please check the paths in `scripts/unilavira_eval.sh` and update them to match your local environment if needed.

## Citation

```bibtex
@article{ding2026unilavira,
  title   = {Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation},
  author  = {Ding, Hongyu and Zhang, Sizhuo and Xu, Ziming and Guo, Jinwen and Liu, Hongxiu and Cheng, Xingzhi and Chen, Zixuan and Qi, Haifei and Wang, Duo and Xu, Hao and Shi, Jieqi and Zhang, Yifan and Huo, Jing and Cheng, Jian and Gao, Yang and Luo, Jiebo},
  journal = {arXiv preprint arXiv:2605.27582},
  year    = {2026}
}
@article{ding2025lavira,
  title   = {LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision Language Navigation in Continuous Environments},
  author  = {Ding, Hongyu and Xu, Ziming and Fang, Yudong and Wu, You and Chen, Zixuan and Shi, Jieqi and Huo, Jing and Zhang, Yifan and Gao, Yang},
  journal = {arXiv preprint arXiv:2510.19655},
  year    = {2025}
}
```
