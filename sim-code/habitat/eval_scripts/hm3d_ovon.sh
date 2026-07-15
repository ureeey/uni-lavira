#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export BERT_LOCAL_PATH=data/grounded_sam/bert-base-uncased
export TOKENIZERS_PARALLELISM=false
export GLOG_minloglevel=0
export MAGNUM_LOG=verbose
# NOTE: Commented out because this habitat-sim build was compiled without --headless EGL support.
# It needs X11/GLX rendering (DISPLAY must stay set).
# export EGL_PLATFORM=surfaceless
# unset DISPLAY

TIMESTAMP=$(date +"%m%d-%H%M%S")
mkdir -p logs
LOG_FILE="logs/hm3d-ovon-${TIMESTAMP}.log"

flag="--exp-name hm3d-ovon-${TIMESTAMP}
      --run-type eval
      --exp-config vlnce_baselines/config/objectnav_ovon.yaml
      --nprocesses ${NPROC:-1}
      --episode-file data/datasets/stratified_samples/hm3d_ovon.json
      NUM_ENVIRONMENTS 1
      TRAINER_NAME ZS-Evaluator-mp
      TORCH_GPU_IDS [0]
      SIMULATOR_GPU_IDS [0]
      "

echo "Starting experiment: hm3d-ovon-${TIMESTAMP}"
echo "Logging to: hm3d-ovon-${LOG_FILE}"

python run_mp.py $flag "$@" 2>&1 | tee "${LOG_FILE}"
