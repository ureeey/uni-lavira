#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export BERT_LOCAL_PATH=data/grounded_sam/bert-base-uncased
export TOKENIZERS_PARALLELISM=false
export GLOG_minloglevel=0
export MAGNUM_LOG=verbose
export EGL_PLATFORM=surfaceless
unset DISPLAY

TIMESTAMP=$(date +"%m%d-%H%M%S")
mkdir -p logs
LOG_FILE="logs/${TIMESTAMP}.log"

flag="--exp-name ${TIMESTAMP}
      --run-type eval
      --exp-config vlnce_baselines/config/r2r.yaml
      --nprocesses ${NPROC:-20}
      --use-navdp
      --episode-file data/datasets/stratified_samples/vlnce_r2r.json
      NUM_ENVIRONMENTS 1
      TRAINER_NAME ZS-Evaluator-mp
      TORCH_GPU_IDS [0]
      SIMULATOR_GPU_IDS [0]
      "

echo "Starting experiment: ${TIMESTAMP}"
echo "Logging to: ${LOG_FILE}"

python run_mp.py $flag "$@" 2>&1 | tee "${LOG_FILE}"
