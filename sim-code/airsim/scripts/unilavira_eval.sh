#!/bin/bash
# Uni-LaViRA evaluation script for TravelUAV
# Combines Uni-LaViRA perception/decision with TravelUAV execution
# Usage: bash scripts/unilavira_eval.sh [PORT] [JSON_PATH]
# Example: bash scripts/unilavira_eval.sh 30000 path/to/unseen_valset_30000.json

root_dir=. # TravelUAV directory

PORT=${1:-30000}

JSON_PATH=${2:-path/to/OpenUAV_data/TravelUAV_data_json/data/uav_dataset/unseen_valset_100_balanced.json}

export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "Uni-LaViRA Evaluation for TravelUAV"
echo "========================================"
echo "Using Uni-LaViRA perception + VLM inference"
echo "Using TravelUAV AirSim execution"
echo "Port: $PORT"
echo "Dataset: $JSON_PATH"
echo "========================================"

# Run evaluation
python -u $root_dir/unilavira_evaluator.py \
    --use_gt False \
    --always_help False \
    --run_type eval \
    --name Uni-LaViRA \
    --simulator_tool_port $PORT \
    --batchSize 1 \
    --maxWaypoints 70 \
    --dataset_path path/to/OpenUAV_data/TravelUAV \
    --eval_save_path path/to/OpenUAV_data/eval_runs/outcome/12 \
    --eval_json_path $JSON_PATH \
    --map_spawn_area_json_path path/to/OpenUAV_data/TravelUAV_data_json/data/meta/map_spawnarea_info.json \
    --object_name_json_path path/to/OpenUAV_data/TravelUAV_data_json/data/meta/object_description.json \
    --groundingdino_config $root_dir/data/grounded_sam/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path $root_dir/data/grounded_sam/groundingdino_swint_ogc.pth

echo "========================================"
echo "Evaluation completed!"
echo "========================================"
