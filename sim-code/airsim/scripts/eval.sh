#!/bin/bash
# change the dataset_path to your own path

root_dir=. # TravelUAV directory
model_dir=$root_dir/Model/LLaMA-UAV


CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/vlnce_src/eval.py \
    --run_type eval \
    --name TravelLLM \
    --gpu_id 0 \
    --simulator_tool_port 30000 \
    --DDP_MASTER_PORT 29500 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --dataset_path path/to/OpenUAV_data/TravelUAV \
    --eval_save_path path/to/OpenUAV_data/eval_runs/closeloop_eval_test \
    --eval_json_path path/to/OpenUAV_data//TravelUAV_data_json/data/uav_dataset/seen_valset.json \
    --model_path $model_dir/work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32 \
    --model_base $model_dir/model_zoo/vicuna-7b-v1.5 \
    --traj_model_path $model_dir/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4 \
    --vision_tower $model_dir/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor $model_dir/llamavid/processor/clip-patch14-224 \
    --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth \
    --map_spawn_area_json_path path/to/OpenUAV_data/TravelUAV_data_json/data/meta/map_spawnarea_info.json \
    --object_name_json_path path/to/OpenUAV_data/TravelUAV_data_json/data/meta/object_description.json