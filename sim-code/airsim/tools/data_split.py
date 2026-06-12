#!/usr/bin/env python3

import json
import random
import numpy as np
import os




input_file = 'path/to/OpenUAV_data/TravelUAV_data_json/data/uav_dataset/unseen_valset.json'
output_dir = 'path/to/OpenUAV_data/TravelUAV_data_json/data/uav_dataset/'
dataset_path = 'path/to/OpenUAV_data/TravelUAV'

EASY_HARD_THRESHOLD = 250  


def calc_trajectory_length(item):
    merged_json_path = os.path.join(dataset_path, item['json'])
    with open(merged_json_path, 'r') as f:
        merged_data = json.load(f)
    traj = merged_data['trajectory_raw_detailed']
    length = sum(
        np.linalg.norm(np.array(traj[i+1]['position']) - np.array(traj[i]['position']))
        for i in range(len(traj) - 1)
    )
    return length


def report_difficulty(items, label=""):
    easy, hard = 0, 0
    for item in items:
        if item['_traj_length'] <= EASY_HARD_THRESHOLD:
            easy += 1
        else:
            hard += 1
    total = easy + hard
    print(f"  {label} difficulty distribution: Easy={easy} ({easy/total*100:.1f}%), Hard={hard} ({hard/total*100:.1f}%)")

print("Loadingunseen_valset.json...")
with open(input_file, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)}  items")


frame1_data = [item for item in data if item.get('frame') == 1]
print(f"Frame=1 items: {len(frame1_data)}  items")


maps = {}
for item in frame1_data:
    map_name = item['json'].split('/')[0]
    if map_name not in maps:
        maps[map_name] = []
    maps[map_name].append(item)

print(f"\nMap statistics:")
for map_name, items in maps.items():
    print(f"  {map_name}: {len(items)}  trajectories")


print(f"\nComputing trajectory length...")
for item in frame1_data:
    item['_traj_length'] = calc_trajectory_length(item)


print(f"\nAll frame=1 data({len(frame1_data)} items):")
report_difficulty(frame1_data, "All")


total_needed = 100
total_population = len(frame1_data)


strata = {}
for item in frame1_data:
    map_name = item['json'].split('/')[0]
    difficulty = 'easy' if item['_traj_length'] <= EASY_HARD_THRESHOLD else 'hard'
    key = (map_name, difficulty)
    if key not in strata:
        strata[key] = []
    strata[key].append(item)


stratum_counts = {}
for key, items in strata.items():
    stratum_counts[key] = len(items) / total_population * total_needed


int_counts = {k: int(v) for k, v in stratum_counts.items()}
remainders = {k: stratum_counts[k] - int_counts[k] for k in stratum_counts}
deficit = total_needed - sum(int_counts.values())
for k in sorted(remainders, key=remainders.get, reverse=True)[:deficit]:
    int_counts[k] += 1

print(f"\nDouble stratified sampling (Scene×Difficulty), total needed {total_needed}  items:")
print(f"  {'Scene':<25s} {'Difficulty':<6s} {'All':>5s} {'Selected':>5s}")
print(f"  {'-'*45}")

selected_data = []
for key in sorted(int_counts.keys()):
    map_name, difficulty = key
    pool = strata[key]
    n = int_counts[key]
    if n > len(pool):
        print(f"  Warning: {map_name}/{difficulty} only has{len(pool)} items, needs{n} items, selecting all")
        n = len(pool)
    sampled = random.sample(pool, n)
    selected_data.extend(sampled)
    print(f"  {map_name:<25s} {difficulty:<6s} {len(pool):>5d} {n:>5d}")

random.shuffle(selected_data)


selected_trajs = [item['json'] for item in selected_data]
assert len(selected_trajs) == len(set(selected_trajs)),\
    f"data contains duplicates! {len(selected_trajs)} items contain {len(selected_trajs) - len(set(selected_trajs))}  duplicated items"

print(f"  {'-'*45}")
print(f"  {'Total':<32s} {total_population:>5d} {len(selected_data):>5d}")


print(f"\nSelected {len(selected_data)} items:")
report_difficulty(selected_data, "Selected")


splits = {
    f'3000{i}': selected_data[i*10:(i+1)*10]
    for i in range(10)
}


all_trajs = []
for port, split_data in splits.items():
    trajs = set(item['json'] for item in split_data)
    assert len(trajs) == len(split_data), f"split{port}contains duplicates internally!"
    all_trajs.extend(trajs)
assert len(all_trajs) == len(set(all_trajs)), "duplicate data exists across splits!"
print(f"\nValidation passed: 10splits are mutually unique, total {len(all_trajs)}unique trajectories")


for port, split_data in splits.items():
    output_file = f"{output_dir}unseen_valset_{port}.json"

    
    save_data = [{k: v for k, v in item.items() if k != '_traj_length'} for item in split_data]
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    
    split_maps = {}
    for item in split_data:
        map_name = item['json'].split('/')[0]
        if map_name not in split_maps:
            split_maps[map_name] = 0
        split_maps[map_name] += 1

    print(f"\nSaved unseen_valset_{port}.json ({len(split_data)}  items)")
    report_difficulty(split_data, f"  {port}")
    print(f"  Map distribution:")
    for map_name, count in split_maps.items():
        print(f"    {map_name}: {count}  items")

print("\nDone! Generated the following files:")
for i in range(10):
    print(f"  - {output_dir}unseen_valset_3000{i}.json")
