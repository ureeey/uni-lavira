#!/usr/bin/env python3

import json
import random


random.seed(42)


input_file = 'path/to/OpenUAV_data/TravelUAV_data_json/data/uav_dataset/unseen_valset.json'
output_file = 'path/to/OpenUAV_data/TravelUAV_data_json/data/uav_dataset/unseen_valset_100.json'

print("Loadingunseen_valset.json...")
with open(input_file, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)}  items")


trajectory_groups = {}
for item in data:
    traj_path = item['json']  
    if traj_path not in trajectory_groups:
        trajectory_groups[traj_path] = []
    trajectory_groups[traj_path].append(item)

print(f"Total different trajectories: {len(trajectory_groups)} different trajectories")


valid_trajectories = {}
for traj_path, frames in trajectory_groups.items():
    if any(item.get('frame') == 1 for item in frames):
        valid_trajectories[traj_path] = frames

print(f"Trajectories containing frame=1: {len(valid_trajectories)}  items")


maps = {}
for traj_path in valid_trajectories.keys():
    map_name = traj_path.split('/')[0]
    if map_name not in maps:
        maps[map_name] = []
    maps[map_name].append(traj_path)

print(f"\nMap statistics:")
for map_name, trajs in maps.items():
    print(f"  {map_name}: {len(trajs)}  trajectories")


trajectory_paths = list(valid_trajectories.keys())
if len(trajectory_paths) < 100:
    print(f"\nWarning: valid trajectories only have {len(trajectory_paths)}  items, fewer than 100 items")
    selected_trajectories = trajectory_paths
else:
    selected_trajectories = random.sample(trajectory_paths, 100)
    print(f"\nRandomly selected 100 trajectories")


selected_data = []
for traj_path in selected_trajectories:
    frames = valid_trajectories[traj_path]
    
    frames_sorted = sorted(frames, key=lambda x: x.get('frame', 0))
    selected_data.extend(frames_sorted)

print(f"The selected 100 trajectories contain {len(selected_data)} frames")


selected_maps = {}
for traj_path in selected_trajectories:
    map_name = traj_path.split('/')[0]
    if map_name not in selected_maps:
        selected_maps[map_name] = 0
    selected_maps[map_name] += 1

print(f"\nMap distribution in sampled results:")
for map_name, count in selected_maps.items():
    print(f"  {map_name}: {count}  trajectories")


frame_counts = []
for traj_path in selected_trajectories:
    frame_count = len(valid_trajectories[traj_path])
    frame_counts.append(frame_count)

print(f"\nTrajectory length statistics:")
print(f"  average frame count: {sum(frame_counts) / len(frame_counts):.1f}")
print(f"  minimum frame count: {min(frame_counts)}")
print(f"  maximum frame count: {max(frame_counts)}")


with open(output_file, 'w') as f:
    json.dump(selected_data, f, indent=2)

print(f"\nSaved to: {output_file}")
print("Done!")
