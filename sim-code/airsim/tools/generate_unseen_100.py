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


frame1_data = [item for item in data if item.get('frame') == 1]
print(f"Frame=1 items: {len(frame1_data)} ")


maps = {}
for item in frame1_data:
    map_name = item['json'].split('/')[0]
    if map_name not in maps:
        maps[map_name] = []
    maps[map_name].append(item)

print(f"\nMap statistics:")
for map_name, items in maps.items():
    print(f"  {map_name}: {len(items)}  trajectories")


if len(frame1_data) < 100:
    print(f"\nWarning: frame=1 data only has {len(frame1_data)} , fewer than 100")
    selected_data = frame1_data
else:
    selected_data = random.sample(frame1_data, 100)
    print(f"\nRandomly selected 100 items")


selected_maps = {}
for item in selected_data:
    map_name = item['json'].split('/')[0]
    if map_name not in selected_maps:
        selected_maps[map_name] = 0
    selected_maps[map_name] += 1

print(f"\nMap distribution in sampled results:")
for map_name, count in selected_maps.items():
    print(f"  {map_name}: {count}  trajectories")


with open(output_file, 'w') as f:
    json.dump(selected_data, f, indent=2)

print(f"\nSaved to: {output_file}")
print("Done!")
