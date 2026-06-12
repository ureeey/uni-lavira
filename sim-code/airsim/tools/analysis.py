#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np


EASY_HARD_THRESHOLD = 250  


def get_scene_and_length(ori_info_path):
    with open(ori_info_path) as f:
        info = json.load(f)
    ori_dir = info['ori_traj_dir']
    
    scene = ori_dir.rstrip('/').split('/')[-2]

    merged_path = os.path.join(ori_dir, 'merged_data.json')
    length = -1
    if os.path.exists(merged_path):
        with open(merged_path) as f:
            merged = json.load(f)
        traj = merged['trajectory_raw_detailed']
        length = sum(
            np.linalg.norm(
                np.array(traj[i + 1]['position']) - np.array(traj[i]['position'])
            )
            for i in range(len(traj) - 1)
        )
    return scene, length


def classify_folder(name):
    if name.startswith('success_'):
        return 'success', name[len('success_'):]
    elif name.startswith('oracle_'):
        return 'oracle', name[len('oracle_'):]
    else:
        return 'failed', name


def main():
    parser = argparse.ArgumentParser(description='Analyze eval outcome directory')
    parser.add_argument('outcome_dir', help='outcome directory path')
    args = parser.parse_args()

    entries = []
    for folder in sorted(os.listdir(args.outcome_dir)):
        folder_path = os.path.join(args.outcome_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        ori_info_path = os.path.join(folder_path, 'ori_info.json')
        if not os.path.exists(ori_info_path):
            continue
        result, uuid = classify_folder(folder)
        scene, length = get_scene_and_length(ori_info_path)
        difficulty = 'Easy' if 0 < length <= EASY_HARD_THRESHOLD else ('Hard' if length > EASY_HARD_THRESHOLD else '?')
        entries.append({'uuid': uuid, 'scene': scene, 'length': length, 'difficulty': difficulty, 'result': result})

    
    total = len(entries)
    n_success = sum(1 for e in entries if e['result'] == 'success')
    n_oracle = sum(1 for e in entries if e['result'] == 'oracle')
    n_failed = sum(1 for e in entries if e['result'] == 'failed')
    n_easy = sum(1 for e in entries if e['difficulty'] == 'Easy')
    n_hard = sum(1 for e in entries if e['difficulty'] == 'Hard')

    print(f'=== {args.outcome_dir} ===')
    print(f'Total: {total}  items')
    print(f'  Success: {n_success} ({n_success/max(total,1)*100:.1f}%)')
    print(f'  Oracle:  {n_oracle} ({n_oracle/max(total,1)*100:.1f}%)')
    print(f'  Failed:  {n_failed} ({n_failed/max(total,1)*100:.1f}%)')
    print(f'  Easy:    {n_easy}, Hard: {n_hard}')

    
    scenes = sorted(set(e['scene'] for e in entries))
    print(f'\n{"Scene":<25s} {"Total":>4s} {"Success":>4s} {"Oracle":>6s} {"Failure":>4s} {"SR%":>6s}')
    print('-' * 55)
    for s in scenes:
        se = [e for e in entries if e['scene'] == s]
        ss = sum(1 for e in se if e['result'] == 'success')
        so = sum(1 for e in se if e['result'] == 'oracle')
        sf = sum(1 for e in se if e['result'] == 'failed')
        sr = ss / len(se) * 100
        print(f'{s:<25s} {len(se):>4d} {ss:>4d} {so:>6d} {sf:>4d} {sr:>5.1f}%')

    
    print(f'\n{"UUID":<38s} {"Scene":<22s} {"Length(m)":>8s} {"Difficulty":>4s} {"Result":>7s}')
    print('-' * 85)
    for e in sorted(entries, key=lambda x: (x['scene'], x['length'])):
        print(f'{e["uuid"]:<38s} {e["scene"]:<22s} {e["length"]:>8.1f} {e["difficulty"]:>4s} {e["result"]:>7s}')


if __name__ == '__main__':
    main()
