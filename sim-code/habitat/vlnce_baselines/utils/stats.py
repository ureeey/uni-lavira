"""Model-usage statistics aggregation.

Split out of ZS_Evaluator_mp.py: merges the per-worker model_usage_stats_*.json
files (LA/VA call counts and token totals) written during a multi-process eval
run into a single summary. Called once from run_mp after all workers finish.
"""
import glob
import json
import os

from habitat import logger


def merge_model_usage_stats(stats_dir, split="val_unseen"):
    """Merge per-worker model-usage stats into a single summary file.

    Args:
        stats_dir: directory holding `model_usage_stats_{split}_r*_w*.json` files.
        split: dataset split name (used in the glob pattern).
    """
    pattern = os.path.join(stats_dir, f"model_usage_stats_{split}_r*_w*.json")
    stat_files = glob.glob(pattern)

    if not stat_files:
        logger.info(f"No model usage stat files found in {stats_dir} with pattern {pattern}")
        return

    merged_stats = {
        'la': {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        },
        'va': {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        },
        'total_calls': 0,
        'total_tokens': 0,
        'num_processes': 0,
        'process_stats': []
    }

    for stat_file in stat_files:
        try:
            with open(stat_file, 'r') as f:
                stats = json.load(f)

            merged_stats['la']['calls'] += stats['la']['calls']
            merged_stats['la']['input_tokens'] += stats['la']['input_tokens']
            merged_stats['la']['output_tokens'] += stats['la']['output_tokens']
            merged_stats['la']['total_tokens'] += stats['la']['total_tokens']

            merged_stats['va']['calls'] += stats['va']['calls']
            merged_stats['va']['input_tokens'] += stats['va']['input_tokens']
            merged_stats['va']['output_tokens'] += stats['va']['output_tokens']
            merged_stats['va']['total_tokens'] += stats['va']['total_tokens']

            merged_stats['total_calls'] += stats['total_calls']
            merged_stats['total_tokens'] += stats['total_tokens']
            merged_stats['num_processes'] += 1

            process_info = {
                'file': os.path.basename(stat_file),
                'stats': stats
            }
            merged_stats['process_stats'].append(process_info)

        except Exception as e:
            logger.info(f"Error loading {stat_file}: {e}")

    merged_file = os.path.join(stats_dir, f"merged_model_usage_stats_{split}.json")
    with open(merged_file, 'w') as f:
        json.dump(merged_stats, f, indent=2)

    logger.info("=== MERGED MODEL USAGE STATISTICS ===")
    logger.info(f"Number of processes: {merged_stats['num_processes']}")
    logger.info(f"Language Action Model:")
    logger.info(f"  - Total calls: {merged_stats['la']['calls']:,}")
    logger.info(f"  - Total input tokens: {merged_stats['la']['input_tokens']:,}")
    logger.info(f"  - Total output tokens: {merged_stats['la']['output_tokens']:,}")
    logger.info(f"  - Total tokens: {merged_stats['la']['total_tokens']:,}")
    logger.info(f"Vision Action Model:")
    logger.info(f"  - Total calls: {merged_stats['va']['calls']:,}")
    logger.info(f"  - Total input tokens: {merged_stats['va']['input_tokens']:,}")
    logger.info(f"  - Total output tokens: {merged_stats['va']['output_tokens']:,}")
    logger.info(f"  - Total tokens: {merged_stats['va']['total_tokens']:,}")
    logger.info(f"OVERALL TOTAL:")
    logger.info(f"  - Total calls: {merged_stats['total_calls']:,}")
    logger.info(f"  - Total tokens: {merged_stats['total_tokens']:,}")
    logger.info(f"Merged statistics saved to: {merged_file}")
    logger.info("=====================================")

    return merged_stats
