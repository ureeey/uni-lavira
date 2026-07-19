import sys
import types
import warnings

# Suppress gym's "unmaintained since 2022" stderr notice before any import triggers it.
# gym/__init__.py does `import gym_notices.notices as notices; print(notice, file=sys.stderr)`
# which bypasses Python's warnings system. Mocking gym_notices before gym is imported
# makes the `import gym_notices.notices as notices` line grab our no-op stub.
_dummy_notices = types.ModuleType('gym_notices')
_dummy_notices.notices = types.ModuleType('gym_notices.notices')
_dummy_notices.notices.notices = {}
sys.modules['gym_notices'] = _dummy_notices
sys.modules['gym_notices.notices'] = _dummy_notices.notices

# Also suppress noisy third-party FutureWarnings
warnings.filterwarnings('ignore', message='.*Importing from timm.models.layers is deprecated.*')

import argparse
import random
import os
import gzip
import json
import re
from copy import deepcopy
import glob
from pprint import pprint
import time

import torch
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.config.default import get_config
from vlnce_baselines.utils.misc import seed_everything

def get_episode_ids_from_config(config):
    data_path = config.TASK_CONFIG.DATASET.DATA_PATH
    split = config.TASK_CONFIG.DATASET.SPLIT
    if _is_rxr_dataset(config): role = config.TASK_CONFIG.DATASET.ROLES
    logger.info(split)
    if _is_rxr_dataset(config): data_path = data_path.format(split=split,role=role[0])
    else: data_path = data_path.format(split=split)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Episode data file not found: {data_path}")

    if data_path.endswith('.gz'):
        open_fn = gzip.open
        mode = 'rt'
    else:
        open_fn = open
        mode = 'r'

    with open_fn(data_path, mode) as f:
        data = json.load(f)
        episodes = data.get("episodes", [])
        # Return detailed info: id and scene_id
        episode_info = [{"id": str(e["episode_id"]), "scene_id": e["scene_id"]} for e in episodes]
    return episode_info


def extract_episode_ids_from_log(log_path: str) -> set:
    """Extract numeric episode ids from log lines like: ep187:{...}

    Returns a set of episode id strings, e.g. {"187", "1051"}.
    """
    if not log_path:
        return set()
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    pattern = re.compile(r"\bep(\d+)\s*:")
    episode_ids = set()
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                episode_ids.add(m.group(1))
    return episode_ids


def _is_rxr_dataset(cfg) -> bool:
    try:
        dp = getattr(cfg.TASK_CONFIG.DATASET, 'DATA_PATH', None)
        if dp and 'rxr' in str(dp).lower():
            return True
        name = (
            getattr(cfg.TASK_CONFIG.DATASET, 'DATASET', None)
            or getattr(cfg.TASK_CONFIG.DATASET, 'NAME', None)
            or getattr(cfg.TASK_CONFIG.DATASET, 'DATASET_NAME', None)
        )
        if name and 'rxr' in str(name).lower():
            return True
    except Exception:
        pass
    return False

def run_exp(exp_name: str, exp_config: str,
            run_type: str, nprocesses: int, opts=None, use_navdp: bool = False, use_fmm: bool = True, debug_episodes: str = None, episode_file: str = None, resume_from_log: str = None, api_format: str = None, dashscope_maas: bool = False, rollout_v2: bool = False, rollout_v3: bool = False, rollout_v4: bool = False) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    # Set API format early so agent.py picks it up via env var
    if api_format:
        os.environ['LAVIRA_API_FORMAT'] = api_format
        logger.info(f"API format set to: {api_format}")
    if dashscope_maas:
        os.environ['DASHSCOPE_USE_MAAS'] = '1'
        logger.info("DashScope MaaS mode: using dedicated workspace endpoint")

    config = get_config(exp_config, opts)
    config.defrost()
    config.TENSORBOARD_DIR += exp_name
    config.CHECKPOINT_FOLDER += exp_name
    config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.VIDEO_DIR += exp_name
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE
    if use_navdp:
        config.USE_NAVDP = True
    elif 'USE_NAVDP' not in config:
        config.USE_NAVDP = False

    if use_fmm:
        config.USE_FMM = True
    elif 'USE_FMM' not in config:
        config.USE_FMM = False

    # --no-fmm sets use_fmm=False; default is True. Always honor the CLI value.
    config.USE_FMM = use_fmm

    config.ROLLOUT_V2 = rollout_v2
    config.ROLLOUT_V3 = rollout_v3
    config.ROLLOUT_V4 = rollout_v4

    config.freeze()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.EVAL_CKPT_PATH_DIR, exist_ok=True)
    os.system("mkdir -p data/logs/running_log")
    logger.add_filehandler('data/logs/running_log/' + config.LOG_FILE)
    from vlnce_baselines.utils.logging import LOG_PROGRESS_BAR
    if not LOG_PROGRESS_BAR:
        logger.info(f"hyper parameters:\n{config.EVAL}")

    # dataset split, start multi-processes
    gpu_ids = config.TORCH_GPU_IDS
    num_devices = len(gpu_ids)
    logger.info(f'num devices: {num_devices}, gpu_ids: {gpu_ids}, num processes: {nprocesses}')

    # Warn if oversubscribing GPUs (recommend at most 2 processes per GPU).
    if nprocesses > num_devices * 2:
        logger.info(f"Warning: {nprocesses} processes on {num_devices} GPUs may cause resource contention")

    episode_info_list = get_episode_ids_from_config(config)
    total_available = len(episode_info_list)
    logger.info(f"total available episodes: {total_available}")
    import sys
    sys.stdout.flush()

    if episode_file:
        with open(episode_file, 'r') as f:
            fixed_ids = set(json.load(f))
        episode_info_list = [info for info in episode_info_list if info['id'] in fixed_ids]
        logger.info(f"EPISODE FILE: Loaded {len(fixed_ids)} IDs from {episode_file}, matched {len(episode_info_list)}/{total_available}")
    elif debug_episodes:
        debug_ids = [eid.strip() for eid in debug_episodes.split(',')]
        debug_ids_set = set(debug_ids)
        episode_info_list = [info for info in episode_info_list if info['id'] in debug_ids_set]
        logger.info(f"DEBUG MODE: Filtering for specific episodes: {debug_ids}")
        logger.info(f"Found {len(episode_info_list)} matching episodes out of {total_available}")
        if len(episode_info_list) == 0:
             logger.info("Warning: No matching episodes found!")

        config.defrost()
        config.DEBUG_LOGGING = True
        config.DEBUG_EPISODES = debug_ids
        config.freeze()
    else:
        logger.info(f"Using all {total_available} episodes (no sampling)")

    # Resume mode: skip episodes that already appeared in a previous run log.
    # Treat "appeared in log" as "already evaluated" (success=0 is still done).
    if resume_from_log:
        done_ids = extract_episode_ids_from_log(resume_from_log)
        before = len(episode_info_list)
        if done_ids:
            episode_info_list = [info for info in episode_info_list if info["id"] not in done_ids]
        skipped = before - len(episode_info_list)
        logger.info(
            f"RESUME: log={resume_from_log}, found {len(done_ids)} episode tags; "
            f"skipping {skipped}/{before}; remaining {len(episode_info_list)}"
        )

        # When resuming, ignore EPISODE_COUNT truncation unless user explicitly
        # provides debug_episodes.
        config.defrost()
        config.EVAL.EPISODE_COUNT = -1
        config.freeze()

        if len(episode_info_list) == 0:
            logger.info("RESUME: No remaining episodes to process. Exiting.")
            return

        try:
            remaining_ids_path = os.path.join(
                "data",
                "logs",
                "running_log",
                f"remaining_episode_ids_from_{os.path.basename(resume_from_log)}.json",
            )
            os.makedirs(os.path.dirname(remaining_ids_path), exist_ok=True)
            with open(remaining_ids_path, "w") as f:
                json.dump([info["id"] for info in episode_info_list], f)
            logger.info(f"RESUME: wrote remaining episode ids to {remaining_ids_path}")
        except Exception as e:
            logger.info(f"RESUME: failed to write remaining episode ids file: {e}")

    if config.EVAL.EPISODE_COUNT > -1 and len(episode_info_list) > config.EVAL.EPISODE_COUNT:
        logger.info(f"Truncating episode list from {len(episode_info_list)} to {config.EVAL.EPISODE_COUNT} based on EVAL.EPISODE_COUNT")
        random.seed(time.time())
        random.shuffle(episode_info_list)
        episode_info_list = episode_info_list[:config.EVAL.EPISODE_COUNT]

    episode_ids_for_print = [info['id'] for info in episode_info_list]
    logger.info(f"Total episodes to process: {len(episode_info_list)}, {episode_ids_for_print}")

    dynamic_queue = bool(getattr(config.EVAL, 'DYNAMIC_QUEUE', False))
    if dynamic_queue:
        logger.info(f"DYNAMIC_QUEUE=True: {nprocesses} workers will pop from a shared queue of {len(episode_info_list)} episodes")
        # Sort by descending bucket size — the LPT (longest-processing-time-first)
        # heuristic: hand out episodes from the largest scenes first so the
        # last-claimed episodes are short ones.
        from collections import Counter
        scene_sizes = Counter(info['scene_id'] for info in episode_info_list)
        episode_info_list.sort(key=lambda x: -scene_sizes[x['scene_id']])

    split_episode_infos = [episode_info_list[i::nprocesses] for i in range(nprocesses)]

    if not dynamic_queue:
        for i, ep_infos in enumerate(split_episode_infos):
            logger.info(f"Process {i}: {len(ep_infos)} episodes")

    configs = []
    if dynamic_queue:
        # Build N worker configs with empty EPISODES_ALLOWED; per-ep filter is
        # set inside eval_dynamic() as the worker pops from the queue.
        for i in range(nprocesses):
            shared_config = deepcopy(config)
            shared_config.defrost()
            device_num = gpu_ids[i % num_devices]
            shared_config.local_rank = i
            shared_config.world_size = nprocesses
            shared_config.TORCH_GPU_ID = device_num
            shared_config.TORCH_GPU_IDS = [device_num]
            shared_config.SIMULATOR_GPU_IDS = [device_num]
            shared_config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = []
            shared_config.EVAL.EPISODE_COUNT = -1
            shared_config.freeze()
            configs.append(shared_config)
            logger.info(f"Process {i}: GPU {device_num}, dynamic queue")
    else:
        for i, ep_infos in enumerate(split_episode_infos):
            if len(ep_infos) == 0:  # skip processes with no episodes assigned
                logger.info(f"Warning: Process {i} has no episodes assigned, skipping")
                continue

            # Save assignment file for the worker. Namespace by exp_name so
            # parallel runs don't clobber each other's assignments.
            assignment_file = f"data/logs/running_log/worker_{i}_assignments_{exp_name}.json"
            os.makedirs(os.path.dirname(assignment_file), exist_ok=True)
            with open(assignment_file, 'w') as f:
                json.dump(ep_infos, f)

            shared_config = deepcopy(config)
            shared_config.defrost()
            device_num = gpu_ids[i % num_devices]
            shared_config.local_rank = i
            shared_config.world_size = nprocesses
            shared_config.TORCH_GPU_ID = device_num
            shared_config.TORCH_GPU_IDS = [device_num]
            shared_config.SIMULATOR_GPU_IDS = [device_num]
            # Pass IDs for Habitat filtering (may include duplicates)
            shared_config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = [info['id'] for info in ep_infos]
            shared_config.EVAL.EPISODE_COUNT = -1
            shared_config.freeze()
            configs.append(shared_config)
            logger.info(f"Process {i}: GPU {device_num}, {len(ep_infos)} episodes")

    logger.info(f"Actually starting {len(configs)} processes")

    # Use individual mp.Process per worker instead of Pool so a finished
    # worker's subprocess exits immediately and releases its GPU context
    # (Pool keeps workers alive until the last task finishes, holding memory).
    procs = []
    if dynamic_queue:
        manager = mp.Manager()
        ep_queue = manager.Queue()
        for ep_info in episode_info_list:
            ep_queue.put(ep_info)
        for cfg in configs:
            p = mp.Process(target=worker_dynamic, args=(cfg, ep_queue))
            p.start()
            procs.append(p)
    else:
        for cfg in configs:
            p = mp.Process(target=worker, args=(cfg,))
            p.start()
            procs.append(p)

    logger.info(f"Starting multiprocessing with {len(procs)} workers...")
    start_time = time.time()

    try:
        while True:
            alive = [p for p in procs if p.is_alive()]
            if not alive:
                break
            elapsed = time.time() - start_time
            if not LOG_PROGRESS_BAR:
                logger.info(f"Progress check: {elapsed:.0f}s elapsed, {len(alive)}/{len(procs)} workers still running")
            if elapsed > 36000:  # 10h safety timeout
                logger.info("Timeout reached, terminating stragglers")
                for p in alive:
                    p.terminate()
                break
            time.sleep(30)

        for p in procs:
            p.join()

        total_time = time.time() - start_time
        successful = sum(1 for p in procs if p.exitcode == 0)
        failed = len(procs) - successful
        logger.info(f"Multiprocessing completed in {total_time:.0f}s")
        logger.info(f"Successful workers: {successful}/{len(procs)}")
        if failed > 0:
            logger.info(f"Failed workers: {failed} (exitcodes: {[p.exitcode for p in procs if p.exitcode != 0]})")

    except Exception as e:
        logger.info(f"Error occurred: {e}")
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()
        logger.info("All processes terminated")
        return
    # Merge and print final statistics
    logger.info("=" * 50)
    logger.info("AGGREGATED STATISTICS ACROSS ALL PROCESSES")
    logger.info("=" * 50)

    # Load and merge episode stats
    from collections import defaultdict
    aggregated_metrics = defaultdict(list)
    total_episodes = 0
    evaluated_episode_ids = []

    stats_files = glob.glob(os.path.join(config.EVAL_CKPT_PATH_DIR, "stats_ep_ckpt_*.json"))
    for stats_file in stats_files:
        try:
            with open(stats_file, 'r') as f:
                data = json.load(f)
                total_episodes += len(data)
                evaluated_episode_ids.extend(list(data.keys()))
                for ep_id, metrics in data.items():
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            aggregated_metrics[k].append(v)
        except Exception as e:
            logger.info(f"Error loading {stats_file}: {e}")

    # Calculate averages
    final_averages = {}
    for k, v_list in aggregated_metrics.items():
        if v_list:
            final_averages[k] = sum(v_list) / len(v_list)

    logger.info(f"Total Episodes Evaluated: {total_episodes}")
    logger.info(f"Evaluated Episode IDs: {sorted(evaluated_episode_ids)}")
    logger.info("Final Average Metrics:")
    pprint(final_averages)

    # Merge model usage stats
    from vlnce_baselines.utils.stats import merge_model_usage_stats
    merge_model_usage_stats(config.EVAL_CKPT_PATH_DIR, config.TASK_CONFIG.DATASET.SPLIT)

    # Merge navigation stats
    nav_stats_files = glob.glob(os.path.join(config.EVAL_CKPT_PATH_DIR, "nav_stats_*.json"))
    total_backtracks = 0
    total_waypoints = 0

    for fpath in nav_stats_files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                total_backtracks += data.get('total_backtracks', 0)
                total_waypoints += data.get('total_waypoints', 0)
        except:
            pass

    logger.info("=" * 50)
    logger.info("NAVIGATION STATISTICS")
    logger.info("=" * 50)
    logger.info(f"Total Backtracks: {total_backtracks}")
    logger.info(f"Total Waypoints: {total_waypoints}")
    logger.info("="*50)

def worker(config):
    try:
        worker_log_file = f"data/logs/running_log/worker_{config.local_rank}_{config.LOG_FILE}"
        logger.add_filehandler(worker_log_file)

        logger.info(f"Worker started: local_rank={config.local_rank}, device={config.TORCH_GPU_ID}")
        import sys
        sys.stdout.flush()
        logger.info(f"Worker {config.local_rank} started on GPU {config.TORCH_GPU_ID}")
        logger.info(f"Worker {config.local_rank} processing {len(config.TASK_CONFIG.DATASET.EPISODES_ALLOWED)} episodes")

        seed_everything(config.TASK_CONFIG.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        if torch.cuda.is_available():
            torch.set_num_threads(1)

        TRAINER = baseline_registry.get_trainer(config.TRAINER_NAME)
        assert TRAINER is not None, f"{config.TRAINER_NAME} is not supported"
        logger.info(f"Worker {config.local_rank}: Starting trainer")
        logger.info(f"Worker {config.local_rank}: Starting trainer")
        trainer = TRAINER(config, r2r=not _is_rxr_dataset(config))
        trainer.eval()
        logger.info(f"Worker {config.local_rank}: Completed successfully")
        logger.info(f"Worker {config.local_rank}: Completed successfully")
        return True
    except Exception as e:
        logger.info(f"Worker {config.local_rank} failed with error: {str(e)}")
        logger.error(f"Worker {config.local_rank} failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(traceback.format_exc())
        return False


def worker_dynamic(config, ep_queue):
    """Worker entry for EVAL.DYNAMIC_QUEUE: pop episodes from a shared queue.

    Builds the trainer once (heavy: GroundedSAM, RepViTSAM, policy, agent),
    then loops on the queue rebuilding only the habitat env per ep.
    """
    try:
        worker_log_file = f"data/logs/running_log/worker_{config.local_rank}_{config.LOG_FILE}"
        logger.add_filehandler(worker_log_file)

        logger.info(f"Worker(dyn) started: local_rank={config.local_rank}, device={config.TORCH_GPU_ID}")
        import sys
        sys.stdout.flush()
        logger.info(f"Worker {config.local_rank} (dyn) started on GPU {config.TORCH_GPU_ID}")

        seed_everything(config.TASK_CONFIG.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        if torch.cuda.is_available():
            torch.set_num_threads(1)

        TRAINER = baseline_registry.get_trainer(config.TRAINER_NAME)
        assert TRAINER is not None, f"{config.TRAINER_NAME} is not supported"
        trainer = TRAINER(config, r2r=not _is_rxr_dataset(config))
        trainer.eval_dynamic(ep_queue)
        logger.info(f"Worker {config.local_rank} (dyn): Completed successfully")
        logger.info(f"Worker {config.local_rank} (dyn): Completed successfully")
        return True
    except Exception as e:
        logger.info(f"Worker {config.local_rank} (dyn) failed: {e}")
        logger.error(f"Worker {config.local_rank} (dyn) failed: {e}")
        import traceback
        traceback.print_exc()
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default="test",
        required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--run-type",
        choices=["eval"],
        required=True,
        help="run type of the experiment(train, eval, inference), only eval for zero-shot vln",
    )
    parser.add_argument(
        "--nprocesses",
        type=int,
        default=1,
        help="number of processes",
    )
    parser.add_argument(
        "--use-navdp",
        action="store_true",
        default=False,
        help="Use NavDP for trajectory planning (default: False)",
    )
    parser.add_argument(
        "--use-fmm",
        action="store_true",
        default=True,
        help="Use FMM for trajectory planning (default: True)",
    )
    parser.add_argument(
        "--no-fmm",
        action="store_false",
        dest="use_fmm",
        help="Disable FMM",
    )
    parser.add_argument(
        "--debug-episodes",
        type=str,
        default=None,
        help="Comma-separated list of episode IDs to run for debugging (e.g., '232,330'). Overrides other sampling options.",
    )
    parser.add_argument(
        "--episode-file",
        type=str,
        default=None,
        help="Path to JSON file containing list of episode IDs to evaluate (e.g., joint_stratified_100_seed42.json).",
    )
    parser.add_argument(
        "--resume-from-log",
        type=str,
        default=None,
        help=(
            "Resume eval by skipping episodes already present in a previous log. "
            "The log should contain lines like 'ep187:{...}'. "
            "Example: --resume-from-log logs/0310-165718.log"
        ),
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--api-format",
        type=str,
        default=None,
        choices=["openai", "dashscope"],
        help="API format: 'openai' (OpenAI-compatible, default) or 'dashscope' (DashScope native SDK). "
             "Overrides the LAVIRA_API_FORMAT env var.",
    )
    parser.add_argument(
        "--dashscope-maas",
        action="store_true",
        default=False,
        help="DashScope mode: use MaaS dedicated workspace (derived from VA_BASE_URL) "
             "instead of the public dashscope.aliyuncs.com. Gives dedicated resources "
             "but inherits the MaaS gateway body-size limit (~22MB).",
    )
    parser.add_argument(
        "--rollout-v2",
        action="store_true",
        default=False,
        help="Use rollout_v2 (experimental) instead of the default rollout loop.",
    )
    parser.add_argument(
        "--rollout-v3",
        action="store_true",
        default=False,
        help="Use rollout_v3 (experimental) instead of the default rollout loop.",
    )
    parser.add_argument(
        "--rollout-v4",
        action="store_true",
        default=False,
        help="Use rollout_v4 (experimental) instead of the default rollout loop.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    # --api-format / --dashscope-maas may appear after REMAINDER opts;
    # scan opts to extract them so config merge doesn't choke.
    _bool_flags = {'--dashscope-maas', '--rollout-v2', '--rollout-v3', '--rollout-v4'}
    _val_flags  = {'--api-format', '--debug-episodes'}
    if args.opts:
        i = 0
        while i < len(args.opts):
            tok = args.opts[i]
            if tok in _bool_flags:
                setattr(args, tok[2:].replace('-', '_'), True)
                args.opts.pop(i)
            elif tok in _val_flags and i + 1 < len(args.opts):
                if getattr(args, tok[2:].replace('-', '_'), None) is None:
                    setattr(args, tok[2:].replace('-', '_'), args.opts[i + 1])
                args.opts.pop(i)       # value
                args.opts.pop(i)       # flag
            else:
                i += 1
    logger.info(args)

    mp.set_start_method('spawn', force=True)
    run_exp(**vars(args))
