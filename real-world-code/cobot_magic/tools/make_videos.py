#!/usr/bin/env python3
"""Build MP4 videos from per-run ``front_view/`` image folders.

Each run directory is expected to contain a sub-folder named ``front_view/``
that holds JPEG/PNG frames captured during navigation. The tool iterates over
the run directories, sorts the frames by filename, and writes one MP4 per run
into the output folder.

Usage
-----
Process all run directories under the current working directory::

    python tools/make_videos.py

Process only directories whose names start with a specific prefix::

    python tools/make_videos.py --prefix 20260414

Process directories under a custom root and write videos elsewhere::

    python tools/make_videos.py --root /data/cobot_runs --output /data/videos

Process a single specific run directory by name::

    python tools/make_videos.py --run-dir 20260414-165632

Arguments
---------
--root      Root directory to scan for run folders (default: current directory).
--output    Directory to write MP4 files into (default: <root>/output_videos).
--prefix    Only process run directories whose name starts with this string.
--run-dir   Process only this single named subdirectory (relative to --root).
--fps       Frames per second for the output video (default: 10).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def create_videos_from_folders(
    root_dir: str,
    output_folder: str,
    fps: int = 10,
    prefix: str = "",
    run_dir: str = "",
) -> None:
    """Iterate run directories and encode a video for each.

    Parameters
    ----------
    root_dir:
        Directory containing timestamped run subdirectories.
    output_folder:
        Destination for the generated ``.mp4`` files.
    fps:
        Frames per second for the encoded video.
    prefix:
        When non-empty, skip run directories that do not start with this string.
    run_dir:
        When non-empty, process only this single directory name inside
        ``root_dir`` (the ``prefix`` filter is ignored in this mode).
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required. Install it with: pip install opencv-python")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    if run_dir:
        # Single-directory mode: ignore prefix, process only the named dir.
        candidates = [run_dir]
    else:
        candidates = sorted(os.listdir(root_dir))

    processed = 0
    for item in candidates:
        item_path = os.path.join(root_dir, item)

        if not os.path.isdir(item_path):
            continue

        # Apply prefix filter (only in multi-directory mode).
        if not run_dir and prefix and not item.startswith(prefix):
            continue

        front_view_path = os.path.join(item_path, "front_view")
        if not os.path.exists(front_view_path) or not os.path.isdir(front_view_path):
            print(f"Skipping {item}: no 'front_view' subfolder found.")
            continue

        print(f"Processing: {front_view_path}")

        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images.extend(glob.glob(os.path.join(front_view_path, ext)))
        images.sort()

        if not images:
            print(f"Skipping {item}: no images found in front_view/.")
            continue

        first_frame = cv2.imread(images[0])
        if first_frame is None:
            print(f"Skipping {item}: could not read first image {images[0]}.")
            continue

        height, width, _ = first_frame.shape
        size = (width, height)

        video_name = f"{item}.mp4"
        video_path = os.path.join(output_folder, video_name)

        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )

        for image_file in images:
            img = cv2.imread(image_file)
            if img is None:
                print(f"Warning: could not read {image_file}, skipping frame.")
                continue
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, size)
            out.write(img)

        out.release()
        print(f"Saved video: {video_path}")
        processed += 1

    print(f"Done. {processed} video(s) written to {output_folder}.")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MP4 videos from front_view/ image folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--root",
        default=os.getcwd(),
        help="Root directory containing run subdirectories (default: cwd).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for MP4 files (default: <root>/output_videos).",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Only process run directories whose name starts with this prefix.",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        dest="run_dir",
        help="Process only this single named subdirectory inside --root.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the output video (default: 10).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    output = args.output or os.path.join(args.root, "output_videos")
    create_videos_from_folders(
        root_dir=args.root,
        output_folder=output,
        fps=args.fps,
        prefix=args.prefix,
        run_dir=args.run_dir,
    )
