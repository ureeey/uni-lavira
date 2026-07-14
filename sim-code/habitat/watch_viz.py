#!/usr/bin/env python3
"""
Real-time visualization watcher for LaViRA evaluation.

Monitors saved_rgb_images/ for new combined_step*.png files and displays them
in an OpenCV window. Designed to run persistently: start once, keep it open
across multiple run_mp.py invocations.

Usage:
    python watch_viz.py --auto              # auto-track the active episode
    python watch_viz.py <save_dir>          # watch a specific episode directory

Controls:
    q / ESC  - quit
    Space    - pause/resume auto-advance
    Left/Right arrows - step through frames
"""

import argparse
import os
import sys

# Suppress Qt thread warnings from OpenCV's GUI backend.
os.environ['QT_LOGGING_RULES'] = '*=false'

# Fallback: filter stderr for Qt moveToThread noise that bypasses the logging framework.
_stderr_write = sys.stderr.write


def _filtered_write(s):
    if isinstance(s, str) and 'moveToThread' in s:
        return len(s)  # pretend we wrote it
    return _stderr_write(s)


sys.stderr.write = _filtered_write

import cv2
import numpy as np
import re
import subprocess
import time
import glob


# ── screen detection ──────────────────────────────────────────────────

def _parse_xrandr():
    try:
        out = subprocess.run(['xrandr', '--current'], capture_output=True, text=True,
                             timeout=3)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    monitors = []
    for line in out.stdout.splitlines():
        if ' connected ' not in line and ' connected\t' not in line:
            continue
        is_primary = ' primary ' in line or ' primary\t' in line
        m = re.search(r'(\d{3,5})x(\d{3,5})', line)
        if m:
            monitors.append((int(m.group(1)), int(m.group(2)), is_primary))
    return monitors


def get_screen_size():
    monitors = _parse_xrandr()
    if monitors:
        primary = [m for m in monitors if m[2]]
        if primary:
            return primary[0][0], primary[0][1]
        smallest = min(monitors, key=lambda m: m[0] * m[1])
        return smallest[0], smallest[1]
    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if 800 <= w <= 7680 and 600 <= h <= 4320:
            if w >= 3000 and w / max(h, 1) > 2.5:
                w //= 2
            return w, h
    except Exception:
        pass
    return 1920, 1080


# ── helpers ───────────────────────────────────────────────────────────

def make_placeholder(w, h, text, sub=""):
    img = 30 * np.ones((h, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    ts = cv2.getTextSize(text, font, 1.2, 2)[0]
    cv2.putText(img, text, ((w - ts[0]) // 2, h // 2 - 15),
                font, 1.2, (180, 180, 180), 2, cv2.LINE_AA)
    if sub:
        ts = cv2.getTextSize(sub, font, 0.7, 1)[0]
        cv2.putText(img, sub, ((w - ts[0]) // 2, h // 2 + 30),
                    font, 0.7, (100, 100, 100), 1, cv2.LINE_AA)
    return img


def find_latest_episode(base):
    """Return (ep_dir, mtime) of the most recently modified episode with images,
    or (None, 0) if none exist."""
    best = (None, 0)
    if not os.path.isdir(base):
        return best
    for exp in os.listdir(base):
        exp_d = os.path.join(base, exp)
        if not os.path.isdir(exp_d):
            continue
        for ep in os.listdir(exp_d):
            ep_d = os.path.join(exp_d, ep)
            if not os.path.isdir(ep_d):
                continue
            imgs = glob.glob(os.path.join(ep_d, "combined_step*.png"))
            if imgs:
                mt = max(os.path.getmtime(p) for p in imgs)
                if mt > best[1]:
                    best = (ep_d, mt)
    return best


def load_dir(ep_dir):
    """Return (step_to_file, max_step, seen_set) for an episode directory."""
    s2f, seen = {}, set()
    mx = -1
    for fp in sorted(glob.glob(os.path.join(ep_dir, "combined_step*.png"))):
        seen.add(fp)
        try:
            s = int(os.path.basename(fp).replace("combined_step", "").replace(".png", ""))
            s2f[s] = fp
            if s > mx:
                mx = s
        except ValueError:
            continue
    return s2f, mx, seen


def draw_bar(img, text, color=(0, 255, 0)):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(overlay, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 1, cv2.LINE_AA)
    img[0:35, :, :] = cv2.addWeighted(overlay[0:35, :, :], 0.7,
                                       img[0:35, :, :], 0.3, 0)


# ── main ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Watch LaViRA combined_step PNGs")
    p.add_argument("watch_dir", nargs="?", help="Episode directory")
    p.add_argument("--auto", action="store_true", help="Auto-track active episode")
    p.add_argument("--base", default="saved_rgb_images", help="Base directory")
    args = p.parse_args()

    WIN = "LaViRA Viz"
    sw, sh = get_screen_size()
    MW = min(sw - 80, 1400)
    MH = min(sh - 80, 900)

    def fit(img):
        h, w = img.shape[:2]
        if w <= MW and h <= MH:
            return img
        s = min(MW / w, MH / h)
        return cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)),
                          interpolation=cv2.INTER_AREA)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.imshow(WIN, fit(np.zeros((min(MH, 600), min(MW, 800), 3), dtype=np.uint8)))
    cv2.waitKey(1)

    manual = not args.auto
    base = os.path.abspath(args.base)
    wdir = None

    if manual:
        if not args.watch_dir:
            p.print_help(); sys.exit(1)
        wdir = os.path.abspath(args.watch_dir)
        if not os.path.isdir(wdir):
            wdir = os.path.abspath(os.path.join("saved_rgb_images", args.watch_dir))

    # State
    s2f, seen = {}, set()
    mx = cur = -1
    paused = False
    last_good = None  # holds last successfully-read frame to avoid flashes
    dot = 0

    print("LaViRA Viz  |  [q/ESC] quit  [Space] pause  [← →] step")

    while True:
        now = time.time()

        # ── auto: find the most recent episode ──
        if args.auto:
            cand, _cmt = find_latest_episode(base)
            if cand is None:
                # No episode with images anywhere → show waiting
                if wdir is not None:
                    wdir = None
                    s2f, seen, mx = {}, set(), -1
                    cur = -1
            elif cand != wdir:
                # New or different episode → switch
                wdir = cand
                s2f, mx, seen = load_dir(wdir)
                cur = mx
                paused = False
                exp = os.path.basename(os.path.dirname(wdir))
                ep = os.path.basename(wdir)
                print(f"[{time.strftime('%H:%M:%S')}] {exp}/{ep}  step {mx}")
            else:
                # Same episode — check for new step files
                new = False
                for fp in sorted(glob.glob(os.path.join(wdir, "combined_step*.png"))):
                    if fp not in seen:
                        seen.add(fp)
                        try:
                            s = int(os.path.basename(fp).replace("combined_step", "").replace(".png", ""))
                            s2f[s] = fp
                            if s > mx:
                                mx = s
                            new = True
                        except ValueError:
                            continue
                if new and not paused:
                    cur = mx

        # ── manual: poll fixed directory ──
        else:
            new = False
            for fp in sorted(glob.glob(os.path.join(wdir, "combined_step*.png"))):
                if fp not in seen:
                    seen.add(fp)
                    try:
                        s = int(os.path.basename(fp).replace("combined_step", "").replace(".png", ""))
                        s2f[s] = fp
                        if s > mx:
                            mx = s
                        new = True
                    except ValueError:
                        continue
            if new and not paused:
                cur = mx

        # ── render ────────────────────────────────────────────────
        if wdir is None:
            # No episode exists yet
            dot += 1
            display = make_placeholder(MW, MH,
                                       f"Waiting for evaluation to start{'.' * ((dot // 3) % 4)}",
                                       "Start run_mp.py to begin")
        elif cur >= 0 and cur in s2f and os.path.exists(s2f[cur]):
            img = cv2.imread(s2f[cur])
            if img is not None:
                last_good = img
                display = img.copy()
                exp = os.path.basename(os.path.dirname(wdir))
                ep = os.path.basename(wdir)
                status = f"Step {cur}  |  {exp}/{ep}"
                if paused:
                    status += "  [PAUSED]"
                draw_bar(display, status)
            elif last_good is not None:
                display = last_good.copy()
                draw_bar(display, "(read error — showing last good frame)", (0, 165, 255))
            else:
                display = make_placeholder(MW, MH, "Loading...")
        elif last_good is not None:
            # No valid current step, but we have a last good frame
            display = last_good.copy()
            draw_bar(display, "Waiting for frame...", (100, 100, 255))
        else:
            dot += 1
            display = make_placeholder(MW, MH,
                                       f"Waiting for first frame{'.' * ((dot // 3) % 4)}")

        cv2.imshow(WIN, fit(display))

        # ── keys ─────────────────────────────────────────────────
        key = cv2.waitKey(500) & 0xFF
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
            if not paused:
                cur = mx
        elif key in (81, 65361):  # Left
            if s2f:
                steps = sorted(s2f)
                idx = steps.index(cur) if cur in steps else len(steps) - 1
                if idx > 0:
                    cur = steps[idx - 1]; paused = True
        elif key in (83, 65363):  # Right
            if s2f:
                steps = sorted(s2f)
                idx = steps.index(cur) if cur in steps else 0
                if idx < len(steps) - 1:
                    cur = steps[idx + 1]; paused = True
        elif key in (80, 65360):  # Home
            if s2f:
                cur = mx; paused = False
        elif key in (87, 65367):  # End
            if s2f:
                cur = sorted(s2f)[0]; paused = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
