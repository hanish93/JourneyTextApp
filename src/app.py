# src/app.py

import os
import cv2
import torch
import numpy as np
import logging
from glob import glob

from utils import (
    frames,
    load_det,
    detect_signal_color,
    landmarks,
)

def move(prev_gray, curr_gray, dx=1.5, stop_thr=0.2):
    """Optical‑flow‐based verb (drive/stop/turn_left/turn_right)."""
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, .5, 3, 15, 3, 5, 1.2, 0
    )
    dxm = flow[...,0].mean()
    mag = np.linalg.norm(flow, axis=2).mean()
    if mag < stop_thr:       return "stop"
    if dxm > dx:             return "turn_right"
    if dxm < -dx:            return "turn_left"
    return "drive"

def run_clip(path, models, dev):
    # ───── build frame iterator ─────────────────────────
    if os.path.isdir(path):
        # try images first
        imgs = sorted(glob(os.path.join(path, "*.jpg")))
        if imgs:
            it = (cv2.imread(f) for f in imgs)
        else:
            # fallback: videos in folder
            vids = sorted(glob(os.path.join(path, "*.mp4")))
            if vids:
                return "\n\n".join(run_clip(v, models, dev) for v in vids)
            raise FileNotFoundError(f"No .jpg or .mp4 in '{path}'")
    else:
        it = frames(path, fps=1)

    yolo, ocr = models["det"]
    prev_gray = None

    seen_signs = set()
    prev_signal = None
    timeline = []

    for idx, img in enumerate(it, start=1):
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        verb = move(prev_gray, gray)
        prev_gray = gray

        # detect traffic light color
        color = detect_signal_color(img, yolo)
        if color and color != prev_signal:
            timeline.append(f"signal_{color}")
        prev_signal = color

        # detect signage
        names = landmarks(img, yolo, ocr)
        for nm in names:
            if nm not in seen_signs:
                seen_signs.add(nm)
                timeline.append(f"sign_{nm}")

        # road motion
        timeline.append(verb)

        # progress
        print(f"[frame {idx:03d}] verb={verb:10s} light={color or '-':6s}"
              f" signs={names or ['-']}")

    # ───── map events → English phrases ───────────────────
    phrases = []
    for ev in timeline:
        if ev == "signal_green":
            phrases.append("the signal turned green")
        elif ev == "signal_red":
            phrases.append("stopped at the red signal")
        elif ev == "drive":
            phrases.append("drove straight")
        elif ev == "turn_right":
            phrases.append("took a slight right")
        elif ev == "turn_left":
            phrases.append("took a slight left")
        elif ev == "stop":
            phrases.append("came to a stop")
        elif ev.startswith("sign_"):
            name = ev.split("_",1)[1]
            phrases.append(f"passed {name}")

    # ───── final single‐sentence summary ──────────────────
    summary = "I " + " and ".join(phrases) + "." if phrases else "No events detected."
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary

def run(target, custom_yolo=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    yolo, ocr = load_det(dev, custom_yolo)
    return run_clip(target, {"det": (yolo, ocr)}, dev)

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Journey summariser")
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a folder of .jpg frames or a single .mp4"
    )
    parser.add_argument(
        "--yolo-model", "-m", default=None,
        help="Path to custom YOLOv8 .pt weights"
    )
    args = parser.parse_args()
    run(args.input, args.yolo_model)
