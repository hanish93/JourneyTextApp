# src/app.py

import os
import cv2
import torch
import numpy as np
import logging
from glob import glob

from utils import (
    frames, move, load_yolo,
    detect_signal_color, load_ocr, ocr_signs
)

# ——— USER CONFIGURATION ———————————————
# exact frame indices (1‑based) you’ve labeled with shop/cinema text:
FRAME_WHITELIST = [7, 10, 77, 96, 116]

# exact shop/cinema names you want to capture:
WHITELIST_SIGNS = ["Tesco Express", "CREMA", "Vue", "Townhall", "Wool Pack Hub"]

def run_clip(path: str, model, ocr_reader, dev: str):
    # build frame iterator
    if os.path.isdir(path):
        imgs = sorted(glob(os.path.join(path, "*.jpg")))
        if not imgs:
            vids = sorted(glob(os.path.join(path, "*.mp4")))
            return "\n\n".join(run_clip(v, model, ocr_reader, dev) for v in vids)
        it = (cv2.imread(fp) for fp in imgs)
    else:
        it = frames(path, fps=1)

    prev_gray = None
    prev_light = None
    raw_tl = []
    seen_signs = set()

    for idx, img in enumerate(it, start=1):
        if img is None:
            continue

        # 1) motion verb
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        verb = move(prev_gray, gray)
        prev_gray = gray

        # 2) traffic light
        res = model(img, conf=0.25, verbose=False)[0]
        for b in res.boxes:
            cls = model.model.names[int(b.cls[0])]
            if cls == "traffic light":
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                roi = img[y1:y2, x1:x2]
                color = detect_signal_color(roi)
                if color and color != prev_light:
                    raw_tl.append(f"signal_{color}")
                    prev_light = color

        # 3) signage only on your labeled frames
        if idx in FRAME_WHITELIST:
            texts = ocr_signs(img, ocr_reader)
            for t in texts:
                for key in WHITELIST_SIGNS:
                    if key.lower() in t.lower() and key not in seen_signs:
                        seen_signs.add(key)
                        raw_tl.append(f"sign_{key}")

        # 4) record verb only on stops or turns, or if it’s a labeled frame
        if verb in {"stop", "turn_left", "turn_right"} or idx in FRAME_WHITELIST:
            raw_tl.append(verb)

        # progress
        print(f"[{idx:03d}] verb={verb:10s} light={prev_light or '-':6s}"
              f" signs={list(seen_signs)}")

    # collapse consecutive duplicates
    timeline = []
    for e in raw_tl:
        if not timeline or timeline[-1] != e:
            timeline.append(e)

    # map to English
    mapping = {
        "drive":        "drove straight",
        "stop":         "stopped",
        "turn_right":   "took a slight right",
        "turn_left":    "took a slight left",
        "signal_red":   "stopped at the red light",
        "signal_green": "the signal turned green",
    }
    parts = []
    for ev in timeline:
        if ev in mapping:
            parts.append(mapping[ev])
        elif ev.startswith("sign_"):
            parts.append(f"passed {ev.split('_',1)[1]}")

    summary = "I " + " and ".join(parts) + "."
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary

def run(input_path: str, yolo_weights: str = None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_yolo(dev, yolo_weights)
    ocr_reader = load_ocr()
    return run_clip(input_path, model, ocr_reader, dev)

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Journey summariser")
    p.add_argument("--input","-i", required=True,
                   help="Folder of JPG frames or single MP4")
    p.add_argument("--yolo-model","-m", default=None,
                   help="Path to custom YOLOv8 .pt (omit for yolov8n)")
    args = p.parse_args()
    run(args.input, args.yolo_model)
