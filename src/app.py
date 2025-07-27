import os
import cv2
import torch
import numpy as np
import logging
from glob import glob

from utils import frames, move, load_yolo, detect_signal_color

# ─── User config: exact frames & labels ───────────────────────
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS    = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub",
]
# ────────────────────────────────────────────────────────────────

def run_clip(path: str, yolo_model, dev: str):
    # build frame iterator
    if os.path.isdir(path):
        imgs = sorted(glob(os.path.join(path, "*.jpg")))
        if imgs:
            it = (cv2.imread(fp) for fp in imgs)
        else:
            vids = sorted(glob(os.path.join(path, "*.mp4")))
            return "\n\n".join(run_clip(v, yolo_model, dev) for v in vids)
    else:
        it = frames(path, fps=1)

    prev_gray  = None
    prev_verb  = None
    prev_light = None
    timeline   = []

    for idx, img in enumerate(it, start=1):
        if img is None:
            continue

        # 1) verb change only
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        verb = move(prev_gray, gray)
        prev_gray = gray
        if verb != prev_verb:
            timeline.append(verb)
            prev_verb = verb

        # 2) traffic‐light flips
        res = yolo_model(img, conf=0.25, verbose=False)[0]
        for b in res.boxes:
            cls = yolo_model.model.names[int(b.cls[0])]
            if cls == "traffic light":
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                roi = img[y1:y2, x1:x2]
                col = detect_signal_color(roi)
                if col and col != prev_light:
                    timeline.append(f"signal_{col}")
                    prev_light = col

        # 3) forced signs on whitelisted frames
        if idx in FRAME_WHITELIST:
            label = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            timeline.append(f"sign_{label}")

        # debug
        print(f"[{idx:03d}] verb={verb:10s} light={prev_light or '-':6s}"
              f" label={'YES' if idx in FRAME_WHITELIST else ''}")

    # collapse consecutive dupes
    events = []
    for e in timeline:
        if not events or events[-1] != e:
            events.append(e)

    # map → English
    mapping = {
        "drive":        "drove straight",
        "stop":         "stopped",
        "turn_right":   "took a slight right",
        "turn_left":    "took a slight left",
        "signal_red":   "stopped at the red light",
        "signal_green": "the signal turned green",
    }
    parts = []
    for ev in events:
        if ev in mapping:
            parts.append(mapping[ev])
        elif ev.startswith("sign_"):
            parts.append(f"passed {ev.split('_',1)[1]}")

    summary = "I " + " and ".join(parts) + "."
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary

def run(input_path: str, yolo_weights: str = None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = load_yolo(dev, yolo_weights)
    return run_clip(input_path, yolo_model, dev)

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Journey summariser")
    p.add_argument("-i","--input", required=True,
                   help="Folder of JPG frames or single MP4")
    p.add_argument("-m","--yolo-model", default=None,
                   help="Custom YOLO .pt (omit to use yolov8n)")
    args = p.parse_args()
    run(args.input, args.yolo_model)
