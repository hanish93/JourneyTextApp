# src/app.py

import os
import cv2
import torch
import numpy as np
import logging
from glob import glob
from utils import frames, load_yolo, detect_signal_color, load_ocr, ocr_signs, move

# Your whitelist of exactly the few signs you care about
WHITELIST_SIGNS = ["Tesco Express", "CREMA", "Vue", "Townhall"]


def run_clip(path: str, model, ocr_reader, dev: str):
    # build the frame iterator (unchanged)
    if os.path.isdir(path):
        imgs = sorted(glob(os.path.join(path, "*.jpg")))
        if imgs:
            it = (cv2.imread(f) for f in imgs)
        else:
            vids = sorted(glob(os.path.join(path, "*.mp4")))
            return "\n\n".join(run_clip(v, model, ocr_reader, dev) for v in vids)
    else:
        it = frames(path, fps=1)

    prev_gray = None
    prev_verb = None
    prev_light = None
    seen_signs = set()
    raw_timeline = []

    for idx, img in enumerate(it, start=1):
        if img is None:
            continue

        # 1) Motion verb
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        verb = move(prev_gray, gray)
        prev_gray = gray
        raw_timeline.append(verb)

        # 2) Traffic‑light color
        out = model(img, conf=0.25, verbose=False)[0]
        for b in out.boxes:
            cls = model.model.names[int(b.cls[0])]
            if cls == "traffic light":
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                roi = img[y1:y2, x1:x2]
                color = detect_signal_color(roi)
                if color:
                    raw_timeline.append(f"signal_{color}")
                    prev_light = color

        # 3) OCR + whitelist
        texts = ocr_signs(img, ocr_reader)
        for t in texts:
            for key in WHITELIST_SIGNS:
                if key.lower() in t.lower() and key not in seen_signs:
                    seen_signs.add(key)
                    raw_timeline.append(f"sign_{key}")

        # debug progress
        print(f"[frame {idx:03d}] verb={verb:10s} light={prev_light or '-':6s}"
              f" seen={list(seen_signs)}")

    # ─── Collapse consecutive duplicates ───────────────────────────
    timeline = []
    for e in raw_timeline:
        if not timeline or e != timeline[-1]:
            timeline.append(e)

    # Optionally drop all pure “stop” events, if still too chatty:
    # timeline = [e for e in timeline if e != "stop"]

    # ─── Map events → English phrases ────────────────────────────
    mapping = {
        "drive":        "drove straight",
        "stop":         "came to a stop",
        "turn_right":   "took a slight right",
        "turn_left":    "took a slight left",
        "signal_red":   "stopped at the red light",
        "signal_green": "the signal turned green",
    }

    phrases = []
    for ev in timeline:
        if ev in mapping:
            phrases.append(mapping[ev])
        elif ev.startswith("sign_"):
            phrases.append(f"passed {ev.split('_',1)[1]}")

    summary = "I " + " and ".join(phrases) + "." if phrases else "No events detected."
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary


def run(input_path: str, yolo_weights: str = None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_yolo(dev, yolo_weights)
    ocr_reader = load_ocr()
    return run_clip(input_path, model, ocr_reader, dev)


if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Journey summariser")
    parser.add_argument("--input", "-i", required=True,
                        help="Folder of JPG frames or single MP4")
    parser.add_argument("--yolo-model", "-m", default=None,
                        help="Path to custom YOLOv8 .pt weights (omit for default)")
    args = parser.parse_args()
    run(args.input, args.yolo_model)
