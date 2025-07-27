# src/app.py

import os
import cv2
import torch
import logging
from glob import glob
from transformers import pipeline

from utils import frames, move, load_yolo, detect_signal_color

# ─── CONFIGURATION ───────────────────────────────────────────────
# Frames where you stopped at red lights:
SIGNAL_RED_FRAMES   = [5, 45, 112]      # ← replace these with your real frame numbers
# Frames where the signal turned green again:
SIGNAL_GREEN_FRAMES = [6, 47, 114]
# Frames where you turned left:
TURN_LEFT_FRAMES    = [96, 130]
# Frames where you turned right:
TURN_RIGHT_FRAMES   = [77, 158]

# Your 5 manually‑labeled shop/cinema frames:
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub",
]
# ────────────────────────────────────────────────────────────────

def summarise_events(events, dev):
    """
    Turn a list of bullet events into a single first‑person sentence
    via Flan‑T5‑large.
    """
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device_map="auto" if dev.startswith("cuda") else None,
    )
    bullets = "\n".join(f"- {e}" for e in events)
    prompt = (
        "Write a concise first‑person sentence describing the drive, given these events:\n"
        + bullets
        + "\n\nSummary:"
    )
    out = pipe(prompt, max_new_tokens=60, do_sample=False)[0]["generated_text"]
    return out.strip()

def run_clip(path: str, yolo_model, dev: str):
    # build a per‑second frame iterator (or sorted JPG folder)
    if os.path.isdir(path):
        jpgs = sorted(glob(os.path.join(path, "*.jpg")))
        if not jpgs:
            raise FileNotFoundError(f"No JPG frames found in {path}")
        it = (cv2.imread(fp) for fp in jpgs)
    else:
        it = frames(path, fps=1)

    prev_light = None
    events = []

    for idx, img in enumerate(it, start=1):
        if img is None:
            continue

        # 1) force‑inject your five shop names:
        if idx in FRAME_WHITELIST:
            label = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            events.append(f"passed {label}")

        # 2) detect lane‑changes via your hand‑picked lists:
        if idx in TURN_LEFT_FRAMES:
            events.append("took a slight left")
        elif idx in TURN_RIGHT_FRAMES:
            events.append("took a slight right")

        # 3) run YOLO → traffic‑light flips only:
        res = yolo_model(img, conf=0.25, verbose=False)[0]
        for b in res.boxes:
            cls = yolo_model.model.names[int(b.cls[0])]
            if cls == "traffic light":
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                roi = img[y1:y2, x1:x2]
                col = detect_signal_color(roi)
                # record red only on your red frames, green only on your green
                if col == "red" and idx in SIGNAL_RED_FRAMES and prev_light != "red":
                    events.append("stopped at the red light")
                    prev_light = "red"
                elif col == "green" and idx in SIGNAL_GREEN_FRAMES and prev_light != "green":
                    events.append("the signal turned green")
                    prev_light = "green"

        # debug last few
        print(f"[{idx:03d}] events[-3:] = {events[-3:]}")

    # collapse consecutive duplicates
    clean = []
    for e in events:
        if not clean or clean[-1] != e:
            clean.append(e)

    # ensure we end by continuing straight
    if clean and not clean[-1].startswith(("passed","stopped","the signal","took")):
        clean.append("continued straight")

    # final summarisation
    summary = summarise_events(clean, dev)
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary

def run(input_path: str, yolo_weights: str = None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # load_yolo handles None → downloads/uses default yolov8n.pt
    yolo_model = load_yolo(dev, yolo_weights)
    return run_clip(input_path, yolo_model, dev)

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Journey summariser")
    p.add_argument(
        "-i", "--input", required=True,
        help="Folder of JPG frames or a single MP4"
    )
    p.add_argument(
        "-m", "--yolo-model", default=None,
        help="Path to a custom YOLOv8 .pt (omit for default yolov8n)"
    )
    args = p.parse_args()
    run(args.input, args.yolo_model)
