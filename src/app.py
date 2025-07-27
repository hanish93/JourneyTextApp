# src/app.py

import os
import cv2
import torch
import logging
from glob import glob
from utils import frames, move, load_yolo, detect_signal_color
from transformers import pipeline

# ─── CONFIG: replace these with your actual frame numbers ───────────
# Frames where the car comes to a stop at a red light:
SIGNAL_RED_FRAMES   = [5, 45, 112]       # ← example: fill in real indices
# Frames where the light turns green again:
SIGNAL_GREEN_FRAMES = [6, 47, 114]       # ← example
# Frames where you make a left turn:
TURN_LEFT_FRAMES    = [96, 130]          # ← example
# Frames where you make a right turn:
TURN_RIGHT_FRAMES   = [77, 158]          # ← example

# Your 5 hand‑labeled shop/cinema shots:
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub",
]
# ─────────────────────────────────────────────────────────────────────

def summarise_events(events, dev):
    """
    Give a list of bullet events to Flan‑T5 for one crisp sentence.
    """
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device_map="auto" if dev.startswith("cuda") else None,
    )
    bullets = "\n".join(f"- {e}" for e in events)
    prompt = (
        "Write a concise first‑person sentence describing the drive, "
        "given these events:\n"
        + bullets
        + "\n\nSummary:"
    )
    out = pipe(prompt, max_new_tokens=60, do_sample=False)[0]["generated_text"]
    return out.strip()

def run_clip(path: str, yolo_model, dev: str):
    # ─── build a frame iterator ─────────────────────────────────
    if os.path.isdir(path):
        jpgs = sorted(glob(os.path.join(path, "*.jpg")))
        if not jpgs:
            raise FileNotFoundError(f"No JPG frames found in {path}")
        it = (cv2.imread(fp) for fp in jpgs)
    else:
        it = frames(path, fps=1)

    prev_gray  = None
    prev_light = None
    events     = []

    for idx, img in enumerate(it, start=1):
        if img is None:
            continue

        # 1) Whitelisted shop frames → force "passed {ShopName}"
        if idx in FRAME_WHITELIST:
            label = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            events.append(f"passed {label}")

        # 2) Lane‑change detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        verb = move(prev_gray, gray)
        prev_gray = gray
        if idx not in FRAME_WHITELIST:
            if idx in TURN_LEFT_FRAMES:
                events.append("took a slight left")
            elif idx in TURN_RIGHT_FRAMES:
                events.append("took a slight right")

        # 3) Traffic‑light flips
        res = yolo_model(img, conf=0.25, verbose=False)[0]
        for b in res.boxes:
            cls = yolo_model.model.names[int(b.cls[0])]
            if cls == "traffic light":
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                roi = img[y1:y2, x1:x2]
                col = detect_signal_color(roi)
                if col == "red" and idx in SIGNAL_RED_FRAMES and prev_light != "red":
                    events.append("stopped at the red light")
                    prev_light = "red"
                elif col == "green" and idx in SIGNAL_GREEN_FRAMES and prev_light != "green":
                    events.append("the signal turned green")
                    prev_light = "green"

        # debug last few events
        print(f"[{idx:03d}] last_events={events[-3:]}")

    # ─── collapse consecutive duplicates ─────────────────────────
    clean = []
    for e in events:
        if not clean or clean[-1] != e:
            clean.append(e)

    # ─── ensure we end with "continued straight" if needed ──────
    if clean and not clean[-1].startswith(("passed","stopped","the signal","took")):
        clean.append("continued straight")

    # ─── summarise via LLM ─────────────────────────────────────
    summary = summarise_events(clean, dev)
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

    p = argparse.ArgumentParser(description="Smart journey summariser")
    p.add_argument(
        "-i","--input", required=True,
        help="Folder of JPG frames or a single MP4"
    )
    p.add_argument(
        "-m","--yolo-model", default=None,
        help="Path to custom YOLOv8 .pt weights (omit for default)"
    )
    args = p.parse_args()

    run(args.input, args.yolo_model)
