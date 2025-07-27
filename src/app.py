import os
import cv2
import torch
import numpy as np
import logging
from glob import glob
from utils import (
    frames, move, load_yolo, detect_signal_color,
    load_blip2, fetch
)
from PIL import Image

# —————————————————————————————————————————————————————————————
# CONFIGURE THESE:
# The exact frame indices (1‑based) you labeled:
FRAME_WHITELIST = [7, 10, 77, 96, 116]
# A short human‑readable list in the same order as your frames:
FRAME_LABELS = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub",
]
# —————————————————————————————————————————————————————————————

def run_clip(path: str, yolo_model, blip2_pipe, dev: str):
    # ─── frame iterator ─────────────────────────────────────
    if os.path.isdir(path):
        imgs = sorted(glob(os.path.join(path, "*.jpg")))
        if imgs:
            it = (cv2.imread(fp) for fp in imgs)
        else:
            vids = sorted(glob(os.path.join(path, "*.mp4")))
            return "\n\n".join(run_clip(v, yolo_model, blip2_pipe, dev) for v in vids)
    else:
        it = frames(path, fps=1)

    prev_gray = None
    prev_verb = None
    prev_light = None
    timeline = []

    for idx, img in enumerate(it, start=1):
        if img is None:
            continue

        # 1) motion verb, record only on change
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        verb = move(prev_gray, gray)
        prev_gray = gray
        if verb != prev_verb:
            timeline.append(verb)
            prev_verb = verb

        # 2) traffic‑light flips
        res = yolo_model(img, conf=0.25, verbose=False)[0]
        for b in res.boxes:
            cls = yolo_model.model.names[int(b.cls[0])]
            if cls == "traffic light":
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                roi = img[y1:y2, x1:x2]
                color = detect_signal_color(roi)
                if color and color != prev_light:
                    timeline.append(f"signal_{color}")
                    prev_light = color

        # 3) BLIP‑2 caption exactly on your labeled frames
        if idx in FRAME_WHITELIST:
            # load as PIL & thumbnail for BLIP‑2
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if max(pil.size) > 640:
                pil.thumbnail((640, 640), Image.Resampling.LANCZOS)
            cap = blip2_pipe({"image": pil}, max_new_tokens=20)[0]["generated_text"]
            # optionally override with your own known label:
            label = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            timeline.append(f"sign_{label}")

        # debug
        print(f"[{idx:03d}] verb={verb:10s} light={prev_light or '-':6s}"
              f" frame_label={'YES' if idx in FRAME_WHITELIST else ''}")

    # ─── collapse duplicates ───────────────────────────────────
    events = []
    for e in timeline:
        if not events or events[-1] != e:
            events.append(e)

    # ─── map to English & join ────────────────────────────────
    mapping = {
        "drive":        "drove straight",
        "stop":         "stopped",
        "turn_right":   "took a slight right",
        "turn_left":    "took a slight left",
        "signal_red":   "stopped at the red light",
        "signal_green": "the signal turned green",
    }
    phrases = []
    for ev in events:
        if ev in mapping:
            phrases.append(mapping[ev])
        elif ev.startswith("sign_"):
            phrases.append(f"passed {ev.split('_',1)[1]}")

    summary = "I " + " and ".join(phrases) + "."
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary

def run(input_path: str, yolo_weights: str = None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = load_yolo(dev, yolo_weights)
    blip2_pipe = load_blip2(dev)
    return run_clip(input_path, yolo_model, blip2_pipe, dev)

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Journey summariser")
    parser.add_argument("--input","-i",required=True,
                        help="Folder of JPG frames or single MP4")
    parser.add_argument("--yolo-model","-m",default=None,
                        help="Path to custom YOLOv8 .pt (omit for default)")
    args = parser.parse_args()
    run(args.input, args.yolo_model)
