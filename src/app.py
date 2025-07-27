# src/app.py

import os
import cv2
import torch
import numpy as np
import logging
from glob import glob
from utils import frames, load_detector, detect_signal_color

def move(prev_gray, curr_gray, dx=1.5, stop_thr=0.2):
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    dxm = flow[...,0].mean()
    mag = np.linalg.norm(flow, axis=2).mean()
    if mag < stop_thr:
        return "stop"
    if dxm > dx:
        return "turn_right"
    if dxm < -dx:
        return "turn_left"
    return "drive"

def run_clip(path: str, model, dev: str):
    # build frame iterator
    if os.path.isdir(path):
        imgs = sorted(glob(os.path.join(path, "*.jpg")))
        if imgs:
            frames_iter = (cv2.imread(f) for f in imgs)
        else:
            vids = sorted(glob(os.path.join(path, "*.mp4")))
            if not vids:
                raise FileNotFoundError(f"No .jpg or .mp4 under {path}")
            return "\n\n".join(run_clip(v, model, dev) for v in vids)
    else:
        frames_iter = frames(path, fps=1)

    prev_gray = None
    seen_signs = set()
    prev_light = None
    timeline = []

    for idx, img in enumerate(frames_iter, start=1):
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        verb = move(prev_gray, gray)
        prev_gray = gray

        # YOLO inference
        res = model(img, conf=0.25, verbose=False)[0]
        for box in res.boxes:
            cls = model.model.names[int(box.cls[0])]
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            if cls == "traffic light":
                color = detect_signal_color(crop)
                if color and color != prev_light:
                    timeline.append(f"signal_{color}")
                prev_light = color
            else:
                # must be one of your custom sign classes
                if cls not in seen_signs:
                    seen_signs.add(cls)
                    timeline.append(f"sign_{cls}")

        timeline.append(verb)
        print(f"[frame {idx:03d}] verb={verb:10s} light={prev_light or '-':6s}"
              f" new_signs={seen_signs}")

    # map events to phrases
    mapping = {
        "signal_green":"the signal turned green",
        "signal_red":"stopped at the red light",
        "drive":"drove straight",
        "stop":"came to a stop",
        "turn_right":"took a slight right",
        "turn_left":"took a slight left",
    }
    parts = []
    for ev in timeline:
        if ev in mapping:
            parts.append(mapping[ev])
        elif ev.startswith("sign_"):
            parts.append(f"passed {ev.split('_',1)[1]}")

    summary = "I " + " and ".join(parts) + "." if parts else "No events detected."
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary

def run(input_path: str, yolo_weights: str = None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_detector(dev, yolo_weights)
    return run_clip(input_path, model, dev)

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Journey summariser")
    p.add_argument("--input", "-i", required=True,
                   help="Folder of JPG frames or single MP4")
    p.add_argument("--yolo-model", "-m", default=None,
                   help="Path to your custom-trained YOLOv8 .pt file")
    args = p.parse_args()
    run(args.input, args.yolo_model)
