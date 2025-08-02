import os
import cv2
import torch
import argparse
from glob import glob

from .utils import (
    extract_frames,
    detect_event,
    load_signal_model,
    detect_signal_color,
    debounce_events,
    debounce_signals,
    generate_summary,
)

def run_pipeline(src):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo   = load_signal_model(device)

    frames = []
    if os.path.isdir(src):
        for p in sorted(glob(os.path.join(src, "*.jpg"))):
            img = cv2.imread(p)
            if img is not None:
                frames.append(img)
    else:
        frames = list(extract_frames(src))

    raw_ev, raw_sig = [], []
    prev_gray = None

    # per-frame inference
    for idx, frame in enumerate(frames, start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ev = detect_event(prev_gray, gray)
        raw_ev.append(ev)
        prev_gray = gray

        sig = detect_signal_color(frame, yolo)
        raw_sig.append(sig)

    # debounce out noise
    evs = debounce_events(raw_ev, window=3, min_count=3)
    sgs = debounce_signals(raw_sig, window=3, min_count=2)

    # print per-frame table
    print("FRAME │ EVENT               │ SIGNAL")
    print("──────┼─────────────────────┼────────")
    for i,(e,s) in enumerate(zip(evs, sgs), start=1):
        print(f"{i:5d} │ {e:<19} │ {s or 'none'}")

    # final summary
    print("\nFinal summary:\n")
    print(generate_summary(evs, sgs))
    print("\n" + "─"*40 + "\n")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input","-i",required=True,
                   help="path to video file or frames folder")
    args=p.parse_args()
    run_pipeline(args.input)
