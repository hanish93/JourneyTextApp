import os
import cv2
import torch
import logging
from glob import glob

from src.utils import (
    extract_frames,
    detect_event_for_frame,
    debounce_lane_changes,
    get_yolo_model,
    detect_signal_color,
    debounce_signals,
    generate_long_summary,
)

# ─── these are your manually‑labelled key frames ─────────────────────────────
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express", "CREMA", "Townhall", "Vue", "Wool Pack Hub"
]

def run_pipeline(src):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    print(f"\n=== Processing {src} (device={dev}) ===\n")

    # load your YOLOv8n detector
    yolo = get_yolo_model(dev)

    raw_ev, raw_sig = [], []
    prev_gray = None

    # read frames either from folder or video
    if os.path.isdir(src):
        paths  = sorted(glob(os.path.join(src, "*.jpg")))
        frames = [cv2.imread(p) for p in paths]
    else:
        frames = list(extract_frames(src, fps=1))

    # iterate & detect
    for idx, frame in enumerate(frames, start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # force your labelled landmarks
        if idx in FRAME_WHITELIST:
            lbl = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            raw_ev.append(f"passed {lbl}")
            raw_sig.append(None)
            prev_gray = gray
            continue

        # motion‑based event
        ev = detect_event_for_frame(prev_gray, gray)
        raw_ev.append(ev)
        prev_gray = gray

        # vision‑based signal
        sg = detect_signal_color(frame, yolo)
        raw_sig.append(sg)

    # debounce turns & signals
    evs  = debounce_lane_changes(raw_ev, window=3, min_count=3)
    sgs  = debounce_signals(raw_sig, window=3)

    # print per‑frame
    print("STEP │ EVENT               │ SIGNAL")
    print("─────┼─────────────────────┼────────")
    for i,(e,s) in enumerate(zip(evs, sgs), start=1):
        mark = "⚑" if e.startswith("passed ") else " "
        print(f"{i:3d}  │ {mark}{e:<19} │ {s or 'none'}")

    # final sentence
    print("\n―――――  Final summary  ―――――――\n")
    print(generate_long_summary(evs, sgs))
    print("\n――――――――――――――――――――\n")
