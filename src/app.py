import os
import cv2
import torch
import logging
from glob import glob

from .utils import (
    extract_frames,
    detect_event_for_frame,
    debounce_lane_changes,
    get_yolo_model,
    detect_signal_color,
    debounce_signals,
    generate_long_summary,
)

# ─── YOUR MANUAL “PASSED X” FRAMES ────────────────────────────────────────
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub",
]
# ────────────────────────────────────────────────────────────────────────────

def process_frames(src, yolo):
    raw_ev, raw_sig = [], []
    prev_gray = None

    if os.path.isdir(src):
        paths = sorted(glob(os.path.join(src, "*.jpg")))
        it = (cv2.imread(p) for p in paths)
    else:
        it = extract_frames(src)

    for idx, img in enumerate(it, start=1):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if idx in FRAME_WHITELIST:
            lbl = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            raw_ev .append(f"passed {lbl}")
            raw_sig.append(None)
            prev_gray = gray
            continue

        ev = detect_event_for_frame(prev_gray, gray)
        raw_ev.append(ev)
        prev_gray = gray

        sg = detect_signal_color(img, yolo)
        raw_sig.append(sg)

    events  = debounce_lane_changes(raw_ev)
    signals = debounce_signals(raw_sig)
    return events, signals

def run_pipeline(src):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    print(f"\n=== FRAME‑BY‑FRAME (device={dev}) ===\n")

    yolo = get_yolo_model(dev)
    events, signals = process_frames(src, yolo)

    print("STEP │ EVENT               │ SIGNAL")
    print("─────┼─────────────────────┼────────")
    for i,(ev,sg) in enumerate(zip(events, signals), start=1):
        mark = "⚑" if ev.startswith("passed ") else " "
        s    = sg or "none"
        print(f"{i:3d}  │ {mark}{ev:<19} │ {s}")

    print("\n―――――  Final summary  ―――――――\n")
    print(generate_long_summary(events, signals))
    print("\n――――――――――――――――――――\n")

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    p = argparse.ArgumentParser(description="Journey summariser")
    p.add_argument(
        "-i","--input", required=True,
        help="Path to .mp4 or folder of .jpg frames"
    )
    args = p.parse_args()
    run_pipeline(args.input)
