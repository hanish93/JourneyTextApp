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

# ─── MANUAL “PASSED X” FRAMES ─────────────────────────────────────────────
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
    raw_events, raw_signals = [], []
    prev_gray = None

    # choose frames from folder or video
    if os.path.isdir(src):
        paths = sorted(glob(os.path.join(src, "*.jpg")))
        it = (cv2.imread(p) for p in paths)
    else:
        it = extract_frames(src)

    for idx, img in enumerate(it, start=1):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # manual frame?
        if idx in FRAME_WHITELIST:
            lbl = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            raw_events .append(f"passed {lbl}")
            raw_signals.append(None)
            prev_gray = gray
            continue

        # motion
        ev = detect_event_for_frame(prev_gray, gray)
        raw_events.append(ev)
        prev_gray = gray

        # signal
        sig = detect_signal_color(img, yolo)
        raw_signals.append(sig)

    # debounce spurious detections
    events  = debounce_lane_changes(raw_events)
    signals = debounce_signals(raw_signals)
    return events, signals

def run_pipeline(src):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    print(f"\n=== Frame‑by‑frame (device={dev}) ===\n")

    # load YOLO once for traffic lights
    yolo = get_yolo_model(dev)

    events, signals = process_frames(src, yolo)

    # print table
    print("STEP │ EVENT               │ SIGNAL")
    print("─────┼─────────────────────┼────────")
    for i, (ev, sig) in enumerate(zip(events, signals), start=1):
        mark = "⚑" if ev.startswith("passed ") else " "
        sig_str = sig or "none"
        print(f"{i:3d}  │ {mark}{ev:<19} │ {sig_str}")

    # final summary
    print("\n―――――  Final summary  ―――――――\n")
    print(generate_long_summary(events, signals))
    print("\n――――――――――――――――――――\n")

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    p = argparse.ArgumentParser(description="Journey summariser")
    p.add_argument(
        "-i","--input", required=True,
        help="Path to .mp4 video or folder of .jpg frames"
    )
    args = p.parse_args()
    run_pipeline(args.input)
