# src/app.py
import os, cv2, torch, logging
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

# your labeled frames
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = ["Tesco Express", "CREMA", "Townhall", "Vue", "Wool Pack Hub"]

def run_pipeline(src):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    print(f"\n=== Processing {src} (device={dev}) ===\n")

    yolo = get_yolo_model(dev)
    raw_ev, raw_sig = [], []
    prev_gray = None

    # load frames
    if os.path.isdir(src):
        paths  = sorted(glob(os.path.join(src, "*.jpg")))
        frames = [cv2.imread(p) for p in paths]
    else:
        frames = list(extract_frames(src, fps=1))

    # per‑frame
    for idx, frame in enumerate(frames, start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if idx in FRAME_WHITELIST:
            lbl = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            raw_ev.append(f"passed {lbl}")
            raw_sig.append(None)
            prev_gray = gray
            continue
        ev = detect_event_for_frame(prev_gray, gray)
        raw_ev.append(ev)
        prev_gray = gray
        sg = detect_signal_color(frame, yolo)
        raw_sig.append(sg)

    # debounce
    evs = debounce_lane_changes(raw_ev, window=3, min_count=3)
    sgs = debounce_signals(raw_sig, window=3)

    # print table
    print("STEP │ EVENT               │ SIGNAL")
    print("─────┼─────────────────────┼────────")
    for i,(e,s) in enumerate(zip(evs, sgs), start=1):
        mark = "⚑" if e.startswith("passed ") else " "
        print(f"{i:3d}  │ {mark}{e:<19} │ {s or 'none'}")

    # final summary
    print("\n―――――  Final summary  ―――――――\n")
    print(generate_long_summary(evs, sgs))
    print("\n――――――――――――――――――――\n")
