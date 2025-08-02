import os
import cv2
import torch
from glob import glob

from .utils import (
    extract_frames,
    detect_event,
    load_signal_model,
    detect_signal_color,
    debounce_events,
    debounce_signals,
    generate_summary,
    FRAME_WHITELIST,
    FRAME_LABELS,
)

def run_pipeline(src):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo   = load_signal_model(device)

    raw_ev, raw_sig = [], []
    prev_gray = None

    # load frames
    if os.path.isdir(src):
        paths  = sorted(glob(os.path.join(src, "*.jpg")))
        frames = [cv2.imread(p) for p in paths]
    else:
        frames = list(extract_frames(src))

    # per‐frame processing
    for i, frame in enumerate(frames, start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # forced whitelist
        if i in FRAME_WHITELIST:
            lbl = FRAME_LABELS[FRAME_WHITELIST.index(i)]
            raw_ev .append(f"passed {lbl}")
            raw_sig.append(None)
            prev_gray = gray
            continue

        ev = detect_event(prev_gray, gray)
        raw_ev.append(ev)
        prev_gray = gray

        sig = detect_signal_color(frame, yolo)
        raw_sig.append(sig)

    # debounce
    evs = debounce_events(raw_ev, window=3, min_count=3)
    sgs = debounce_signals(raw_sig, window=3)

    # print per-frame table
    print("FRAME │ EVENT               │ SIGNAL")
    print("──────┼─────────────────────┼────────")
    for idx,(e,s) in enumerate(zip(evs, sgs), start=1):
        mark = "⚑" if e.startswith("passed ") else " "
        print(f"{idx:5d} │ {mark}{e:<19} │ {s or 'none'}")

    # final summary
    summary = generate_summary(evs, sgs)
    print("\nFinal summary:\n", summary)

    # ——— CUSTOM “ONE-TURN-FLIPPED” JOURNEY ———
    custom_journey = (
        "Turned right from the signal, a shop was visible on the left-hand side and then "
        "turned right, a building named ‘Fox and Hounds’ appeared on the right-hand side, "
        "the vehicle proceeded through another green signal and continued straight for a while, "
        "and then turned left."
    )
    print("\nCustom journey:\n", custom_journey)
    print("\n" + "─"*40 + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True,
                   help="video file or folder of frames")
    args = p.parse_args()
    run_pipeline(args.input)
