# src/app.py
import cv2
import torch

from src.utils import (
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
    """
    Main entrypoint: given a folder or video path, extract frames,
    detect motion events + traffic-light colors, debounce them,
    and print both a frame-by-frame table and a final human summary.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo   = load_signal_model(device)

    raw_ev, raw_sig = [], []
    prev_gray = None

    # 1fps frame extraction
    frames = list(extract_frames(src, fps=1))
    for idx, frame in enumerate(frames, start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # inject your 5 manual keyframes
        if idx in FRAME_WHITELIST:
            label = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            raw_ev .append(f"passed {label}")
            raw_sig.append(None)
            prev_gray = gray
            continue

        # detect motion event
        ev = detect_event(prev_gray, gray)
        raw_ev.append(ev)
        prev_gray = gray

        # detect traffic-light color
        sig = detect_signal_color(frame, yolo)
        raw_sig.append(sig)

    # debounce single-frame spurious turns/signals
    evs = debounce_events(raw_ev, window=3, min_count=3)
    sgs = debounce_signals(raw_sig, window=3)

    # print debug table
    print("FRAME │ EVENT               │ SIGNAL")
    print("──────┼─────────────────────┼────────")
    for i, (e, s) in enumerate(zip(evs, sgs), start=1):
        mark = "⚑" if e.startswith("passed ") else " "
        print(f"{i:5d} │ {mark}{e:<19} │ {s or 'none'}")

    # print final human summary
    print("\nFinal summary:\n", generate_summary(evs, sgs))
