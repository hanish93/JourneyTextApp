import argparse
import torch
import cv2
from .utils import (
    extract_frames,
    detect_event_for_frame, debounce_lane_changes,
    get_yolo_model, detect_signal_color, debounce_signals,
    summarise_frames,
)

def run_pipeline(target):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Models] loading YOLOv8n on {device}…")
    yolo = get_yolo_model(device)
    print("[Models] done.\n")

    prev = None
    raw_events, raw_signals = [], []
    print("[Frames] extracting & analysing…")
    for frame in extract_frames(target, fps=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_events.append(detect_event_for_frame(prev, gray))
        raw_signals.append(detect_signal_color(frame, yolo))
        prev = gray

    print("[Frames] debouncing…")
    ev = debounce_lane_changes(raw_events)
    sg = debounce_signals(raw_signals)

    print("\n=== Final summary ===")
    print(summarise_frames(ev, sg))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Path to .mp4 or folder of .jpg frames")
    args = p.parse_args()
    run_pipeline(args.input)
