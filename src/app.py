import argparse, torch
from utils import (
    extract_frames,
    detect_event_for_frame, debounce_lane_changes,
    get_yolo_model, detect_signal_color, debounce_signals,
    summarise_frames,
)
import cv2

def run_pipeline(target):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Models] loading YOLOv8n on {device}…")
    yolo = get_yolo_model(device)
    print("[Models] done.\n")

    prev_gray = None
    raw_events, raw_signals = [], []

    print("[Frames] extracting & analyzing…")
    for i, frame in enumerate(extract_frames(target, fps=1), start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) motion event
        ev = detect_event_for_frame(prev_gray, gray)
        raw_events.append(ev)

        # 2) signal color
        sig = detect_signal_color(frame, yolo)
        raw_signals.append(sig)

        prev_gray = gray

    print("[Frames] debouncing…")
    events = debounce_lane_changes(raw_events)
    signals = debounce_signals(raw_signals)

    print("\n=== Final summary ===")
    summary = summarise_frames(events, signals)
    print(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Journey summariser")
    p.add_argument("--input", required=True,
                   help="Path to .mp4 or folder of .jpg")
    args = p.parse_args()
    run_pipeline(args.input)
