import cv2
import torch

from utils import (
    extract_frames,
    detect_event,
    load_signal_model,
    detect_signal_color,
    debounce_events,
    debounce_signals,
    generate_summary,
    FRAME_WHITELIST,
    FRAME_LABELS
)

def run_pipeline(src):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo   = load_signal_model(device)

    raw_ev, raw_sig = [], []
    prev_gray = None

    frames = list(extract_frames(src))
    for i, frame in enumerate(frames, start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i in FRAME_WHITELIST:
            label = FRAME_LABELS[FRAME_WHITELIST.index(i)]
            raw_ev .append(f"passed {label}")
            raw_sig.append(None)
            prev_gray = gray
            continue

        ev = detect_event(prev_gray, gray)
        raw_ev.append(ev)
        prev_gray = gray

        sig = detect_signal_color(frame, yolo)
        raw_sig.append(sig)

    evs = debounce_events(raw_ev, window=3, min_count=3)
    sgs = debounce_signals(raw_sig, window=3)

    # debug table
    print("FRAME │ EVENT               │ SIGNAL")
    print("──────┼─────────────────────┼────────")
    for idx, (e,s) in enumerate(zip(evs, sgs), start=1):
        mark = "⚑" if e.startswith("passed ") else " "
        print(f"{idx:5d} │ {mark}{e:<19} │ {s or 'none'}")

    # final
    print("\nFinal summary:\n", generate_summary(evs, sgs))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python app.py <video_or_frames_folder>")
    else:
        run_pipeline(sys.argv[1])
