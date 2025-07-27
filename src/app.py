import os, cv2, torch, logging
from glob import glob

from .utils import (
    extract_frames,
    detect_event_for_frame,
    debounce_lane_changes,
    detect_signal_color,
    debounce_signals,
)

# ─── YOUR MANUAL FRAMES ───────────────────────────────────────────────────
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub"
]
# ──────────────────────────────────────────────────────────────────────────

def process_frames(src, yolo_model):
    raw_events, raw_signals = [], []
    prev_gray = None

    # choose frames
    if os.path.isdir(src):
        paths = sorted(glob(os.path.join(src, "*.jpg")))
        it = (cv2.imread(p) for p in paths)
    else:
        it = extract_frames(src)

    for idx, img in enumerate(it, start=1):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # — manual “passed …” frames —
        if idx in FRAME_WHITELIST:
            label = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            raw_events.append(f"passed {label}")
            raw_signals.append(None)
            prev_gray = gray
            continue

        # — motion event —
        ev = detect_event_for_frame(prev_gray, gray)
        raw_events.append(ev)
        prev_gray = gray

        # — signal color —
        col = detect_signal_color(img, yolo_model)
        raw_signals.append(col)

    # debounce (remove spurious flickers)
    events = debounce_lane_changes(raw_events)
    signals = debounce_signals(raw_signals)
    return events, signals

def run_pipeline(src):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Frame‑by‑frame (device={dev}) ===\n")
    # only need YOLO for traffic lights
    from .utils import get_landmark_models
    yolo, _ = get_landmark_models(dev)

    events, signals = process_frames(src, yolo)

    # header
    print("STEP │ EVENT               │ SIGNAL")
    print("─────┼─────────────────────┼────────")

    # per‑frame
    for i, (ev, sig) in enumerate(zip(events, signals), start=1):
        flag = "⚑" if ev.startswith("passed ") else " "
        sig_str = sig or "none"
        print(f"{i:3d}  │ {flag}{ev:<19} │ {sig_str}")

    # final summary (reuse your helper)
    from .utils import generate_long_summary
    print("\n―――――  Final summary  ―――――――\n")
    print(generate_long_summary(events))
    print("\n――――――――――――――――――――\n")

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Simple journey events+signals")
    p.add_argument(
        "-i", "--input", required=True,
        help="Path to .mp4 video or folder of .jpg frames"
    )
    args = p.parse_args()
    run_pipeline(args.input)
