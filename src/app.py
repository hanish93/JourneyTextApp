import argparse, torch, cv2, os
from glob import glob

from .utils import (
    extract_frames,
    detect_event_for_frame, debounce_lane_changes,
    get_yolo_model, detect_signal_color, debounce_signals,
    generate_long_summary,
)

# Frames you manually labeled → inject their names
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express", "CREMA", "Townhall", "Vue", "Wool Pack Hub"
]

def run_pipeline(src):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Models] loading YOLOv8n on {device}…")
    yolo = get_yolo_model(device)
    print("[Models] done.\n")

    # load frames
    if os.path.isdir(src):
        paths = sorted(glob(os.path.join(src,"*.jpg")))
        frames = [cv2.imread(p) for p in paths]
    else:
        frames = list(extract_frames(src, fps=1))

    raw_ev, raw_sig = [], []
    prev_gray = None

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

    events  = debounce_lane_changes(raw_ev)
    signals = debounce_signals(raw_sig)

    # per-frame table
    print("STEP │ EVENT               │ SIGNAL")
    print("─────┼─────────────────────┼────────")
    for i,(e,s) in enumerate(zip(events,signals), start=1):
        mark = "⚑" if e.startswith("passed ") else " "
        print(f"{i:3d}  │{mark}{e:<19}│ {s or 'none'}")

    # final summary
    print("\n―――――  Final summary  ―――――――\n")
    print(generate_long_summary(events, signals))
    print("\n――――――――――――――――――――\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Path to .mp4 or directory of .jpg frames")
    args = p.parse_args()
    run_pipeline(args.input)
