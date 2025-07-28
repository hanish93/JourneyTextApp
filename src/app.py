import os, cv2, torch, logging
from utils import (
    extract_frames,
    detect_event,
    load_signal_model, detect_signal_color,
    FRAME_WHITELIST, FRAME_LABELS,
    debounce_events, debounce_signals,
    generate_summary,
)

def run_pipeline(src):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    print(f"\n=== Processing {src} on {device} ===\n")

    # load YOLO once for signal detection
    yolo_sig = load_signal_model(device)

    raw_ev, raw_sig = [], []
    prev_gray = None

    # load frames
    frames = list(extract_frames(src))

    for idx, frame in enumerate(frames, start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) forced label?
        if idx in FRAME_WHITELIST:
            lbl = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            raw_ev.append(f"passed {lbl}")
            raw_sig.append(None)
            prev_gray = gray
            continue

        # 2) motion event
        ev = detect_event(prev_gray, gray)
        raw_ev.append(ev)
        prev_gray = gray

        # 3) traffic‑light color
        sig = detect_signal_color(frame, yolo_sig)
        raw_sig.append(sig)

    # debounce
    evs = debounce_events(raw_ev)
    sigs= debounce_signals(raw_sig)

    # print step table
    print("FRAME │ EVENT               │ SIGNAL")
    print("──────┼─────────────────────┼────────")
    for i,(e,s) in enumerate(zip(evs,sigs), start=1):
        mark = "⚑" if e.startswith("passed ") else " "
        print(f"{i:5d} │{mark}{e:<20}│ {s or 'none'}")

    # final summary
    print("\n――――――  Final summary  ――――――――\n")
    print(generate_summary(evs, sigs))
    print("\n" + "―"*40 + "\n")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,help="video.mp4 or folder of .jpg")
    args=p.parse_args()
    run_pipeline(args.input)
