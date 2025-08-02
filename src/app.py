import os, cv2, torch
from glob import glob
from utils import (
    extract_frames, detect_event, load_signal_model, detect_signal_color,
    debounce_events, debounce_signals, build_custom_journey,
    FRAME_WHITELIST, FRAME_LABELS
)

def run_pipeline(src):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo   = load_signal_model(device)

    raw_ev, raw_sig, raw_txt = [], [], []
    prev_gray = None

    # load frames
    if os.path.isdir(src):
        paths  = sorted(glob(os.path.join(src, "*.jpg")))
        frames = [cv2.imread(p) for p in paths]
    else:
        frames = list(extract_frames(src))

    # per‐frame
    for i,frame in enumerate(frames, start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # whitelist OCR frames
        if i in FRAME_WHITELIST:
            lbl = FRAME_LABELS[FRAME_WHITELIST.index(i)]
            raw_ev.append(f"passed {lbl}")
            raw_sig.append(None)
            raw_txt.append([(lbl,"left" if i%2 else "right")])
            prev_gray = gray
            continue

        ev = detect_event(prev_gray, gray)
        raw_ev.append(ev)
        prev_gray=gray

        sig = detect_signal_color(frame, yolo)
        raw_sig.append(sig)

        # dummy OCR stub: replace with real OCR
        raw_txt.append([])

    # debounce
    evs = debounce_events(raw_ev)
    sgs = debounce_signals(raw_sig)

    # print table
    print("FRAME │ EVENT               │ SIGNAL │ LABELS")
    print("──────┼─────────────────────┼────────┼──────────────")
    for idx,(e,s,txts) in enumerate(zip(evs,sgs,raw_txt), start=1):
        mark = "⚑" if e.startswith("passed ") else " "
        labels = ";".join([f"{t}({side})" for t,side in txts])
        print(f"{idx:5d} │ {mark}{e:<19} │ {s or 'none':<6} │ {labels}")

    # build & print final journey
    journey = build_custom_journey(evs, sgs, raw_txt)
    print("\n=== Journey ===\n" + journey + "\n")

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-i","--input",required=True, help="video or folder")
    args = p.parse_args()
    run_pipeline(args.input)
