# JourneyTextApp/src/app.py

import os, cv2, torch
from glob import glob

from src.utils import (
    extract_frames,
    load_light_model, detect_light_state,
    load_turn_model, detect_turn,
    load_ocr, detect_signs,
    debounce_list, debounce_signals,
    build_narrative,
)

def run_pipeline(src):
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    light_model = load_light_model(device)
    turn_model  = load_turn_model(device)
    ocr_reader  = load_ocr()

    raw_ev, raw_sig, raw_sg = [], [], []
    prev_gray = None

    if os.path.isdir(src):
        paths  = sorted(glob(os.path.join(src, "*.jpg")))
        frames = [cv2.imread(p) for p in paths]
    else:
        frames = list(extract_frames(src))

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) turn
        ev = detect_turn(prev_gray, gray)
        raw_ev.append(ev)
        prev_gray = gray

        # 2) light
        raw_sig.append(detect_light_state(frame, light_model))

        # 3) OCR
        raw_sg.append(detect_signs(frame, ocr_reader))

    evs = debounce_list(raw_ev, window=3, min_count=2)
    sgs = debounce_signals(raw_sig, window=3)

    story = build_narrative(evs, sgs, raw_sg)

    print("\n=== Journey Narrative ===\n")
    print(story)
    print("\n=========================\n")


if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-i","--input", required=True,
                   help="video or folder of frames")
    args = p.parse_args()
    run_pipeline(args.input)
