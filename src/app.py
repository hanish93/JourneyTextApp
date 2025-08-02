import os, cv2, torch
from glob import glob

from utils import (
    extract_frames,
    load_light_model, detect_light_state,
    load_turn_model, detect_turn_sequence,
    load_ocr, detect_signs,
    debounce, debounce_signal,
    build_narrative,
)

def run_pipeline(src):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    light_model = load_light_model(device)
    turn_model  = load_turn_model(device)
    ocr_reader  = load_ocr()

    raw_ev, raw_sig, raw_sg = [], [], []
    prev_gray=None

    # load frames
    if os.path.isdir(src):
        paths  = sorted(glob(os.path.join(src, "*.jpg")))
        frames = [cv2.imread(p) for p in paths]
    else:
        frames = list(extract_frames(src))

    # per-frame
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) turn event
        evt, prev_gray = detect_turn_sequence(prev_gray, gray, turn_model)
        raw_ev.append(evt)

        # 2) traffic-light
        raw_sig.append(detect_light_state(frame, light_model))

        # 3) signboards
        raw_sg.append(detect_signs(frame, ocr_reader))

    # debounce
    evs = debounce(raw_ev, window=3, min_count=2)
    sgs = debounce_signal(raw_sig, window=3)

    # build narrative
    story = build_narrative(evs, sgs, raw_sg)

    print("\n=== Journey Narrative ===\n")
    print(story)
    print("\n=========================\n")


if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-i","--input", required=True,
                   help="path to video file or folder of frames")
    args = p.parse_args()
    run_pipeline(args.input)
