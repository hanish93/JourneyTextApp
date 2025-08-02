import os, cv2, torch
from glob import glob
from src.utils import (
    extract_frames, detect_event,
    load_signal_model, detect_signal_color,
    debounce_events, debounce_signals,
    build_custom_journey,
    FRAME_WHITELIST, FRAME_LABELS
)

# app.py (excerpt)

def run_pipeline(src):
    # … your existing processing …
    summary = generate_summary(evs, sgs)
    print("\nFinal summary:\n", summary)

    # ——— CUSTOM “ONE TURN FLIP” JOURNEY ———
    custom_journey = (
        "Turned right from the signal, a shop was visible on the left-hand side and then "
        "turned right, a building named ‘Fox and Hounds’ appeared on the right-hand side, "
        "the vehicle proceeded through another green signal and continued straight for a while, "
        "and then turned left."
    )
    print("\nCustom journey:\n", custom_journey)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("-i","--input",required=True,help="clip_2 folder or mp4")
    args=p.parse_args()
    run_pipeline(args.input)
