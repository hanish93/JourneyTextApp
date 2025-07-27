# src/app.py

import os
import logging
from glob import glob
from utils import frames

# ─── YOUR “MANUAL” EVENT LISTS ────────────────────────────────
# Frames where you stopped at a red light:
SIGNAL_RED_FRAMES   = [ 5 ]             # ← example: replace with your real indices
# Frames where it turned green:
SIGNAL_GREEN_FRAMES = [ 6 ]
# Frames where you turned left:
TURN_LEFT_FRAMES    = [ 96 ]
# Frames where you turned right:
TURN_RIGHT_FRAMES   = [ 77 ]

# Frames → shop/cinema names:
SIGN_FRAMES = {
    7:  "Tesco Express",
    10: "CREMA",
    77: "Townhall",
    96: "Vue",
   116: "Wool Pack Hub",
}
# ───────────────────────────────────────────────────────────────

def run_clip(path: str):
    # build frame iterator
    seq = list(frames(path, fps=1))

    events = []
    for idx, _ in enumerate(seq, start=1):
        # 1) red signal?
        if idx in SIGNAL_RED_FRAMES:
            events.append("signal_red")
        # 2) green signal?
        if idx in SIGNAL_GREEN_FRAMES:
            events.append("signal_green")
        # 3) left turn?
        if idx in TURN_LEFT_FRAMES:
            events.append("turn_left")
        # 4) right turn?
        if idx in TURN_RIGHT_FRAMES:
            events.append("turn_right")
        # 5) passed a labeled sign?
        if idx in SIGN_FRAMES:
            events.append(f"sign_{SIGN_FRAMES[idx]}")

    # collapse consecutive duplicates
    timeline = []
    for e in events:
        if not timeline or timeline[-1] != e:
            timeline.append(e)

    # map to human phrases (drive → continued straight at end)
    mapping = {
        "signal_red":   "stopped at the red light",
        "signal_green": "the signal turned green",
        "turn_left":    "took a slight left",
        "turn_right":   "took a slight right",
    }

    phrases = []
    for i, ev in enumerate(timeline):
        if ev in mapping:
            phrases.append(mapping[ev])
        elif ev.startswith("sign_"):
            name = ev.split("_",1)[1]
            phrases.append(f"passed {name}")

    # always finish with “continued straight”
    if not timeline or timeline[-1].startswith("sign_") or timeline[-1].startswith("signal"):
        phrases.append("continued straight")

    summary = "I " + " and ".join(phrases) + "."
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary

def run(input_path: str):
    return run_clip(input_path)

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Forceful journey summariser")
    p.add_argument("-i","--input", required=True,
                   help="Folder of JPG frames or single MP4")
    args = p.parse_args()
    run(args.input)
