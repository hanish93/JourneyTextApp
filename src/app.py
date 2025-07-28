import os
import cv2
import torch
import logging
from glob import glob

from .utils import (
    extract_frames,
    load_light_model, detect_light_state,
    load_seg_model, detect_road_mask,
    load_turn_model, detect_turn,
    generate_long_summary
)

# your manual frames
FRAME_WHITELIST = [7,10,77,96,116]
FRAME_LABELS   = [
    "Tesco Express","CREMA","Townhall","Vue","Wool Pack Hub"
]

def run_pipeline(src):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    print(f"\n=== Processing {src} (device={dev}) ===\n")

    # load all models
    light_m = load_light_model(dev)
    seg_m   = load_seg_model(dev)
    turn_m  = load_turn_model(dev)

    # buffers
    events, lights = [], []
    prev_gray = cur_gray = next_gray = None

    # prepare iterators
    if os.path.isdir(src):
        files = sorted(glob(os.path.join(src,"*.jpg")))
        frames = [cv2.imread(f) for f in files]
    else:
        frames = list(extract_frames(src))

    # precompute grays
    grays = [cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames]
    for i,frame in enumerate(frames):
        # manual injection
        if (i+1) in FRAME_WHITELIST:
            lbl = FRAME_LABELS[FRAME_WHITELIST.index(i+1)]
            events.append(f"passed {lbl}")
            lights.append(None)
            continue

        # turn detection (need 3 frames)
        if i>0 and i< len(frames)-1:
            ev = detect_turn(grays[i-1], grays[i], grays[i+1], turn_m, dev)
            if ev == "straight":
                # fallback to stop/drive by small mag
                ev = "drive"
        else:
            ev = "drive"
        events.append(ev)

        # signal detection
        lt = detect_light_state(frame, light_m)
        lights.append(lt)

    # debounce events & lights
    from .utils import debounce_lane_changes, debounce_signals
    events = debounce_lane_changes(events)
    lights = debounce_signals(lights)

    # print per-frame
    print("STEP │ EVENT               │ LIGHT")
    print("─────┼─────────────────────┼────────")
    for idx,(ev,lt) in enumerate(zip(events,lights), start=1):
        mark = "⚑" if ev.startswith("passed ") else " "
        print(f"{idx:3d}  │ {mark}{ev:<19} │ {lt or 'none'}")

    # summary
    print("\n――――― Final Summary ―――――\n")
    print(generate_long_summary(events, lights))
    print("\n――――――――――――――――――――\n")
