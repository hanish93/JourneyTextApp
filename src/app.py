import os
import cv2
import torch
import logging
from glob import glob

from utils import (
    extract_frames,
    detect_event_for_frame,
    debounce_lane_changes,
    get_signal_model,
    detect_signal_color,
    debounce_signals,
    get_landmark_models,
    detect_landmarks_for_frame,
    get_caption_models,
    generate_caption_for_frame,
    get_scene_model,
    classify_scene_for_frame,
    summarise_journey,
    generate_long_summary
)

# ─── MANUAL FRAMES ───────────────────────────────────────────────────────
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub",
]
# ─────────────────────────────────────────────────────────────────────────

def load_models(device):
    print("[Models] Loading…")
    # YOLO‑landmarks + OCR
    yolo_obj, ocr = get_landmark_models(device)
    # YOLO‑signals
    sig_mod = get_signal_model(device)
    # BLIP caption
    cap_proc, cap_mod = get_caption_models(device)
    # Places365 scene
    scene_mod, scene_cls = get_scene_model(device)
    print("[Models] Done.")
    return {
        "yolo": yolo_obj,
        "ocr": ocr,
        "signal": sig_mod,
        "cap_proc": cap_proc,
        "cap_mod": cap_mod,
        "scene_mod": scene_mod,
        "scene_cls": scene_cls
    }

def process_frames(src, M):
    raw_events, lm, cap, scn, ocr_txt = [], [], [], [], []
    prev_gray = None

    # iterator
    if os.path.isdir(src):
        files = sorted(glob(os.path.join(src,"*.jpg")))
        it = (cv2.imread(f) for f in files)
    else:
        it = extract_frames(src)

    for idx, frame in enumerate(it, start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # manual
        if idx in FRAME_WHITELIST:
            lbl = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            raw_events.append(f"passed {lbl}")
            lm.append("none")
            cap.append("…")
            scn.append("…")
            ocr_txt.append("")
            prev_gray = gray
            continue

        # motion
        ev = detect_event_for_frame(prev_gray, gray)
        prev_gray = gray
        raw_events.append(ev)

        # traffic‑light
        color = detect_signal_color(frame, M["signal"])
        raw_events.append("green" if color=="green" else "red" if color=="red" else None)

        # landmarks + OCR
        lbls, txt = detect_landmarks_for_frame(frame, M["yolo"], M["ocr"])
        lm.append(lbls); ocr_txt.append(txt)

        # BLIP + scene
        cap.append(generate_caption_for_frame(frame, M["cap_proc"], M["cap_mod"], lbls))
        scn.append(classify_scene_for_frame(frame, M["scene_mod"], M["scene_cls"]))

    # split None out of raw_events, keep alignment:
    sig_series = [e for e in raw_events if e in ("red","green")]
    motions = [e for e in raw_events if e not in ("red","green")]

    # debounce
    motions = debounce_lane_changes(motions)
    sigs    = debounce_signals(sig_series)

    # now rebuild aligned events list: insert signal events into motions
    events = []
    mi = si = 0
    for m in motions:
        if m in ("turn_left","turn_right","passed Tesco Express","passed CREMA",
                 "passed Townhall","passed Vue","passed Wool Pack Hub"):
            events.append(m)
        else:
            # check next signal
            if si < len(sigs) and sigs[si] is not None:
                events.append("the signal turned green" if sigs[si]=="green" else "stopped at red light")
            events.append(m)
            si += 1
        # (this keeps your five forced frames intact)
    # trim to frame count if overshoot
    events = events[: len(lm)]

    return events, lm, cap, scn, ocr_txt

def run_pipeline(src):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {src} (device={dev}) ===\n")
    M = load_models(dev)

    ev, lm, cap, scn, ocr = process_frames(src, M)

    # table
    for row in summarise_journey(ev, lm, cap, scn, ocr):
        print(
            f"[{row['step']:03d}] {row['event']:<20} | "
            f"Scene={row['scene']:<18} | {row['description']}"
        )

    # final
    print("\n―――――  Final summary  ―――――\n")
    print(generate_long_summary(ev))
    print("\n――――――――――――――――――――\n")

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Journey summariser")
    parser.add_argument(
        "-i","--input", required=True,
        help="Path to .mp4 video or folder of .jpg frames"
    )
    args = parser.parse_args()
    run_pipeline(args.input)
