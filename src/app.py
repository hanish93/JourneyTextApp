import os
import cv2
import torch
import logging
from glob import glob

from utils import (
    extract_frames,
    detect_event_for_frame,
    get_landmark_models,
    detect_landmarks_for_frame,
    get_caption_models,
    generate_caption_for_frame,
    get_scene_model,
    classify_scene_for_frame,
    generate_long_summary,
    summarise_journey
)

# ─── YOUR 5 MANUAL FRAMES ────────────────────────────────────────────────
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub",
]
# ────────────────────────────────────────────────────────────────────────

def load_models(device):
    print("[Models] Loading…")
    (obj, sign), ocr = get_landmark_models(device)
    cap_proc, cap_mod = get_caption_models(device)
    scene_mod, scene_cls = get_scene_model(device)
    print("[Models] Done.")
    return {
        "obj": obj, "sign": sign, "ocr": ocr,
        "cap_proc": cap_proc, "cap_mod": cap_mod,
        "scene_mod": scene_mod, "scene_cls": scene_cls
    }

def debounce_events(events):
    """
    Suppress any single-frame turns. Must repeat to stay.
    """
    out = list(events)
    for i in range(1, len(events)-1):
        if events[i] in ("turn_left","turn_right"):
            if events[i-1] != events[i] and events[i+1] != events[i]:
                out[i] = "drive"
    return out

def process_frames(src, M):
    ev, lm, cap, scn, ocr_txt = [], [], [], [], []
    stats = {"shop":{}, "building":{}, "other":{}}
    prev_gray = None

    if os.path.isdir(src):
        files = sorted(glob(os.path.join(src,"*.jpg")))
        it = (cv2.imread(f) for f in files)
    else:
        it = extract_frames(src)

    for idx, frm in enumerate(it, start=1):
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        # Manual injection
        if idx in FRAME_WHITELIST:
            lbl = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            ev.append(f"passed {lbl}")
            lm.append("none"); cap.append("…"); scn.append("…"); ocr_txt.append("")
            prev_gray = gray
            continue

        # Motion
        evt = detect_event_for_frame(prev_gray, gray)
        prev_gray = gray
        ev.append(evt)

        # Landmarks + OCR
        with torch.no_grad():
            lmk, txt = detect_landmarks_for_frame(frm, M["obj"], M["ocr"])
            lm.append(lmk)
            ocr_txt.append(txt)

            # BLIP caption + scene
            cap.append(
                generate_caption_for_frame(frm, M["cap_proc"], M["cap_mod"], lmk)
            )
            scn.append(
                classify_scene_for_frame(frm, M["scene_mod"], M["scene_cls"])
            )

            # sign statistics (unchanged)
            if M["sign"]:
                for b in M["sign"](frm, conf=0.15, verbose=False)[0].boxes:
                    x1,y1,x2,y2 = map(int,b.xyxy[0])
                    crop = frm[y1:y2, x1:x2]
                    t = " ".join(s[1] for s in M["ocr"].readtext(crop, detail=0))
                    # accumulate if needed…

    torch.cuda.empty_cache()
    return ev, lm, cap, scn, ocr_txt, stats

def run_pipeline(src):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {src} (device={dev}) ===\n")
    M = load_models(dev)

    ev, lm, cap, scn, ocr, stats = process_frames(src, M)
    ev = debounce_events(ev)

    # Table
    for row in summarise_journey(ev, lm, cap, scn, ocr):
        print(
            f"[{row['step']:03d}] {row['event']:<15} | "
            f"Scene={row['scene']:<18} | {row['description']}"
        )

    # Final
    print("\n―――――  Summary  ―――――\n")
    print(generate_long_summary(ev, lm, cap, scn, ocr, stats))
    print("\n――――――――――――――――――――\n")

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Console journey summariser")
    p.add_argument("--video", "-i", required=True,
                   help="Path to .mp4 or folder of .jpg frames")
    args = p.parse_args()
    run_pipeline(args.video)
