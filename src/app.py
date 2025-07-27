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
    summarise_journey,
    salient, kind_of
)

# ─── YOUR 5 MANUAL FRAMES ─────────────────────────────────────────────
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub",
]
# ─────────────────────────────────────────────────────────────────────

def load_models(device):
    print("[Models] Loading…")
    (obj, sign), ocr = get_landmark_models(device)
    cap_proc, cap_mod = get_caption_models(device)
    scene_mod, scene_classes = get_scene_model(device)
    print("[Models] Done.")
    return {
        "obj": obj, "sign": sign, "ocr": ocr,
        "cap_proc": cap_proc, "cap_mod": cap_mod,
        "scene_mod": scene_mod, "scene_classes": scene_classes
    }

def process_frames(source, M):
    events, lm, cap, scn, ocr_txt = [], [], [], [], []
    stats = {"shop":{}, "building":{}, "other":{}}
    prev_gray = None

    # pick the right iterator
    if os.path.isdir(source):
        files = sorted(glob(os.path.join(source,"*.jpg")))
        iterator = (cv2.imread(fp) for fp in files)
    else:
        iterator = extract_frames(source)

    for idx, frame in enumerate(iterator, start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) motion
        ev = detect_event_for_frame(prev_gray, gray)
        prev_gray = gray
        events.append(ev)

        # 1a) manual shop frame
        if idx in FRAME_WHITELIST:
            lbl = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            events.append(f"passed {lbl}")
            # placeholders to keep lists aligned
            lm.append("none")
            cap.append("…")
            scn.append("…")
            ocr_txt.append("")
            continue

        # 2) landmarks + OCR
        with torch.no_grad():
            lbls, txt = detect_landmarks_for_frame(
                frame, M["obj"], M["ocr"]
            )
            lm.append(lbls)
            ocr_txt.append(txt)

            # 3) BLIP caption + scene
            cap.append(
                generate_caption_for_frame(frame, M["cap_proc"], M["cap_mod"], lbls)
            )
            scn.append(
                classify_scene_for_frame(frame, M["scene_mod"], M["scene_classes"])
            )

            # 4) sign stats
            if M["sign"]:
                for b in M["sign"](frame, conf=0.15, verbose=False)[0].boxes:
                    x1,y1,x2,y2 = map(int,b.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    t = " ".join(s[1] for s in M["ocr"].readtext(crop, detail=0))
                    if not salient(t):
                        continue
                    k = kind_of(t)
                    e = stats[k].setdefault(t,[0,0.0])
                    e[0] += 1
                    e[1] += float(b.conf[0])

        print(f"[{idx:03d}] last_event={events[-1]}")

    torch.cuda.empty_cache()
    return events, lm, cap, scn, ocr_txt, stats

def run_pipeline(source):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {source} (device={dev}) ===\n")
    M = load_models(dev)

    ev, lm, cap, scn, ocr_txt, stats = process_frames(source, M)

    # per‑step table
    for row in summarise_journey(ev, lm, cap, scn, ocr_txt):
        print(
            f"[{row['step']:03d}] {row['event']:<15} | "
            f"Scene={row['scene']:<18} | {row['description']}"
        )

    print("\n―――――  Long‑form summary  ―――――\n")
    print(generate_long_summary(ev, lm, cap, scn, ocr_txt, stats))
    print("\n―――――――――――――――――――――――――――――\n")

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore",category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Console journey summariser")
    p.add_argument(
        "--video", required=True,
        help="Path to .mp4 or folder of frames"
    )
    args = p.parse_args()
    run_pipeline(args.video)
