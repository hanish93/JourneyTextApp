# src/app.py

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
    salient,
    kind_of
)

# ─── MANUAL INJECTION CONFIG ─────────────────────────────────────────
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

def process_frames(video, M):
    ev, lm, cap, scn, ocr_txt = [], [], [], [], []
    sign_stats = {"shop": {}, "building": {}, "other": {}}
    prev = None

    for idx, frm in enumerate(extract_frames(video), start=1):
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        # 1) motion event
        ev.append(detect_event_for_frame(prev, gray))
        prev = gray

        # 1a) manual shop injection & skip YOLO/OCR for these frames
        if idx in FRAME_WHITELIST:
            label = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            ev.append(f"passed {label}")
            # append placeholders so lm, cap, scn, ocr_txt stay aligned
            lm.append("none")
            cap.append("…")
            scn.append("…")
            ocr_txt.append("")
            continue

        # 2) landmark + OCR
        with torch.no_grad():
            lmk, txt = detect_landmarks_for_frame(frm, M["obj"], M["ocr"])
            lm.append(lmk)
            ocr_txt.append(txt)

            # 3) BLIP caption + scene
            cap.append(
                generate_caption_for_frame(frm, M["cap_proc"], M["cap_mod"], lmk)
            )
            scn.append(
                classify_scene_for_frame(frm, M["scene_mod"], M["scene_classes"])
            )

            # 4) sign statistics (unchanged)
            if M["sign"]:
                for b in M["sign"](frm, conf=0.15, verbose=False)[0].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    s_txt = " ".join(
                        t[1] for t in M["ocr"].readtext(frm[y1:y2, x1:x2])
                    )
                    if not salient(s_txt):
                        continue
                    k = kind_of(s_txt)
                    e = sign_stats[k].setdefault(s_txt, [0, 0.0])
                    e[0] += 1
                    e[1] += float(b.conf[0])

        print(f"[{idx:03d}] last_event={ev[-1]}")

    torch.cuda.empty_cache()
    return ev, lm, cap, scn, ocr_txt, sign_stats

def run_pipeline(video):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {video} (device={device}) ===\n")
    M = load_models(device)

    ev, lm, cap, scn, ocr, stats = process_frames(video, M)

    # step‑by‑step table
    for row in summarise_journey(ev, lm, cap, scn, ocr):
        print(
            f"[{row['step']:03d}] {row['event']:<15} | "
            f"Scene={row['scene']:<18} | {row['description']}"
        )

    print("\n―――――  Long‑form summary  ―――――\n")
    print(generate_long_summary(ev, lm, cap, scn, ocr, stats))
    print("\n―――――――――――――――――――――――――――――\n")

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(
        description="Console journey summariser"
    )
    p.add_argument(
        "--video", required=True,
        help="Path to .mp4 file or folder of frames"
    )
    args = p.parse_args()
    run_pipeline(args.video)
