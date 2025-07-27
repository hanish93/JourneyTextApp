# src/app.py  —  glue + progress + clear summary

import os
import cv2
import torch
import logging
from glob import glob

from utils import frames, move, load_det, landmarks, load_cap, cap_img, diary, DYNAMIC


def run_clip(path, models, dev):
    # ─── Find frames ────────────────────────────────────────────
    if os.path.isdir(path):
        # 1) try images
        img_paths = sorted(glob(os.path.join(path, "*.jpg"))) \
                  + sorted(glob(os.path.join(path, "*.jpeg"))) \
                  + sorted(glob(os.path.join(path, "*.png")))
        if img_paths:
            frame_iter = (cv2.imread(p) for p in img_paths)
        else:
            # 2) try videos
            vid_paths = sorted(glob(os.path.join(path, "*.mp4")))
            if vid_paths:
                # recurse into each video
                return "\n\n".join(run_clip(v, models, dev) for v in vid_paths)
            else:
                raise FileNotFoundError(f"No .jpg/.png or .mp4 under '{path}'")
    else:
        # single video
        frame_iter = frames(path, fps=1)

    yolo, ocr = models["det"]
    cap_pipe = models["cap"]
    prev = None
    sentences = []
    whitelist = set()

    # ─── Process each frame ────────────────────────────────────
    for idx, f in enumerate(frame_iter, start=1):
        if f is None:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        verb = move(prev, gray)
        prev = gray

        names = landmarks(f, yolo, ocr)
        whitelist.update(names)

        caption = cap_img(f, cap_pipe, " ".join(names) if names else "")
        caption = " ".join(w for w in caption.split() if w.lower() not in DYNAMIC)
        sentences.append(f"I {verb} and {caption.lower()}")

        # ← progress print
        print(f"[frame {idx:03d}] saw signs: {', '.join(names) or '—'}")

    # ─── Final summary ──────────────────────────────────────────
    summary = diary(sentences, whitelist)
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary


def run(target, custom_yolo=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    yolo, ocr = load_det(dev, custom_yolo)
    cap = load_cap(dev)
    models = {"det": (yolo, ocr), "cap": cap}

    return run_clip(target, models, dev)


if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Console journey summariser")
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to .mp4, folder of .mp4s, or folder of JPG/PNG frames"
    )
    parser.add_argument(
        "--yolo-model", "-m", default=None,
        help="Optional path to custom YOLOv8 .pt weights"
    )
    args = parser.parse_args()
    run(args.input, args.yolo_model)
