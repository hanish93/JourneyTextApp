# src/app.py — glue everything together

import os
import cv2
import torch
import logging
from glob import glob

from utils import frames, move, load_det, landmarks, load_cap, cap_img, diary, DYNAMIC


def run_clip(path, models, dev):
    # If directory of images
    if os.path.isdir(path) and any(path.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
        imgs = sorted(glob(os.path.join(path, "*.[pj][pn]g")))
        frame_iter = (cv2.imread(im) for im in imgs)
    # If directory of videos
    elif os.path.isdir(path):
        return "\n\n".join(run_clip(v, models, dev) for v in sorted(glob(os.path.join(path, "*.mp4"))))
    else:
        frame_iter = frames(path, fps=1)

    yolo, ocr = models["det"]
    cap_pipe = models["cap"]
    prev = None
    sentences = []
    whitelist = set()

    for f in frame_iter:
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

    return diary(sentences, whitelist)


def run(target, custom_yolo=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    yolo, ocr = load_det(dev, custom_yolo)
    cap = load_cap(dev)
    models = {"det": (yolo, ocr), "cap": cap}

    print(run_clip(target, models, dev))


if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Console journey summariser")
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a .mp4 file, folder of .mp4s, or folder of JPG/PNG frames"
    )
    parser.add_argument(
        "--yolo-model", "-m", default=None,
        help="Optional path to custom YOLOv8 .pt weights"
    )
    args = parser.parse_args()
    run(args.input, args.yolo_model)
