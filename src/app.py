# app.py — glue everything together
import os
import cv2
import torch
import pathlib
import logging
from glob import glob

from utils import frames, move, load_det, landmarks, load_cap, cap_img, diary, DYNAMIC

def run_clip(path, models, dev):
    """
    If 'path' is a folder of images, we read them in sorted order.
    Otherwise we treat it as video and sample at 1fps.
    """
    # choose frame iterator
    if os.path.isdir(path) and any(path.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
        imgs = sorted(glob(os.path.join(path, "*.[pj][pn]g")))
        frame_iter = (cv2.imread(im) for im in imgs)
    elif os.path.isdir(path):
        vids = sorted(glob(os.path.join(path, "*.mp4")))
        # for directories, just recurse into each video
        summaries = []
        for v in vids:
            summaries.append(run_clip(v, models, dev))
        return "\n\n".join(summaries)
    else:
        frame_iter = frames(path, fps=1)

    prev = None
    sentences = []
    whitelist = set()
    yolo, ocr = models["det"]
    cap_pipe = models["cap"]

    for f in frame_iter:
        if f is None:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        verb = move(prev, gray)
        prev = gray

        names = landmarks(f, yolo, ocr)
        whitelist.update(names)

        caption = cap_img(f, cap_pipe, " ".join(names) if names else "")
        # strip out any dynamic nouns
        caption = " ".join(w for w in caption.split() if w.lower() not in DYNAMIC)
        sentences.append(f"I {verb} and {caption.lower()}")

    return diary(sentences, whitelist)


def run(target, custom_yolo=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    yolo, ocr = load_det(dev, custom_yolo)
    cap = load_cap(dev)
    models = {"det": (yolo, ocr), "cap": cap}

    p = pathlib.Path(target)
    if p.is_dir():
        # process folder (of images OR videos)
        print(run_clip(str(p), models, dev))
    else:
        # single .mp4
        print(run_clip(str(p), models, dev))


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
        help="Optional path to your custom YOLOv8 weights (.pt) with extra classes"
    )
    args = parser.parse_args()
    run(args.input, args.yolo_model)
