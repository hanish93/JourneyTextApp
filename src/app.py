import os
import cv2
import torch
import logging
from glob import glob

from utils import load_det, detect_signal_color, landmarks
from utils import signage_names  # if you’d rather use full-frame OCR direct
from utils import fetch      # just so models dir exists

def move(prev_gray, curr_gray, dx=1.5, stop_thr=0.2):
    """
    Exactly your optical‑flow verb as before.
    """
    if prev_gray is None:
        return "drive"
    f = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                     .5,3,15,3,5,1.2,0)
    dxm = f[...,0].mean()
    mag = np.linalg.norm(f,axis=2).mean()
    if mag < stop_thr:
        return "stop"
    if dxm > dx:
        return "turn_right"
    if dxm < -dx:
        return "turn_left"
    return "drive"

def run_clip(path, models, dev):
    # ─── build frame iterator ───────────────────────────────
    if os.path.isdir(path):
        # first try images
        img_files = sorted(glob(os.path.join(path, "*.jpg")))
        if img_files:
            frames = (cv2.imread(f) for f in img_files)
        else:
            # fallback: videos
            vids = sorted(glob(os.path.join(path, "*.mp4")))
            if not vids:
                raise FileNotFoundError(f"No .jpg/.png or .mp4 under {path}")
            # concatenate all video summaries
            return "\n\n".join(run_clip(v, models, dev) for v in vids)
    else:
        # single mp4
        from utils import frames as video_frames
        frames = video_frames(path, fps=1)

    yolo, ocr = models["det"]
    prev_gray = None

    seen_signs = set()
    prev_signal = None
    timeline = []

    for idx, img in enumerate(frames, start=1):
        if img is None:
            continue

        # motion verb
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        verb = move(prev_gray, gray)
        prev_gray = gray

        # light color
        sig = detect_signal_color(img, yolo)
        if sig and sig != prev_signal:
            timeline.append(f"signal_{sig}")
        prev_signal = sig

        # signs
        names = landmarks(img, yolo, ocr)
        for nm in names:
            if nm not in seen_signs:
                seen_signs.add(nm)
                timeline.append(f"sign_{nm}")

        # road motion
        timeline.append(verb)

        # progress
        print(f"[frame {idx:03d}] verb={verb:10s} light={sig or '-':5s} signs={names or ['-']}")

    # ─── map timeline to English phrases ─────────────────────
    phrases = []
    for ev in timeline:
        if ev == "signal_green":
            phrases.append("the signal turned green")
        elif ev == "signal_red":
            phrases.append("stopped at the red signal")
        elif ev == "drive":
            phrases.append("drove straight")
        elif ev == "turn_right":
            phrases.append("took a slight right")
        elif ev == "turn_left":
            phrases.append("took a slight left")
        elif ev == "stop":
            phrases.append("came to a stop")
        elif ev.startswith("sign_"):
            name = ev.split("_",1)[1]
            phrases.append(f"passed {name}")
        # else ignore

    # ─── build the final single‐sentence summary ────────────
    if phrases:
        summary = "I " + " and ".join(phrases) + "."
    else:
        summary = "No events detected."

    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary

def run(target, custom_yolo=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    yolo, ocr = load_det(dev, custom_yolo)
    models = {"det": (yolo, ocr)}
    return run_clip(target, models, dev)

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True,
                        help="folder of JPGs or a single .mp4")
    parser.add_argument("--yolo-model", "-m", default=None,
                        help="optional custom YOLO weights")
    args = parser.parse_args()
    run(args.input, args.yolo_model)
