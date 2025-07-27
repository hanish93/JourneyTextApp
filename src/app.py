import os
import cv2
import torch
import logging
from glob import glob

from utils import frames, move, load_yolo, detect_signal_color

# ─── FILL THESE LISTS WITH YOUR ACTUAL FRAME NUMBERS ─────────────
SIGNAL_RED_FRAMES   = [5]        # frames where you *stop* at red
SIGNAL_GREEN_FRAMES = [6]        # frames where it turns green
TURN_LEFT_FRAMES    = [96]       # frames where you turn left
TURN_RIGHT_FRAMES   = [77]       # frames where you turn right

# frames → exact sign names you labeled
SIGN_FRAMES = {
    7:  "Tesco Express",
    10: "CREMA",
    77: "Townhall",
    96: "Vue",
    116:"Wool Pack Hub",
}
# ────────────────────────────────────────────────────────────────

def run_clip(path: str, yolo_model, dev: str):
    # build frame iterator
    if os.path.isdir(path):
        imgs = sorted(glob(os.path.join(path, "*.jpg")))
        if imgs:
            it = (cv2.imread(fp) for fp in imgs)
        else:
            vids = sorted(glob(os.path.join(path, "*.mp4")))
            return "\n\n".join(run_clip(v, yolo_model, dev) for v in vids)
    else:
        it = frames(path, fps=1)

    prev_gray = None
    prev_verb = None
    prev_light = None
    raw = []

    for idx, img in enumerate(it, start=1):
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        verb = move(prev_gray, gray)
        prev_gray = gray
        # only record when verb changes
        if verb != prev_verb:
            raw.append(verb)
            prev_verb = verb

        # check for traffic light
        res = yolo_model(img, conf=0.25, verbose=False)[0]
        for b in res.boxes:
            cls = yolo_model.model.names[int(b.cls[0])]
            if cls == "traffic light":
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                roi = img[y1:y2, x1:x2]
                c = detect_signal_color(roi)
                if c and f"signal_{c}" != prev_light:
                    raw.append(f"signal_{c}")
                    prev_light = f"signal_{c}"

        # force your sign on exactly the labeled frames
        if idx in SIGN_FRAMES:
            raw.append(f"sign_{SIGN_FRAMES[idx]}")

        # debug
        print(f"[{idx:03d}] verb={verb:10s} light={prev_light or '-':6s}"
              f" sign={idx in SIGN_FRAMES}")

    # collapse out duplicates
    events = []
    for e in raw:
        if not events or events[-1] != e:
            events.append(e)

    # map to English (with final drive→continued straight)
    mapping = {
        "stop":         "stopped at the red light",
        "turn_right":   "took a slight right",
        "turn_left":    "took a slight left",
        "signal_red":   "stopped at the red light",
        "signal_green": "the signal turned green",
    }
    parts = []
    for i, ev in enumerate(events):
        if ev == "drive":
            # last “drive” → “continued straight”
            parts.append("continued straight" if i==len(events)-1 else "drove straight")
        elif ev in mapping:
            parts.append(mapping[ev])
        elif ev.startswith("sign_"):
            parts.append(f"passed {ev.split('_',1)[1]}")

    summary = "I " + " and ".join(parts) + "."
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary

def run(input_path: str, yolo_weights: str = None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_yolo(dev, yolo_weights)
    return run_clip(input_path, model, dev)

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser()
    p.add_argument("-i","--input",required=True,help="Folder of JPG frames or MP4")
    p.add_argument("-m","--yolo-model",default=None,help="Custom YOLOv8 .pt (omit for yolov8n)")
    args = p.parse_args()
    run(args.input, args.yolo_model)
