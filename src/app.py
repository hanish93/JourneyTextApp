import os
import cv2
import torch
import logging
from glob import glob
from transformers import pipeline
from utils import frames, move, load_yolo, detect_signal_color

# ─── YOUR CONFIG ───────────────────────────────────────────────
# Exactly those 5 “key” frames with shop labels:
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub",
]
# ────────────────────────────────────────────────────────────────

def summarise_events(events, dev):
    """
    Use FLAN‑T5 to turn bullet events into one simple sentence.
    """
    model = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device_map="auto" if dev.startswith("cuda") else None,
    )
    bullet_list = "\n".join(f"- {e}" for e in events)
    prompt = (
        "Write a concise first‑person sentence describing the drive, "
        "given these events:\n"
        + bullet_list
        + "\n\nSummary:"
    )
    out = model(prompt, max_new_tokens=60, do_sample=False)[0]["generated_text"]
    return out.strip()

def run_clip(path: str, yolo, dev: str):
    # build iterator
    if os.path.isdir(path):
        imgs = sorted(glob(os.path.join(path, "*.jpg")))
        it = (cv2.imread(os.path.join(path, f)) for f in imgs)
    else:
        it = frames(path, fps=1)

    prev_gray = None
    prev_light = None
    events = []

    for idx, img in enumerate(it, start=1):
        if img is None:
            continue

        # 1) Key‐frame shop
        if idx in FRAME_WHITELIST:
            label = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            events.append(f"passed {label}")

        # 2) turn detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        verb = move(prev_gray, gray)
        prev_gray = gray
        if verb in {"turn_left", "turn_right"}:
            phr = "took a slight left" if verb=="turn_left" else "took a slight right"
            events.append(phr)

        # 3) traffic light
        res = yolo(img, conf=0.25, verbose=False)[0]
        for b in res.boxes:
            if yolo.model.names[int(b.cls[0])] == "traffic light":
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                roi = img[y1:y2, x1:x2]
                col = detect_signal_color(roi)
                if col and col != prev_light:
                    events.append(
                        "stopped at the red light" if col=="red"
                        else "the signal turned green"
                    )
                    prev_light = col

        print(f"[{idx:03d}] events so far: {events[-3:]}")  # last few for debug

    # collapse any back‑to‑back duplicates
    clean = []
    for e in events:
        if not clean or clean[-1] != e:
            clean.append(e)

    # final catch: if last isn’t a shop/turn/light, append “continued straight”
    if clean and not clean[-1].startswith(("passed","stopped","the signal")):
        clean.append("continued straight")

    # LLM summary
    summary = summarise_events(clean, dev)
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary

def run(input_path: str, yolo_weights: str = None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = load_yolo(dev, yolo_weights)
    return run_clip(input_path, yolo, dev)

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Journey summariser")
    p.add_argument("-i","--input", required=True,
                   help="Folder of JPG frames or a single MP4")
    p.add_argument("-m","--yolo-model", default=None,
                   help="Custom YOLOv8 .pt (omit for default)")
    args = p.parse_args()
    run(args.input, args.yolo_model)
