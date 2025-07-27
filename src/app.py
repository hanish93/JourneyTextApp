# src/app.py

import os, cv2, torch, logging
from glob import glob
from transformers import pipeline
from utils import frames, load_custom_yolo, detect_signal_color

# ─── WHICH FRAMES ARE “SHOP” FRAMES ─────────────────────────────
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub",
]
# ────────────────────────────────────────────────────────────────

def summarise_events(events, dev):
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-xl",
        device_map="auto" if dev.startswith("cuda") else None,
    )
    bullets = "\n".join(f"- {e}" for e in events)
    prompt = (
        "Write one concise first‑person sentence about this drive, given these events:\n"
        + bullets + "\n\nSummary:"
    )
    out = pipe(prompt, max_new_tokens=60, do_sample=False)[0]["generated_text"]
    return out.strip()

def run_clip(path: str, yolo, dev: str):
    # build frame iterator
    if os.path.isdir(path):
        jpgs = sorted(glob(os.path.join(path,"*.jpg")))
        it = (cv2.imread(fp) for fp in jpgs)
    else:
        it = frames(path, fps=1)

    prev_light = None
    events = []

    for idx, img in enumerate(it, start=1):
        if img is None: continue

        # 1) force your shop labels
        if idx in FRAME_WHITELIST:
            label = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            events.append(f"passed {label}")

        # 2) YOLO detection of custom classes
        res = yolo(img, conf=0.25, verbose=False)[0]
        seen = set()
        for box in res.boxes:
            cls = yolo.model.names[int(box.cls[0])]
            if cls == "traffic light":
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                roi = img[y1:y2, x1:x2]
                c = detect_signal_color(roi)
                if c and c not in seen:
                    seen.add(c)
                    if c=="red":
                        events.append("stopped at the red light")
                    else:
                        events.append("the signal turned green")
                    prev_light = c

            elif cls == "turn_left" and "turn_left" not in seen:
                events.append("took a slight left")
                seen.add("turn_left")

            elif cls == "turn_right" and "turn_right" not in seen:
                events.append("took a slight right")
                seen.add("turn_right")

        print(f"[{idx:03d}] events so far: {events[-3:]}")

    # collapse consecutive duplicates
    clean = []
    for e in events:
        if not clean or clean[-1]!=e:
            clean.append(e)

    # ensure ending with “continued straight”
    if not clean[-1].startswith(("passed","stopped","the signal","took")):
        clean.append("continued straight")

    summary = summarise_events(clean, dev)
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary

def run(input_path: str, yolo_weights: str):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = load_custom_yolo(dev, yolo_weights)
    return run_clip(input_path, yolo, dev)

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore",category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Journey summariser")
    p.add_argument("-i","--input", required=True,
                   help="Folder of JPGs or MP4")
    p.add_argument("-m","--yolo-model", default="runs/train/journey-model/weights/best.pt",
                   help="Path to your custom YOLO .pt")
    args = p.parse_args()
    run(args.input, args.yolo_model)
