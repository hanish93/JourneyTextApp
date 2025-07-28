import cv2
import numpy as np
from ultralytics import YOLO

def load_signal_model(device="cpu", weights="yolov8n.pt"):
    """
    Load a YOLOv8 model fine‑tuned (or the default) for traffic‑light detection.
    """
    model = YOLO(weights).to(device).half()
    return model

def detect_signal_color(frame, model, conf=0.1):
    """
    Returns 'red', 'green', or None by detecting the largest traffic‐light box.
    """
    # 1) run YOLO
    res = model(frame, conf=conf, verbose=False)[0]
    tl_boxes = []
    for b in res.boxes:
        cls = model.model.names[int(b.cls[0])]
        if cls == "traffic light":
            x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
            tl_boxes.append((x1,y1,x2,y2))

    # 2) crop region
    if tl_boxes:
        x1,y1,x2,y2 = max(tl_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        crop = frame[y1:y2, x1:x2]
    else:
        # fallback to top‑center if no box
        h,w = frame.shape[:2]
        crop = frame[0:int(0.2*h), int(0.3*w):int(0.7*w)]

    if crop.size == 0:
        return None

    # 3) HSV thresholding
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0,80,80),   (10,255,255))
    red2 = cv2.inRange(hsv, (160,80,80), (180,255,255))
    red  = cv2.bitwise_or(red1, red2)
    green= cv2.inRange(hsv, (40,80,80),  (85,255,255))

    rc = int(cv2.countNonZero(red))
    gc = int(cv2.countNonZero(green))
    if max(rc, gc) < 100:
        return None
    return "red" if rc>gc else "green"


def generate_full_summary(passes, signals, labels, frame_count):
    """
    Build a single English sentence from:
      - passes: sorted list of frame indices (ints) where you 'passed X'
      - signals: list[str] of length frame_count, each 'red'/'green'/None
      - labels: dict {frame_idx: "Landmark Name"}
    """
    # 1) group any two passes <5 frames apart
    groups = []
    for f in passes:
        if not groups or f - groups[-1][-1] > 5:
            groups.append([f])
        else:
            groups[-1].append(f)

    parts = []
    cur = 0

    for grp in groups:
        start = grp[0]
        end   = grp[-1]

        # stop at red if any red in [cur, start)
        if any(signals[i]=="red" for i in range(cur, start)):
            parts.append("stopped at the red light")

        # emit the landmarks in this group
        names = [labels[f] for f in grp]
        parts.append("then passed " + " and ".join(names))

        cur = end + 1

    # tail: another red?
    if any(signals[i]=="red" for i in range(cur, frame_count)):
        parts.append("stopped at the red light once more")

    parts.append("continued straight")

    # stitch into one sentence
    sent = parts[0].capitalize()
    for p in parts[1:]:
        sent += ", " + p
    return sent + "."
