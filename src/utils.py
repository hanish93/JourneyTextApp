import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# 1) FRAME EXTRACTION: yields every 1-fps frame from a video or a folder of .jpg
def extract_frames(src, fps=1):
    if os.path.isdir(src):
        for fn in sorted(os.listdir(src)):
            if fn.lower().endswith(".jpg"):
                img = cv2.imread(os.path.join(src, fn))
                if img is not None:
                    yield img
        return
    cap = cv2.VideoCapture(src)
    nat = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(nat / fps))
    idx, ok, frame = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield frame
        ok, frame = cap.read()
        idx += 1
    cap.release()

# ─────────────────────────────────────────────────────────────────────────────
# 2) OPTICAL-FLOW EVENT DETECTION: turn_left, turn_right, stop, or drive
def detect_event(prev_gray, cur_gray, dx_thresh=1.5, stop_thresh=0.2):
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        0.5,3,15,3,5,1.2,0)
    dx  = float(flow[...,0].mean())
    mag = float(np.linalg.norm(flow,axis=2).mean())
    if mag < stop_thresh:
        return "stop"
    if dx > dx_thresh:
        return "turn_right"
    if dx < -dx_thresh:
        return "turn_left"
    return "drive"

# ─────────────────────────────────────────────────────────────────────────────
# 3) TRAFFIC-LIGHT DETECTION + COLOUR via YOLOv8 + HSV
_signal_model = None
def load_signal_model(device="cpu", weights="yolov8n.pt"):
    global _signal_model
    if _signal_model is None:
        _signal_model = YOLO(weights).to(device).half()
    return _signal_model

def detect_signal_color(frame, model, conf=0.15):
    res = model(frame, conf=conf, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        cls = model.model.names[int(b.cls[0])]
        if cls == "traffic light":
            x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
            boxes.append((x1,y1,x2,y2))
    if boxes:
        x1,y1,x2,y2 = max(boxes, key=lambda bb: (bb[2]-bb[0])*(bb[3]-bb[1]))
        crop = frame[y1:y2, x1:x2]
    else:
        h,w = frame.shape[:2]
        crop = frame[0:int(0.2*h), int(0.3*w):int(0.7*w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0,80,80), (10,255,255))
    red2 = cv2.inRange(hsv, (160,80,80), (180,255,255))
    red  = cv2.bitwise_or(red1, red2)
    green= cv2.inRange(hsv, (40,80,80), (85,255,255))
    rc, gc = int(cv2.countNonZero(red)), int(cv2.countNonZero(green))
    if max(rc,gc) < 100:
        return None
    return "red" if rc>gc else "green"

# ─────────────────────────────────────────────────────────────────────────────
# 4) DEBOUNCE: require at least N frames of a turn or signal in a window
def debounce_events(evts, window=3, min_count=3):
    out = list(evts)
    n = len(evts)
    for i,e in enumerate(evts):
        if e in ("turn_left","turn_right"):
            cnt = sum(1 for j in range(max(0,i-window), min(n,i+window+1))
                      if evts[j]==e)
            if cnt < min_count:
                out[i] = "drive"
    return out

def debounce_signals(sigs, window=3, min_count=2):
    out = [None]*len(sigs)
    n = len(sigs)
    for i,s in enumerate(sigs):
        if s in ("red","green"):
            cnt = sum(1 for j in range(max(0,i-window), min(n,i+window+1))
                      if sigs[j]==s)
            if cnt >= min_count:
                out[i] = s
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 5) SUMMARY GENERATOR: stitch together stops, greens, turns, drives
def generate_summary(events, signals):
    parts, last_sig = [], None

    for i,(e,s) in enumerate(zip(events, signals), start=1):
        # start with a stop if frame 1 is red
        if i==1 and s=="red":
            parts.append("stopped at the red light")
            last_sig="red"
            continue
        # after red→green
        if last_sig=="red" and s=="green":
            parts.append("once green, drove on")
            last_sig="green"
        # record a stop
        if s=="red" and last_sig!="red":
            parts.append("then stopped at the red light")
            last_sig="red"
        # landmark? none, so skip
        # turns
        if e=="turn_left":
            parts.append("turned left")
        elif e=="turn_right":
            parts.append("took a slight right")
        # plain drive segments we collapse later

    # ensure we begin by driving straight if no initial stop
    if not parts or not parts[0].startswith("stopped"):
        parts.insert(0,"drove straight")

    # collapse consecutive duplicates
    clean=[parts[0]]
    for p in parts[1:]:
        if p!=clean[-1]:
            clean.append(p)

    # form into one sentence
    sent = clean[0].capitalize()
    for p in clean[1:]:
        sent += " and " + p
    # always finish with continued straight
    if not sent.endswith("straight"):
        sent += " and continued straight."
    else:
        sent += "."

    return sent
