import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# 1) FRAME WHITELIST & LABELS (for your five manually labelled key frames)
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub"
]

# ─────────────────────────────────────────────────────────────────────────────
# 2) FRAME EXTRACTION (1 fps) from video or folder of .jpg
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
# 3) MOTION CLASSIFICATION via optical flow
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
# 4) SIGNAL COLOR via YOLOv8 + simple HSV thresholding
_yolo_sig = None
def load_signal_model(device="cpu"):
    global _yolo_sig
    if _yolo_sig is None:
        _yolo_sig = YOLO("yolov8n.pt").to(device).half()
    return _yolo_sig

def detect_signal_color(frame, yolo, conf=0.15):
    r = yolo(frame, conf=conf, verbose=False)[0]
    tbs = []
    for b in r.boxes:
        cls = yolo.model.names[int(b.cls[0])]
        if cls == "traffic light":
            x1,y1,x2,y2 = map(int,b.xyxy[0].cpu().numpy())
            tbs.append((x1,y1,x2,y2))
    if tbs:
        x1,y1,x2,y2 = max(tbs, key=lambda bb:(bb[2]-bb[0])*(bb[3]-bb[1]))
        crop = frame[y1:y2, x1:x2]
    else:
        h,w = frame.shape[:2]
        crop = frame[0:int(0.2*h), int(0.3*w):int(0.7*w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv,(0,80,80),(10,255,255))
    red2 = cv2.inRange(hsv,(160,80,80),(180,255,255))
    red  = cv2.bitwise_or(red1, red2)
    green= cv2.inRange(hsv,(40,80,80),(85,255,255))
    rc, gc = int(cv2.countNonZero(red)), int(cv2.countNonZero(green))
    if max(rc,gc) < 100:
        return None
    return "red" if rc>gc else "green"

# ─────────────────────────────────────────────────────────────────────────────
# 5) DEBOUNCE repeated turns/signals
def debounce_events(evts, window=3, min_count=3):
    """
    Only keep a 'turn_left' or 'turn_right' if it appears in >= min_count frames
    within a sliding window of +/- window frames around each index.
    Otherwise force it back to 'drive'.
    """
    out = evts.copy()
    n = len(evts)
    for i,e in enumerate(evts):
        if e in ("turn_left","turn_right"):
            cnt = sum(
                1
                for j in range(max(0, i-window), min(n, i+window+1))
                if evts[j] == e
            )
            if cnt < min_count:
                out[i] = "drive"
    return out

def debounce_signals(sigs, window=3):
    """
    Only keep a 'red' or 'green' if it appears in >=2 frames
    within a sliding window; fewer detections are dropped.
    """
    out = [None]*len(sigs)
    n = len(sigs)
    for i,s in enumerate(sigs):
        if s in ("red","green"):
            cnt = sum(
                1
                for j in range(max(0, i-window), min(n, i+window+1))
                if sigs[j] == s
            )
            if cnt >= 2:
                out[i] = s
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 6) FINAL SUMMARY BUILDER
def generate_summary(events, signals):
    parts, last_sig = [], None

    for idx, (e,s) in enumerate(zip(events, signals), start=1):
        # 1) initial red
        if idx == 1 and s == "red":
            parts.append("stopped at the red light")
            last_sig = "red"
            continue

        # 2) green after red
        if last_sig == "red" and s == "green":
            parts.append("once it turned green, I drove on")
            last_sig = "green"

        # 3) landmark passes
        if e.startswith("passed "):
            name = e.split(" ",1)[1]
            parts.append(f"passed {name}")

        # 4) turns (already debounced for >=3 frames)
        if e == "turn_left":
            parts.append("turned left")
        elif e == "turn_right":
            parts.append("took a slight right")

        # 5) new red (after something else)
        if s == "red" and last_sig != "red":
            parts.append("then stopped at the red light")
            last_sig = "red"

    # If nothing ever stopped us at start, mark that we began by driving straight
    if not parts or not parts[0].startswith("stopped"):
        parts.insert(0, "drove straight")

    # Dedupe consecutive duplicates
    clean = [parts[0]]
    for p in parts[1:]:
        if p != clean[-1]:
            clean.append(p)

    # Build the sentence
    sent = clean[0].capitalize()
    for p in clean[1:]:
        # ensure “then” only where it makes sense:
        if p.startswith("passed") or p.startswith("turned") or p.startswith("took"):
            sent += " and " + p
        else:
            sent += ", " + p

    # Always end by continuing straight, if final action wasn’t “stopped at…”
    if events and events[-1] in ("drive",) or any(ev.startswith("passed ") for ev in events[-1:]):
        sent += " and continued straight."
    else:
        sent = sent.rstrip('.') + "."

    return sent
