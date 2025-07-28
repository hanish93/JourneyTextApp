import os
import cv2
import numpy as np
from ultralytics import YOLO

# ─── FRAME EXTRACTION ────────────────────────────────────────────────────────
def extract_frames(path, fps=1):
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(".jpg"):
                img = cv2.imread(os.path.join(path, fn))
                if img is not None:
                    yield img
        return
    cap = cv2.VideoCapture(path)
    nat = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(nat / fps))
    idx, ok, frame = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield frame
        ok, frame = cap.read()
        idx += 1
    cap.release()

# ─── MOTION DETECTION ────────────────────────────────────────────────────────
def detect_event_for_frame(prev_gray, cur_gray, dx_thresh=2.5, stop_thresh=0.3):
    if prev_gray is None:
        return "drive"
    f = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                     0.5, 3, 15, 3, 5, 1.2, 0)
    dx  = f[...,0].mean()
    mag = np.linalg.norm(f, axis=2).mean()
    if mag < stop_thresh:
        return "stop"
    if dx > dx_thresh:
        return "turn_right"
    if dx < -dx_thresh:
        return "turn_left"
    return "drive"

def debounce_lane_changes(events, window=5):
    out = list(events)
    n   = len(events)
    for i, e in enumerate(events):
        if e in ("turn_left","turn_right"):
            cnt = sum(1 for j in range(max(0,i-window), min(n, i+window+1))
                      if events[j]==e)
            if cnt < 2:
                out[i] = "drive"
    return out

# ─── SIGNAL DETECTION ───────────────────────────────────────────────────────
def get_yolo_model(device):
    # fetch yolov8n automatically
    return YOLO(fetch_yolo(), task="detect").to(device).half()

def fetch_yolo():
    from ultralytics.yolo.utils import yaml_load
    # This will auto‐download yolov8n.pt if missing
    return "yolov8n.pt"

def detect_signal_color(frame, yolo, conf=0.2):
    r = yolo(frame, conf=conf, verbose=False)[0]
    cands = []
    for b in r.boxes:
        cls = yolo.model.names[int(b.cls[0])]
        if cls != "traffic light":
            continue
        x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
        w, h = x2-x1, y2-y1
        # allow boxes not too squat
        if h < w * 0.8:
            continue
        cands.append((x1,y1,x2,y2))
    if not cands:
        return None
    x1,y1,x2,y2 = max(cands, key=lambda bb:(bb[2]-bb[0])*(bb[3]-bb[1]))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    r1  = cv2.inRange(hsv, (0,60,60), (10,255,255))
    r2  = cv2.inRange(hsv, (160,60,60),(180,255,255))
    red = cv2.bitwise_or(r1, r2)
    green = cv2.inRange(hsv, (40,60,60), (85,255,255))

    rc = int(cv2.countNonZero(red))
    gc = int(cv2.countNonZero(green))
    if max(rc,gc) < 200:
        return None
    return "red" if rc>gc else "green"

def debounce_signals(states, window=3):
    out = list(states)
    n   = len(states)
    for i, s in enumerate(states):
        if s in ("red","green"):
            cnt = sum(1 for j in range(max(0,i-window), min(n,i+window+1))
                      if states[j]==s)
            if cnt < 2:
                out[i] = None
    return out

# ─── SUMMARY ────────────────────────────────────────────────────────────────
def generate_long_summary(events, signals, *args, **kwargs):
    parts, last_sig = [], None
    for ev, sig in zip(events, signals):
        if sig=="red"   and last_sig!="red":
            parts.append("stopped at the red light")
        if sig=="green" and last_sig=="red":
            parts.append("when it turned green, I drove on")
        last_sig = sig or last_sig

        if ev.startswith("passed "):
            parts.append(f"passed {ev.split(' ',1)[1]}")
        if ev=="turn_left":
            parts.append("turned left")
        if ev=="turn_right":
            parts.append("took a slight right")

    if not parts or not parts[0].startswith("stopped"):
        parts.insert(0, "drove straight")

    # collapse duplicates
    out = [parts[0]]
    for p in parts[1:]:
        if p != out[-1]:
            out.append(p)

    sent = out[0].capitalize()
    for p in out[1:]:
        sent += " and " + p
    return sent + ("" if sent.endswith(".") else ".")
