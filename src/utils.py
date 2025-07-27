import os
import cv2
import numpy as np
from ultralytics import YOLO

# ─── 1) FRAME EXTRACTION ────────────────────────────────────────────────────
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

# ─── 2) MOTION DETECTION ────────────────────────────────────────────────────
def detect_event_for_frame(prev_gray, cur_gray, dx_thresh=5.0, stop_thresh=0.4):
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    dx  = flow[...,0].mean()
    mag = np.linalg.norm(flow, axis=2).mean()
    if mag < stop_thresh:
        return "stop"
    if dx > dx_thresh:
        return "turn_right"
    if dx < -dx_thresh:
        return "turn_left"
    return "drive"

def debounce_lane_changes(events, window=5):
    out = list(events)
    n = len(events)
    for i,e in enumerate(events):
        if e in ("turn_left","turn_right"):
            cnt = sum(
                1
                for j in range(max(0,i-window), min(n,i+window+1))
                if events[j]==e
            )
            if cnt < 2:
                out[i] = "drive"
    return out

# ─── 3) SIGNAL DETECTION ────────────────────────────────────────────────────
def get_yolo_model(device):
    # expects yolov8n.pt in project root
    model = YOLO("yolov8n.pt").to(device).half()
    return model

def detect_signal_color(frame, yolo_model, conf=0.15):
    r = yolo_model(frame, conf=conf, verbose=False)[0]
    cand = []
    for b in r.boxes:
        cls = yolo_model.model.names[int(b.cls[0])]
        if cls != "traffic light":
            continue
        x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
        w,h = x2-x1, y2-y1
        if h < w * 1.2:  # require tall box
            continue
        cand.append((x1,y1,x2,y2))
    if not cand:
        return None
    # select largest area
    x1,y1,x2,y2 = max(cand, key=lambda bb:(bb[2]-bb[0])*(bb[3]-bb[1]))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1  = cv2.inRange(hsv, (0,60,60), (10,255,255))
    red2  = cv2.inRange(hsv, (160,60,60),(180,255,255))
    green = cv2.inRange(hsv, (40,60,60), (85,255,255))

    rc = int(cv2.countNonZero(cv2.bitwise_or(red1,red2)))
    gc = int(cv2.countNonZero(green))
    if max(rc,gc) < 200:
        return None
    return "red" if rc > gc else "green"

def debounce_signals(states, window=3):
    out = list(states)
    n = len(states)
    for i,s in enumerate(states):
        if s in ("red","green"):
            cnt = sum(
                1
                for j in range(max(0,i-window), min(n,i+window+1))
                if states[j] == s
            )
            if cnt < 2:
                out[i] = None
    return out

# ─── 4) FINAL SUMMARY ───────────────────────────────────────────────────────
def generate_long_summary(events, signals, *_, **__):
    parts, last_sig = [], None

    for ev,sig in zip(events, signals):
        if sig == "red" and last_sig != "red":
            parts.append("stopped at the red light")
        if sig == "green" and last_sig == "red":
            parts.append("when it turned green, I drove on")
        if ev.startswith("passed "):
            parts.append(f"passed {ev.split(' ',1)[1]}")
        if ev == "turn_left":
            parts.append("turned left")
        if ev == "turn_right":
            parts.append("took a slight right")
        last_sig = sig or last_sig

    if not parts or not parts[0].startswith("stopped"):
        parts.insert(0, "drove straight")

    # collapse duplicates
    out = [parts[0]]
    for p in parts[1:]:
        if p != out[-1]:
            out.append(p)

    # build sentence
    sentence = out[0].capitalize()
    for p in out[1:]:
        sentence += " and " + p
    if not sentence.endswith("."):
        sentence += "."
    return sentence
