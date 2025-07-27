import os
import cv2
import numpy as np
from ultralytics import YOLO

# ─── FRAME EXTRACTION ────────────────────────────────────────────────────────
def extract_frames(path, fps=1):
    # if folder of .jpg
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(".jpg"):
                img = cv2.imread(os.path.join(path, fn))
                if img is not None:
                    yield img
        return
    # else video
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

# ─── MOTION DETECTION ───────────────────────────────────────────────────────
def detect_event_for_frame(prev_gray, cur_gray,
                           dx_thresh=3.0, stop_thresh=0.2):
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    dxm = flow[...,0].mean()
    mag = np.linalg.norm(flow, axis=2).mean()
    if mag < stop_thresh:
        return "stop"
    if dxm > dx_thresh:
        return "turn_right"
    if dxm < -dx_thresh:
        return "turn_left"
    return "drive"

def debounce_lane_changes(events, window=3):
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

# ─── SIGNAL DETECTION WITH FALLBACK ────────────────────────────────────────
def get_yolo_model(device):
    # assumes yolov8n.pt sits in your project root
    model = YOLO("yolov8n.pt").to(device).half()
    return model

def detect_signal_color(frame, yolo_model, conf=0.2):
    # 1) YOLO box
    r = yolo_model(frame, conf=conf, verbose=False)[0]
    boxes = [
        tuple(map(int, b.xyxy[0].cpu().numpy()))
        for b in r.boxes
        if yolo_model.model.names[int(b.cls[0])] == "traffic light"
    ]
    if boxes:
        # largest
        areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes]
        x1,y1,x2,y2 = boxes[int(np.argmax(areas))]
        crop = frame[y1:y2, x1:x2]
    else:
        # fallback: top‑center slice
        h,w = frame.shape[:2]
        crop = frame[: h//3, w//3 : 2*w//3]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0,60,60), (10,255,255))
    m2 = cv2.inRange(hsv, (160,60,60), (180,255,255))
    red = cv2.bitwise_or(m1, m2)
    green = cv2.inRange(hsv, (40,60,60), (85,255,255))

    rc = int(red.sum()/255)
    gc = int(green.sum()/255)
    if max(rc,gc) < 100:
        return None
    return "red" if rc>gc else "green"

def debounce_signals(states, window=2):
    out = list(states)
    n = len(states)
    for i,s in enumerate(states):
        if s in ("red","green"):
            cnt = sum(
                1
                for j in range(max(0,i-window), min(n,i+window+1))
                if states[j]==s
            )
            if cnt < 2:
                out[i] = None
    return out

# ─── FINAL SUMMARY BUILDER ─────────────────────────────────────────────────
def generate_long_summary(events, signals, *args, **kwargs):
    parts = []
    stopped = False

    for ev, sig in zip(events, signals):
        if sig == "red" and not stopped:
            parts.append("stopped at the red light")
            stopped = True
            continue
        if sig == "green" and stopped:
            parts.append("when it turned green, I drove on")
            stopped = False
            continue

        if ev == "drive":
            continue
        if ev.startswith("passed "):
            parts.append(f"passed {ev[len('passed '):]}")
            continue
        if ev == "turn_left":
            parts.append("turned left")
            continue
        if ev == "turn_right":
            parts.append("took a slight right")
            continue
        if ev == "stop":
            parts.append("paused briefly")
            continue

    # ensure we start by driving
    if parts and not parts[0].startswith("stopped"):
        parts.insert(0, "drove straight")

    # join
    journey = parts[0].capitalize()
    for p in parts[1:]:
        journey += " and " + p
    if not journey.endswith("."):
        journey += "."
    return journey
