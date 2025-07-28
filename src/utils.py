# src/utils.py
import os, cv2, numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# 1) FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
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
    step = max(1, round(nat/fps))
    idx, ok, frame = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield frame
        ok, frame = cap.read()
        idx += 1
    cap.release()

# ─────────────────────────────────────────────────────────────────────────────
# 2) MOTION‑BASED EVENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_event_for_frame(prev, cur, dx_thresh=2.5, stop_thresh=0.3):
    if prev is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None,
                                        0.5,3,15,3,5,1.2,0)
    dxm = float(np.median(flow[...,0]))
    mag = float(np.linalg.norm(flow,axis=2).mean())
    if mag < stop_thresh:    return "stop"
    if dxm > dx_thresh:      return "turn_right"
    if dxm < -dx_thresh:     return "turn_left"
    return "drive"

def debounce_lane_changes(evts, window=3, min_count=3):
    out = evts.copy()
    n = len(evts)
    for i,e in enumerate(evts):
        if e in ("turn_left","turn_right"):
            cnt = sum(1 for j in range(max(0,i-window),
                                       min(n,i+window+1))
                      if evts[j]==e)
            if cnt < min_count:
                out[i] = "drive"
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 3) YOLO SIGNAL‑COLOR DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def get_yolo_model(device):
    # ensure models/yolov8n.pt exists
    return YOLO("models/yolov8n.pt").to(device).half()

def detect_signal_color(frame, yolo, conf=0.15):
    r = yolo(frame, conf=conf, verbose=False)[0]
    boxes = []
    for b in r.boxes:
        cls = yolo.model.names[int(b.cls[0])]
        if cls == "traffic light":
            x1,y1,x2,y2 = map(int,b.xyxy[0].cpu().numpy())
            boxes.append((x1,y1,x2,y2))
    if boxes:
        x1,y1,x2,y2 = max(boxes,
                          key=lambda bb: (bb[2]-bb[0])*(bb[3]-bb[1]))
        crop = frame[y1:y2,x1:x2]
    else:
        h,w = frame.shape[:2]
        crop = frame[0:int(0.2*h), int(0.3*w):int(0.7*w)]
    if crop.size==0: return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv,(0,80,80),(10,255,255))
    r2 = cv2.inRange(hsv,(160,80,80),(180,255,255))
    red = cv2.bitwise_or(r1,r2)
    green = cv2.inRange(hsv,(40,80,80),(85,255,255))
    rc, gc = int(cv2.countNonZero(red)), int(cv2.countNonZero(green))
    if max(rc,gc) < 200:
        return None
    return "red" if rc>gc else "green"

def debounce_signals(states, window=3):
    out = [None]*len(states)
    n = len(states)
    for i,s in enumerate(states):
        if s in ("red","green"):
            cnt = sum(1 for j in range(max(0,i-window),
                                       min(n,i+window+1))
                      if states[j]==s)
            if cnt >= 3:
                out[i] = s
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 4) SUMMARY GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_long_summary(events, signals):
    parts = []
    last_sig = None
    drive_run = 0

    def flush_drive():
        nonlocal drive_run
        if drive_run >= 2:
            parts.append("continued straight for a while")
        drive_run = 0

    for ev, sig in zip(events, signals):
        # handle signal
        if sig=="red" and last_sig!="red":
            flush_drive()
            parts.append("stopped at the red light")
        if sig=="green" and last_sig=="red":
            parts.append("once the signal turned green, I drove on")
        last_sig = sig or last_sig

        # handle event
        if ev.startswith("passed "):
            flush_drive()
            landmark = ev.split(" ",1)[1]
            parts.append(f"passed {landmark}")
        elif ev=="turn_left":
            flush_drive()
            parts.append("turned left")
        elif ev=="turn_right":
            flush_drive()
            parts.append("took a slight right")
        else:  # drive
            drive_run += 1

    flush_drive()
    if not parts:
        return "No events detected."
    sent = parts[0].capitalize()
    for p in parts[1:]:
        sent += " and " + p
    if not sent.endswith("."):
        sent += "."
    return sent
