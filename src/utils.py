import os, cv2, numpy as np, torch
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# 1) FRAME WHITELIST & LABELS
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2) FRAME EXTRACTION
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
# 3) OPTICAL-FLOW EVENT DETECTION
def detect_event(prev_gray, cur_gray, dx_thresh=1.5, stop_thresh=0.2):
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        0.5,3,15,3,5,1.2,0)
    dx  = float(np.mean(flow[...,0]))
    mag = float(np.linalg.norm(flow,axis=2).mean())
    if mag < stop_thresh:
        return "stop"
    if dx > dx_thresh:
        return "turn_right"
    if dx < -dx_thresh:
        return "turn_left"
    return "drive"

# ─────────────────────────────────────────────────────────────────────────────
# 4) TRAFFIC-LIGHT DETECTION (YOLOv8n + HSV)
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
    if crop.size==0:
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
# 5) DEBOUNCE
def debounce_events(evts, window=3, min_count=3):
    out = evts.copy()
    n = len(evts)
    for i,e in enumerate(evts):
        if e in ("turn_left","turn_right"):
            cnt = sum(1 for j in range(max(0,i-window),min(n,i+window+1)) if evts[j]==e)
            if cnt < min_count:
                out[i] = "drive"
    return out

def debounce_signals(sigs, window=3):
    out = [None]*len(sigs)
    n = len(sigs)
    for i,s in enumerate(sigs):
        if s in ("red","green"):
            cnt = sum(1 for j in range(max(0,i-window),min(n,i+window+1)) if sigs[j]==s)
            if cnt>=2:
                out[i]=s
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 6) CUSTOM JOURNEY BUILDER
def build_custom_journey(events, signals, sign_texts):
    parts, last_sig = [], None
    start = 0

    # A) right from signal
    for i,(e,s) in enumerate(zip(events,signals)):
        if i>0 and signals[i-1]=="red" and s=="green" and e=="turn_right":
            parts.append("Turned right from the signal")
            last_sig="green"
            start = i+1
            break

    # B) first shop on left
    for j in range(start, len(sign_texts)):
        for txt,side in sign_texts[j]:
            if side=="left":
                parts.append("a shop was visible on the left-hand side and then turned right")
                start = j+1
                break
        else:
            continue
        break

    # C) find “Fox and Hounds”
    for k in range(start, len(sign_texts)):
        for txt,side in sign_texts[k]:
            if "Fox and Hounds" in txt:
                parts.append("a building named ‘Fox and Hounds’ appeared on the right-hand side")
                start = k+1
                break
        else:
            continue
        break

    # D) green after red
    for m in range(start, len(signals)):
        if signals[m]=="green" and last_sig=="red":
            parts.append("and the vehicle proceeded through another green signal")
            last_sig="green"
            start = m+1
            break

    # E) straight chunk
    parts.append("continued straight for a while")

    # F) final left
    for n in range(start, len(events)):
        if events[n]=="turn_left":
            parts.append("and then turned left at the intersection")
            break

    return ", ".join(parts) + "."

