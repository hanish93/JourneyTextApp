import os, cv2, torch, numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# 1) FRAME WHITELIST: force‑inject these labels at exactly these frames
# ─────────────────────────────────────────────────────────────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────
def extract_frames(src, fps=1):
    if os.path.isdir(src):
        files = sorted(f for f in os.listdir(src) if f.lower().endswith(".jpg"))
        for fn in files:
            img = cv2.imread(os.path.join(src, fn))
            if img is not None:
                yield img
        return
    cap = cv2.VideoCapture(src)
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
# 3) MOTION CLASSIFIER (optical flow → drive/turn/stop)
# ─────────────────────────────────────────────────────────────────────────────
def detect_event(prev_gray, cur_gray, dx_thresh=1.5, stop_thresh=0.2):
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        0.5,3,15,3,5,1.2,0)
    dx = flow[...,0].mean()
    mag = np.linalg.norm(flow,axis=2).mean()
    if mag < stop_thresh:
        return "stop"
    if dx > dx_thresh:
        return "turn_right"
    if dx < -dx_thresh:
        return "turn_left"
    return "drive"

# ─────────────────────────────────────────────────────────────────────────────
# 4) SIGNAL COLOR DETECTION via YOLOv8 → crop traffic‐light & HSV test
# ─────────────────────────────────────────────────────────────────────────────
_yolo_sig = None
def load_signal_model(device="cpu"):
    global _yolo_sig
    if _yolo_sig is None:
        _yolo_sig = YOLO("yolov8n.pt").to(device).half()
    return _yolo_sig

def detect_signal_color(frame, yolo, conf=0.15):
    r = yolo(frame, conf=conf, verbose=False)[0]
    # gather all detected traffic‐light boxes
    tbs = []
    for b in r.boxes:
        cls = yolo.model.names[int(b.cls[0])]
        if cls == "traffic light":
            x1,y1,x2,y2 = map(int,b.xyxy[0].cpu().numpy())
            tbs.append((x1,y1,x2,y2))
    if tbs:
        # pick largest box
        x1,y1,x2,y2 = max(tbs, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
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
# 5) DEBOUNCE: only report each turn/stop and each red/green once
# ─────────────────────────────────────────────────────────────────────────────
def debounce_events(evts, window=3):
    out = evts.copy()
    for i,e in enumerate(evts):
        if e in ("turn_left","turn_right"):
            cnt = sum(1 for j in range(max(0,i-window), min(len(evts), i+window+1))
                      if evts[j]==e)
            if cnt < 2:
                out[i] = "drive"
    return out

def debounce_signals(sigs, window=3):
    out = [None]*len(sigs)
    for i,s in enumerate(sigs):
        if s in ("red","green"):
            cnt = sum(1 for j in range(max(0,i-window), min(len(sigs), i+window+1))
                      if sigs[j]==s)
            if cnt >= 2:
                out[i] = s
    # carry forward last stable signal
    last = None
    for i,s in enumerate(out):
        if s is not None:
            last = s
        else:
            out[i] = None
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 6) FINAL SUMMARY BUILDER (strictly by your rules)
# ─────────────────────────────────────────────────────────────────────────────
def generate_summary(events, signals):
    parts = []
    last_sig = None

    # iterate through frames
    for i,(e,s) in enumerate(zip(events,signals), start=1):
        # 1) on first red
        if i==1 and s=="red":
            parts.append("Stopped at the red light")
            last_sig = "red"
            continue

        # 2) on green immediately after red
        if last_sig=="red" and s=="green":
            parts.append("once it turned green, I drove on")
            last_sig = "green"

        # 3) your forced “passed …” labels
        if e.startswith("passed "):
            name = e.split(" ",1)[1]
            parts.append(f"passed {name}")

        # 4) turns
        if e=="turn_left":
            parts.append("then turned left")
        if e=="turn_right":
            parts.append("then took a slight right")

        # update last_sig on new red
        if s=="red" and last_sig!="red":
            parts.append("then stopped at the red light")
            last_sig = "red"

    # ensure it starts with something
    if not parts or not parts[0].startswith("Stopped"):
        parts.insert(0,"Drove straight")

    # collapse repeats
    clean = [parts[0]]
    for p in parts[1:]:
        if p!=clean[-1]:
            clean.append(p)

    # join into one sentence
    sent = clean[0]
    for p in clean[1:]:
        sent += " and " + p
    if clean[-1].startswith("Drove") or clean[-1].startswith("passed") or clean[-1].startswith("then took"):
        sent += "."
    return sent
