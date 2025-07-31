# src/utils.py
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Your five manually-labeled keyframes & their in-frame text
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub"
]

# ─────────────────────────────────────────────────────────────────────────────
def extract_frames(src, fps=1):
    """
    Yields one frame per second, either from a folder of .jpg or a video file.
    """
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
def detect_event(prev_gray, cur_gray, dx_thresh=1.5, stop_thresh=0.2):
    """
    Simple optical-flow based motion classifier:
      • stop      ↔ magnitude < stop_thresh
      • turn_right↔ mean_dx > dx_thresh
      • turn_left ↔ mean_dx < –dx_thresh
      • drive     ↔ otherwise
    """
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    dx  = float(np.mean(flow[..., 0]))
    mag = float(np.linalg.norm(flow, axis=2).mean())
    if mag < stop_thresh:
        return "stop"
    if dx > dx_thresh:
        return "turn_right"
    if dx < -dx_thresh:
        return "turn_left"
    return "drive"

# ─────────────────────────────────────────────────────────────────────────────
_yolo_sig = None
def load_signal_model(device="cpu"):
    """
    Loads YOLOv8n (or your custom) for traffic-light detection.
    """
    global _yolo_sig
    if _yolo_sig is None:
        _yolo_sig = YOLO("yolov8n.pt").to(device).half()
    return _yolo_sig

def detect_signal_color(frame, yolo, conf=0.15):
    """
    Finds the largest 'traffic light' box via YOLO, crops it,
    applies HSV thresholds for red/green, and returns "red"/"green"/None.
    """
    r = yolo(frame, conf=conf, verbose=False)[0]
    boxes = []
    for b in r.boxes:
        cls = yolo.model.names[int(b.cls[0])]
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
    if max(rc, gc) < 100:
        return None
    return "red" if rc > gc else "green"

# ─────────────────────────────────────────────────────────────────────────────
def debounce_events(evts, window=3, min_count=3):
    """
    Only keep a turn if it appears at least `min_count` times
    in a ±window neighborhood; otherwise reset to 'drive'.
    """
    out = evts.copy()
    n   = len(evts)
    for i, e in enumerate(evts):
        if e in ("turn_left", "turn_right"):
            cnt = sum(
                1
                for j in range(max(0,i-window), min(n,i+window+1))
                if evts[j] == e
            )
            if cnt < min_count:
                out[i] = "drive"
    return out

def debounce_signals(sigs, window=3):
    """
    Only keep a red/green if it appears in at least 2 frames
    in a ±window window; else None.
    """
    out = [None]*len(sigs)
    n   = len(sigs)
    for i, s in enumerate(sigs):
        if s in ("red","green"):
            cnt = sum(
                1
                for j in range(max(0,i-window), min(n,i+window+1))
                if sigs[j] == s
            )
            if cnt >= 2:
                out[i] = s
    return out

# ─────────────────────────────────────────────────────────────────────────────
def generate_summary(events, signals):
    """
    Build a human-readable journey string from the debounced
    event+signal streams.
    """
    parts, last_sig = [], None

    for idx, (e, s) in enumerate(zip(events, signals), start=1):
        # initial red
        if idx==1 and s=="red":
            parts.append("stopped at the red light")
            last_sig="red"
            continue

        # green after a red
        if last_sig=="red" and s=="green":
            parts.append("once it turned green, I drove on")
            last_sig="green"

        # landmark passes
        if e.startswith("passed "):
            name = e.split(" ",1)[1]
            parts.append(f"passed {name}")

        # turns (already debounced)
        if e=="turn_left":
            parts.append("turned left")
        elif e=="turn_right":
            parts.append("took a slight right")

        # new red after anything else
        if s=="red" and last_sig!="red":
            parts.append("then stopped at the red light")
            last_sig="red"

    # if never stopped at start, we began by driving straight
    if not parts or not parts[0].startswith("stopped"):
        parts.insert(0, "drove straight")

    # dedupe consecutive duplicates
    clean = [parts[0]]
    for p in parts[1:]:
        if p != clean[-1]:
            clean.append(p)

    # join into one sentence
    sent = clean[0].capitalize()
    for p in clean[1:]:
        if p.startswith(("passed","turned","took")):
            sent += " and " + p
        else:
            sent += ", " + p

    # always end by noting continued straight if last event was a drive/pass
    if events and (events[-1]=="drive" or events[-1].startswith("passed ")):
        sent += " and continued straight."
    else:
        sent = sent.rstrip('.') + "."

    return sent
