import os
import cv2
import numpy as np
from ultralytics import YOLO

# ─── FRAME EXTRACTION ────────────────────────────────────────────────────────
def extract_frames(path, fps=1):
    """
    Yield one frame per second from a video, or all .jpg images in a folder.
    """
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
    """
    Compute optical flow, median-filter dx, then classify:
    'drive', 'stop', 'turn_left', or 'turn_right'.
    """
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    dx = flow[..., 0]
    dxm = float(np.median(dx))
    mag = float(np.linalg.norm(flow, axis=2).mean())

    if mag < stop_thresh:
        return "stop"
    if dxm > dx_thresh:
        return "turn_right"
    if dxm < -dx_thresh:
        return "turn_left"
    return "drive"

def debounce_lane_changes(events, window=5):
    """
    Keep a turn only if it repeats at least once within ±window frames.
    Otherwise treat as 'drive'.
    """
    out = list(events)
    n   = len(events)
    for i, e in enumerate(events):
        if e in ("turn_left", "turn_right"):
            cnt = sum(
                1
                for j in range(max(0, i - window), min(n, i + window + 1))
                if events[j] == e
            )
            if cnt < 2:
                out[i] = "drive"
    return out

# ─── SIGNAL DETECTION ───────────────────────────────────────────────────────
def get_yolo_model(device):
    """
    Load default yolov8n weights (auto‑download if missing).
    """
    return YOLO("yolov8n").to(device).half()

def detect_signal_color(frame, yolo, conf=0.1):
    """
    1) Run YOLO traffic‑light detection at low confidence.
    2) If found, crop largest box; else fallback to top‑center 20%.
    3) HSV‑mask for red vs green; return only if >=100 pixels.
    """
    res = yolo(frame, conf=conf, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        cls = yolo.model.names[int(b.cls[0])]
        if cls == "traffic light":
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
            boxes.append((x1, y1, x2, y2))

    if boxes:
        x1, y1, x2, y2 = max(
            boxes, key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1])
        )
        crop = frame[y1:y2, x1:x2]
    else:
        h, w = frame.shape[:2]
        top    = int(0.2 * h)
        left   = int(0.3 * w)
        right  = int(0.7 * w)
        crop = frame[0:top, left:right]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    r1  = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    r2  = cv2.inRange(hsv, (160,80,80), (180,255,255))
    red_mask   = cv2.bitwise_or(r1, r2)
    green_mask = cv2.inRange(hsv, (40,80,80), (85,255,255))

    rc = int(cv2.countNonZero(red_mask))
    gc = int(cv2.countNonZero(green_mask))
    if max(rc, gc) < 100:
        return None
    return "red" if rc > gc else "green"

def debounce_signals(states):
    """
    Only keep a 'red' or 'green' if it appears in >=3 consecutive frames.
    Otherwise set to None. This collapses runs to a single mention.
    """
    out = [None] * len(states)
    n   = len(states)
    for i in range(n):
        s = states[i]
        if s in ("red", "green"):
            # check run of len 3: self ±1
            start = max(0, i-1)
            end   = min(n, i+2)
            cnt   = sum(1 for j in range(start, end) if states[j] == s)
            if cnt >= 3:
                out[i] = s
    return out

# ─── SUMMARY GENERATOR ─────────────────────────────────────────────────────
def generate_long_summary(events, signals, *args, **kwargs):
    """
    Build a single sentence:
    - Announce each red→green only once (after 3-frame run).
    - Include each 'passed X'.
    - Include each turn once.
    - If the journey ends 'drive', append 'continued straight.'
    """
    parts, last_sig = [], None
    for ev, sig in zip(events, signals):
        if sig == "red" and last_sig != "red":
            parts.append("stopped at the red light")
        if sig == "green" and last_sig == "red":
            parts.append("when it turned green, I drove on")
        last_sig = sig or last_sig

        if ev.startswith("passed "):
            parts.append(f"passed {ev.split(' ',1)[1]}")
        if ev == "turn_left":
            parts.append("turned left")
        if ev == "turn_right":
            parts.append("took a slight right")

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

    # end "continued straight."
    if events and events[-1] == "drive":
        sentence += " continued straight."
    elif not sentence.endswith("."):
        sentence += "."
    return sentence
