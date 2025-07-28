import os
import cv2
import numpy as np
from ultralytics import YOLO

# ─── FRAME EXTRACTION ────────────────────────────────────────────────────────
def extract_frames(path, fps=1):
    """Yield one frame per second from a video or all .jpg in a folder."""
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
    """Optical‑flow median dx → drive/stop/turn_left/turn_right."""
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    dxm = float(np.median(flow[..., 0]))
    mag = float(np.linalg.norm(flow, axis=2).mean())
    if mag < stop_thresh:
        return "stop"
    if dxm > dx_thresh:
        return "turn_right"
    if dxm < -dx_thresh:
        return "turn_left"
    return "drive"

def debounce_lane_changes(events, window=5):
    """Only keep a turn if it repeats ≥2 times within ±window."""
    out = list(events)
    n = len(events)
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

# ─── TRAFFIC‑LIGHT DETECTION ─────────────────────────────────────────────────
def get_yolo_model(device):
    """Load yolov8n for traffic‑light detection."""
    return YOLO("yolov8n").to(device).half()

def detect_signal_color(frame, yolo, conf=0.1):
    """
    Run YOLO traffic‑light → crop largest box or top band → HSV mask red/green.
    Returns "red", "green", or None.
    """
    r = yolo(frame, conf=conf, verbose=False)[0]
    boxes = []
    for b in r.boxes:
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
        top = int(0.2 * h)
        left = int(0.3 * w)
        right = int(0.7 * w)
        crop = frame[0:top, left:right]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    r2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(r1, r2)
    green_mask = cv2.inRange(hsv, (40, 80, 80), (85, 255, 255))
    rc = int(cv2.countNonZero(red_mask))
    gc = int(cv2.countNonZero(green_mask))
    if max(rc, gc) < 100:
        return None
    return "red" if rc > gc else "green"

def debounce_signals(states, window=3):
    """
    Only keep a red/green if it appears ≥3 times in ±window frames.
    """
    out = [None] * len(states)
    n = len(states)
    for i, s in enumerate(states):
        if s in ("red", "green"):
            cnt = sum(
                1
                for j in range(max(0, i - window), min(n, i + window + 1))
                if states[j] == s
            )
            if cnt >= 3:
                out[i] = s
    return out

# ─── ONE‑SENTENCE SUMMARY ────────────────────────────────────────────────────
def summarise_frames(events, signals):
    """
    Build your single-sentence summary from parallel lists:
      - events[i] ∈ {"drive","stop","turn_left","turn_right","passed X"}
      - signals[i] ∈ {None,"red","green"}
    """
    n = len(events)

    def find_run(start, val):
        i = start
        while i < n and signals[i] != val:
            i += 1
        if i >= n:
            return None, None
        j = i
        while j < n and signals[j] == val:
            j += 1
        return i, j

    def collect_passes(a, b):
        names = []
        for e in events[a:b]:
            if e.startswith("passed "):
                names.append(e.split(" ", 1)[1])
        return names

    parts = []
    idx = 0

    # initial red-run
    r0, r0e = find_run(0, "red")
    if r0 is not None:
        parts.append("I stopped at the red light")
        idx = r0e
    else:
        parts.append("I drove straight")

    # passes before next red
    names = collect_passes(idx, r0 or n)
    if names:
        parts.append("passed " + " and ".join(names))
    if r0 is not None:
        parts.append("stopped at the red light again")
        idx = r0e

    # passes before second red
    r1, r1e = find_run(idx, "red")
    names = collect_passes(idx, r1 or n)
    if names:
        parts.append("drove on and passed " + " and ".join(names))
    if r1 is not None:
        parts.append("stopped at the red light once more")
        idx = r1e

    # final passes
    names = collect_passes(idx, n)
    if names:
        parts.append("passed " + " and ".join(names))

    # always end with continued straight
    parts.append("continued straight")

    # join into one sentence
    sent = parts[0]
    for p in parts[1:]:
        sent += ", then " + p
    return sent + "."
