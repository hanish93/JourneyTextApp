import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 1) FRAME EXTRACTION ───────────────────────────────────────────────────────
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
    step = max(1, round(nat/fps))
    idx, ok, frame = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield frame
        ok, frame = cap.read()
        idx += 1
    cap.release()

# 2) SIGNAL (RED/GREEN) DETECTION ───────────────────────────────────────────
def load_light_model(device):
    return YOLO("models/traffic_lights.pt").to(device).half()

def detect_light_state(frame, model, conf=0.15):
    """
    Runs YOLOv8x to detect red_light / green_light / yellow_light.
    Returns one of {'red','green','yellow',None}.
    """
    res = model(frame, conf=conf, verbose=False)[0]
    for box in res.boxes:
        cls = model.model.names[int(box.cls[0])]
        if cls == "red_light":
            return "red"
        if cls == "green_light":
            return "green"
        if cls == "yellow_light":
            return "yellow"
    return None

# 3) ROAD & LANE SEGMENTATION ───────────────────────────────────────────────
def load_seg_model(device):
    return YOLO("models/yolov8s-seg.pt").to(device).half()

def detect_road_mask(frame, seg_model):
    """
    Returns a binary mask of the road/lane area from YOLOv8‑Seg.
    """
    seg = seg_model(frame, verbose=False)[0]
    # assume class 0 = road_surface, class 1 = lane_marking
    # masks is a list of per-instance binary masks
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for m,cls in zip(seg.masks.data, seg.masks.cls):
        # include both road_surface and lane_marking
        mask = cv2.bitwise_or(mask, (m.cpu().numpy().astype(np.uint8)*255))
    return mask

# 4) TURN‑DETECTION CNN ─────────────────────────────────────────────────────
def load_turn_model(device):
    m = torch.jit.load("models/turn_cnn.pt", map_location=device)
    m.eval()
    return m

def detect_turn(prev_gray, cur_gray, next_gray, model, device):
    """
    Compute Farneback flow on (prev→cur) and (cur→next),
    stack dx/dy flows and magnitude into a 3‑channel tensor,
    run the CNN to get straight/turn_left/turn_right.
    """
    def flow_maps(a,b):
        f = cv2.calcOpticalFlowFarneback(a,b,None,0.5,3,15,3,5,1.2,0)
        dx = f[...,0]; dy = f[...,1]
        mag = np.sqrt(dx*dx+dy*dy)
        return dx, dy, mag

    dx1,dy1,m1 = flow_maps(prev_gray, cur_gray)
    dx2,dy2,m2 = flow_maps(cur_gray, next_gray)
    # average
    dx = (dx1+dx2)/2; dy = (dy1+dy2)/2; mag = (m1+m2)/2
    # normalize to [0,1]
    def norm(x):
        x = x - x.min()
        return x / (x.max()+1e-6)
    inp = np.stack([norm(dx), norm(dy), norm(mag)], axis=0)
    tensor = torch.from_numpy(inp).unsqueeze(0).to(device).float()
    with torch.no_grad():
        logits = model(tensor)
        cls = int(logits.argmax(dim=1)[0])
    return ["straight","turn_left","turn_right"][cls]

# 5) SUMMARY BUILDER ────────────────────────────────────────────────────────
def generate_long_summary(events, lights):
    parts = []
    last_light = None

    for ev, lt in zip(events, lights):
        if lt == "red" and last_light != "red":
            parts.append("stopped at the red light")
        if lt == "green" and last_light == "red":
            parts.append("when the light turned green, I drove on")
        last_light = lt or last_light

        if ev.startswith("passed "):
            parts.append(f"passed {ev.split(' ',1)[1]}")
        if ev == "turn_left":
            parts.append("turned left")
        if ev == "turn_right":
            parts.append("took a slight right")

    if not parts or not parts[0].startswith("stopped"):
        parts.insert(0, "drove straight")

    # collapse repeats
    out = [parts[0]]
    for p in parts[1:]:
        if p != out[-1]:
            out.append(p)

    sent = out[0].capitalize()
    for p in out[1:]:
        sent += " and " + p
    return sent + ("" if sent.endswith(".") else ".")

