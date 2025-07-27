import os
import cv2
import urllib.request
import numpy as np
from ultralytics import YOLO

def frames(path: str, fps: int = 1):
    """
    Yield one frame per second from a .mp4 or
    all .jpgs in a folder (sorted).
    """
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(".jpg"):
                yield cv2.imread(os.path.join(path, fn))
        return

    cap = cv2.VideoCapture(path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(orig_fps / fps))
    idx = 0
    ok, frame = cap.read()
    while ok:
        if idx % step == 0:
            yield frame
        ok, frame = cap.read()
        idx += 1
    cap.release()

def move(prev_gray, cur_gray, dx=1.5, stop_thr=0.2):
    """
    Optical‑flow → verb.  We only care about left/right turns.
    """
    if prev_gray is None:
        return None
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    dxm = flow[...,0].mean()
    mag = np.linalg.norm(flow,axis=2).mean()
    if dxm > dx:
        return "turn_right"
    if dxm < -dx:
        return "turn_left"
    return None

def load_yolo(dev: str, weights: str = None):
    """
    Load YOLOv8 for 'traffic light'.  Defaults to yolov8n.pt.
    """
    if weights:
        m = YOLO(weights).to(dev).half()
    else:
        os.makedirs("models", exist_ok=True)
        pt = os.path.join("models", "yolov8n.pt")
        if not os.path.exists(pt):
            urllib.request.urlretrieve(
                "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                pt
            )
        m = YOLO(pt).to(dev).half()
    return m

def detect_signal_color(roi) -> str:
    """
    HSV masks → 'red' or 'green' if prominent, else None.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # red
    r1, r2 = np.array([0,70,50]), np.array([10,255,255])
    r3, r4 = np.array([170,70,50]),np.array([180,255,255])
    red = int(cv2.countNonZero(cv2.inRange(hsv, r1, r2)) +
              cv2.countNonZero(cv2.inRange(hsv, r3, r4)))
    # green
    g1, g2 = np.array([40,40,40]), np.array([90,255,255])
    green = int(cv2.countNonZero(cv2.inRange(hsv, g1, g2)))
    if green > red and green > 50:
        return "green"
    if red > green and red > 50:
        return "red"
    return None
