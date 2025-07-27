# src/utils.py

import os, cv2, urllib.request, numpy as np
from ultralytics import YOLO

def frames(path: str, fps: int = 1):
    """Yield one frame per second from video or sorted JPG folder."""
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(".jpg"):
                yield cv2.imread(os.path.join(path,fn))
        return
    cap = cv2.VideoCapture(path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(orig_fps/fps))
    idx, ok = 0, True
    ok, frame = cap.read()
    while ok:
        if idx % step == 0:
            yield frame
        ok, frame = cap.read()
        idx += 1
    cap.release()

def load_custom_yolo(dev: str, weights: str):
    """
    Load your custom‑trained YOLOv8 (.pt) onto the chosen device.
    """
    model = YOLO(weights).to(dev).half()
    return model

def detect_signal_color(roi) -> str:
    """Simple HSV test for red vs green."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # red mask
    r1,r2 = np.array([0,70,50]), np.array([10,255,255])
    r3,r4 = np.array([170,70,50]), np.array([180,255,255])
    red = int(cv2.countNonZero(cv2.inRange(hsv,r1,r2)) +
              cv2.countNonZero(cv2.inRange(hsv,r3,r4)))
    # green mask
    g1,g2 = np.array([40,40,40]), np.array([90,255,255])
    green = int(cv2.countNonZero(cv2.inRange(hsv,g1,g2)))
    if green>red>50: return "green"
    if red>green>50: return "red"
    return None
