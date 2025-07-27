# src/utils.py

import os
import cv2
import urllib.request
import numpy as np
from ultralytics import YOLO

def frames(path: str, fps: int = 1):
    """
    Yield one frame per second from a video file, or
    all JPGs in a directory (sorted lexically).
    """
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(".jpg"):
                img = cv2.imread(os.path.join(path, fn))
                if img is not None:
                    yield img
        return

    cap = cv2.VideoCapture(path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(original_fps / fps))
    idx = 0
    ok, frame = cap.read()
    while ok:
        if idx % step == 0:
            yield frame
        ok, frame = cap.read()
        idx += 1
    cap.release()

def move(prev_gray, curr_gray, dx: float = 1.5, stop_thr: float = 0.2) -> str:
    """
    Compute a simple motion verb via Farneback optical flow on grayscale frames.
    Returns one of: 'drive', 'stop', 'turn_left', 'turn_right'.
    """
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    dx_mean = flow[..., 0].mean()
    mag_mean = np.linalg.norm(flow, axis=2).mean()
    if mag_mean < stop_thr:
        return "stop"
    if dx_mean > dx:
        return "turn_right"
    if dx_mean < -dx:
        return "turn_left"
    return "drive"

def load_yolo(dev: str, model_path: str = None):
    """
    Load a YOLOv8 model onto device 'cpu' or 'cuda'.
    If `model_path` is None, downloads/uses the default yolov8n.pt.
    """
    if model_path:
        pt = model_path
    else:
        os.makedirs("models", exist_ok=True)
        pt = os.path.join("models", "yolov8n.pt")
        if not os.path.exists(pt):
            urllib.request.urlretrieve(
                "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                pt
            )
    model = YOLO(pt).to(dev).half()
    return model

def detect_signal_color(roi) -> str:
    """
    Given a BGR crop of a traffic light, return 'red', 'green', or None.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # red mask (two ranges)
    lower1, upper1 = np.array([0,70,50]),   np.array([10,255,255])
    lower2, upper2 = np.array([170,70,50]), np.array([180,255,255])
    red_mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    red_count = int(cv2.countNonZero(red_mask))
    # green mask
    lower_g, upper_g = np.array([40,40,40]), np.array([90,255,255])
    green_mask = cv2.inRange(hsv, lower_g, upper_g)
    green_count = int(cv2.countNonZero(green_mask))
    if green_count > red_count and green_count > 50:
        return "green"
    if red_count > green_count and red_count > 50:
        return "red"
    return None
