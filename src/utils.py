import os
import cv2
import urllib.request
import numpy as np
from ultralytics import YOLO

def fetch(dst_dir: str, url: str, fname: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, fname)
    if url and not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path

def frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    nat = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(nat / fps))
    idx, ok, img = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield img
        ok, img = cap.read()
        idx += 1
    cap.release()

def load_detector(dev: str, model_path: str = None):
    """
    Load a YOLO model (custom .pt or default yolov8n) on device `dev`.
    Returns the model.
    """
    if model_path:
        model = YOLO(model_path).to(dev).half()
    else:
        pt = fetch(
            "models",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "yolov8n.pt",
        )
        model = YOLO(pt).to(dev).half()
    return model

def detect_signal_color(roi):
    """
    Given a cropped traffic-light ROI, return "red"/"green" or None.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # red masks
    lower1, upper1 = np.array([0,70,50]), np.array([10,255,255])
    lower2, upper2 = np.array([170,70,50]), np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    red = int(cv2.countNonZero(mask1) + cv2.countNonZero(mask2))
    # green mask
    lowerg, upperg = np.array([40,40,40]), np.array([90,255,255])
    green = int(cv2.countNonZero(cv2.inRange(hsv, lowerg, upperg)))
    if green > red and green > 50:
        return "green"
    if red > green and red > 50:
        return "red"
    return None
