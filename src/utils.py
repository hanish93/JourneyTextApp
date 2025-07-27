# src/utils.py

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

def frames(video_path: str, fps=1):
    cap = cv2.VideoCapture(video_path)
    nat = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(nat/fps))
    idx, ok, img = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield img
        ok, img = cap.read()
        idx += 1
    cap.release()

def load_detector(dev: str, model_path: str = None):
    """
    Load a YOLOv8 model. If `model_path` is given, use that file;
    otherwise download/use default yolov8n.pt.
    """
    if model_path:
        model = YOLO(model_path).to(dev).half()
    else:
        pt = fetch(
            "models",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "yolov8n.pt"
        )
        model = YOLO(pt).to(dev).half()
    return model

def detect_signal_color(roi):
    """
    Given a cropped traffic‑light ROI, return "red", "green", or None.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # red masks
    r1, r2 = np.array([0,70,50]),   np.array([10,255,255])
    r3, r4 = np.array([170,70,50]), np.array([180,255,255])
    red = int(cv2.countNonZero(cv2.inRange(hsv, r1, r2))
            + cv2.countNonZero(cv2.inRange(hsv, r3, r4)))
    # green mask
    g1, g2 = np.array([40,40,40]), np.array([90,255,255])
    green = int(cv2.countNonZero(cv2.inRange(hsv, g1, g2)))
    if green > red and green > 50:
        return "green"
    if red > green and red > 50:
        return "red"
    return None
