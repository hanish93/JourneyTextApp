# src/utils.py

import os
import cv2
import urllib.request
import numpy as np
import torch
import easyocr
from ultralytics import YOLO
from transformers import pipeline

def fetch(dst_dir: str, url: str, fname: str) -> str:
    """
    Ensure dst_dir/fname exists; download from url if missing.
    Returns the local path.
    """
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, fname)
    if url and not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path

def frames(video_path: str, fps: int = 1):
    """
    Yield frames from a video at ~fps frames per second.
    """
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(original_fps / fps))
    idx = 0
    success, frame = cap.read()
    while success:
        if idx % step == 0:
            yield frame
        success, frame = cap.read()
        idx += 1
    cap.release()

def move(prev_gray, curr_gray, dx: float = 1.5, stop_thr: float = 0.2) -> str:
    """
    Compute a simple motion verb (drive/stop/turn_left/turn_right)
    using Farneback optical flow on grayscale frames.
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
    Load a YOLOv8 model onto device `dev` ("cpu" or "cuda").
    If model_path is provided, use that; otherwise download yolov8n.pt.
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

def detect_signal_color(roi) -> str:
    """
    Given a BGR crop of a traffic light, return "red", "green", or None.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # define red masks (two ranges)
    lower1, upper1 = np.array([0,70,50]),   np.array([10,255,255])
    lower2, upper2 = np.array([170,70,50]), np.array([180,255,255])
    red_mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    red_count = int(cv2.countNonZero(red_mask))
    # define green mask
    lower_g, upper_g = np.array([40,40,40]), np.array([90,255,255])
    green_mask = cv2.inRange(hsv, lower_g, upper_g)
    green_count = int(cv2.countNonZero(green_mask))
    if green_count > red_count and green_count > 50:
        return "green"
    if red_count > green_count and red_count > 50:
        return "red"
    return None

def load_blip2(dev: str):
    """
    Load BLIP‑2 as an image-to-text pipeline.
    Use device=0 for GPU, -1 for CPU.
    """
    device = 0 if dev.startswith("cuda") else -1
    return pipeline(
        "image-to-text",
        model="Salesforce/blip2-opt-2.7b",
        device=device,
        tokenizer_kwargs={"padding": True},
        image_processor_kwargs={"size": (384, 384)},
    )

def load_ocr():
    """
    Initialize and return an EasyOCR reader (GPU if available).
    """
    use_gpu = torch.cuda.is_available()
    return easyocr.Reader(["en"], gpu=use_gpu)

def ocr_signs(img, ocr_reader, conf: float = 0.3) -> list[str]:
    """
    Perform full-frame OCR, return deduped list of strings >=3 chars
    with confidence >= conf.
    """
    raw = ocr_reader.readtext(img, detail=1)
    seen = []
    for _, txt, prob in raw:
        t = txt.strip()
        if prob >= conf and len(t) >= 3 and any(c.isalpha() for c in t):
            if t not in seen:
                seen.append(t)
    return seen
