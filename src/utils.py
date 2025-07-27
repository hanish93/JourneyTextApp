# src/utils.py

import os
import cv2
import urllib.request
import numpy as np
from ultralytics import YOLO
import easyocr

def fetch(dst_dir: str, url: str, fname: str) -> str:
    """Download yolov8n.pt (or your custom weights) if needed."""
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, fname)
    if url and not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path

def frames(video_path, fps=1):
    """Yield one frame per second (or given fps) from a .mp4."""
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

def load_det(dev, custom_model_path=None):
    """
    Load YOLOv8 (or your custom .pt) plus EasyOCR.
    Returns (yolo_model, ocr_reader).
    """
    if custom_model_path:
        model = YOLO(custom_model_path).to(dev).half()
    else:
        pt = fetch(
            "models",
            "https://github.com/ultralytics/assets/"
            "releases/download/v0.0.0/yolov8n.pt",
            "yolov8n.pt",
        )
        model = YOLO(pt).to(dev).half()
    ocr = easyocr.Reader(["en"], gpu=dev.startswith("cuda"))
    return model, ocr

def detect_signal_color(img, yolo_model, conf=0.25):
    """
    Return 'red' or 'green' if a traffic‑light box shows mostly that color.
    Otherwise None.
    """
    res = yolo_model(img, conf=conf, verbose=False)[0]
    for b in res.boxes:
        cls = yolo_model.model.names[int(b.cls[0])]
        if cls == "traffic light":
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # red mask
            r1, r2 = np.array([0,70,50]), np.array([10,255,255])
            r3, r4 = np.array([170,70,50]), np.array([180,255,255])
            red = cv2.countNonZero(cv2.inRange(hsv, r1, r2)) + \
                  cv2.countNonZero(cv2.inRange(hsv, r3, r4))
            # green mask
            g1, g2 = np.array([40,40,40]), np.array([90,255,255])
            green = cv2.countNonZero(cv2.inRange(hsv, g1, g2))
            if green > red and green > 50:
                return "green"
            if red > green and red > 50:
                return "red"
    return None

def signage_names(img, ocr_reader, conf=0.4):
    """
    OCR the full frame for any text ≥3 chars long.
    """
    raw = ocr_reader.readtext(img, detail=1)
    seen = []
    for _, txt, prob in raw:
        t = txt.strip()
        if prob >= conf and len(t) >= 3 and any(c.isalpha() for c in t):
            if t not in seen:
                seen.append(t)
    return seen

def landmarks(img, yolo_model, ocr_reader, conf=0.25):
    """
    OCR any built‑in KEEP classes + full‑frame text.
    """
    KEEP = {"street sign","traffic sign","stop sign","clock","bench","potted plant"}
    res = yolo_model(img, conf=conf, verbose=False)[0]
    names = []
    # first: OCR inside KEEP boxes
    for b in res.boxes:
        cls = yolo_model.model.names[int(b.cls[0])]
        if cls in KEEP:
            x1,y1,x2,y2 = map(int,b.xyxy[0])
            snippet = img[y1:y2, x1:x2]
            text = " ".join(ocr_reader.readtext(snippet, detail=0))
            if text and text not in names:
                names.append(text)
    # next: full-frame
    for txt in signage_names(img, ocr_reader, conf):
        if txt not in names:
            names.append(txt)
    return names
