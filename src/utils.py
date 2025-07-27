import os
import cv2
import urllib.request
import numpy as np
from ultralytics import YOLO
import easyocr

def fetch(dst_dir: str, url: str, fname: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, fname)
    if url and not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path

def load_det(dev, custom_model_path=None):
    """
    Load YOLOv8 (or your custom .pt) plus EasyOCR.
    """
    if custom_model_path:
        y = YOLO(custom_model_path).to(dev).half()
    else:
        pt = fetch("models",
                   "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                   "yolov8n.pt")
        y = YOLO(pt).to(dev).half()
    ocr = easyocr.Reader(["en"], gpu=dev.startswith("cuda"))
    return y, ocr

def detect_signal_color(img, yolo_model, conf=0.25):
    """
    Look for a 'traffic light' box and decide red vs green by HSV masking.
    """
    res = yolo_model(img, conf=conf, verbose=False)[0]
    for box in res.boxes:
        cls = yolo_model.model.names[int(box.cls[0])]
        if cls == "traffic light":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # red mask (two ranges)
            lower1, upper1 = np.array([0,70,50]), np.array([10,255,255])
            lower2, upper2 = np.array([170,70,50]), np.array([180,255,255])
            m1 = cv2.inRange(hsv, lower1, upper1)
            m2 = cv2.inRange(hsv, lower2, upper2)
            red_count = int(cv2.countNonZero(m1) + cv2.countNonZero(m2))
            # green mask
            lowerg, upperg = np.array([40,40,40]), np.array([90,255,255])
            mg = cv2.inRange(hsv, lowerg, upperg)
            green_count = int(cv2.countNonZero(mg))
            if green_count > red_count and green_count > 50:
                return "green"
            if red_count > green_count and red_count > 50:
                return "red"
    return None

def signage_names(img, ocr_reader, conf=0.4):
    """
    OCR the full frame for any reasonable‑length alpha text.
    """
    raw = ocr_reader.readtext(img, detail=1)
    picks = []
    for _, text, prob in raw:
        t = text.strip()
        if prob >= conf and len(t)>=3 and any(c.isalpha() for c in t):
            picks.append(t)
    # dedupe, preserve order
    return list(dict.fromkeys(picks))

def landmarks(img, yolo_model, ocr_reader, conf=0.25):
    """
    1) OCR any KEEP classes (e.g. street signs) if you still want them
    2) OCR full frame signage as a fallback
    """
    KEEP = {"street sign","traffic sign","stop sign","clock","bench","potted plant"}
    r = yolo_model(img, conf=conf, verbose=False)[0]
    names = []
    for b in r.boxes:
        cls = yolo_model.model.names[int(b.cls[0])]
        if cls in KEEP:
            x1,y1,x2,y2 = map(int,b.xyxy[0])
            txt = " ".join(ocr_reader.readtext(img[y1:y2, x1:x2], detail=0))
            if txt:
                names.append(txt)
    # full-frame sign OCR as extra
    names += signage_names(img, ocr_reader, conf)
    return list(dict.fromkeys(names))
