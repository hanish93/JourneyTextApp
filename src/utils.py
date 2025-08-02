# JourneyTextApp/src/utils.py

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import easyocr

# ──────────────────── CONFIG ────────────────────
YOLO_LIGHT_WEIGHTS   = "models/traffic_lights.pt"
FLOW_TURN_WEIGHTS    = "models/turn_classifier.pt"
OCR_LANGS            = ["en"]

# ──────────────────── HELPERS ────────────────────

def extract_frames(src, fps=1):
    if os.path.isdir(src):
        for fn in sorted(os.listdir(src)):
            if fn.lower().endswith(".jpg"):
                img = cv2.imread(os.path.join(src, fn))
                if img is not None:
                    yield img
        return
    cap = cv2.VideoCapture(src)
    nat = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(nat / fps)))
    idx, ok, frame = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield frame
        ok, frame = cap.read()
        idx += 1
    cap.release()

# ──────────────────── TRAFFIC LIGHT ────────────────────

_light_model = None
def load_light_model(device="cpu"):
    global _light_model
    if _light_model is None:
        _light_model = YOLO(YOLO_LIGHT_WEIGHTS).to(device).half()
    return _light_model

def detect_light_state(frame, model, conf=0.25):
    res = model(frame, conf=conf, verbose=False)[0]
    bbs = [
        tuple(map(int, box.xyxy[0].cpu().numpy()))
        for box in res.boxes
        if model.model.names[int(box.cls[0])] == "traffic light"
    ]
    if not bbs:
        return None
    x1,y1,x2,y2 = max(bbs, key=lambda bb:(bb[2]-bb[0])*(bb[3]-bb[1]))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv,(0,80,80),(10,255,255))
    red2 = cv2.inRange(hsv,(160,80,80),(180,255,255))
    green= cv2.inRange(hsv,(40,80,80),(85,255,255))
    rc, gc = int(cv2.countNonZero(red1|red2)), int(cv2.countNonZero(green))
    if max(rc,gc) < 100:
        return None
    return "red" if rc>gc else "green"

# ──────────────────── TURN DETECTION ────────────────────

_turn_model = None
def load_turn_model(device="cpu"):
    global _turn_model
    if _turn_model is None:
        _turn_model = torch.load(FLOW_TURN_WEIGHTS, map_location=device)
        _turn_model.eval()
    return _turn_model

def detect_turn(prev_gray, cur_gray):
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        0.5,3,15,3,5,1.2,0)
    dx  = float(flow[...,0].mean())
    mag = float(np.linalg.norm(flow,axis=2).mean())
    if mag < 0.3:     return "stop"
    if dx > 2.0:      return "turn_right"
    if dx < -2.0:     return "turn_left"
    return "drive"

# ──────────────────── SIGN/OCR ────────────────────

_ocr_reader = None
def load_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(OCR_LANGS, gpu=torch.cuda.is_available())
    return _ocr_reader

def detect_signs(frame, reader, conf=0.5):
    h,w = frame.shape[:2]
    results = reader.readtext(frame, detail=1)
    out=[]
    for bbox, txt, score in results:
        if score < conf: continue
        xs = [pt[0] for pt in bbox]; ys = [pt[1] for pt in bbox]
        cx = (min(xs)+max(xs))/2 / w
        out.append((txt, cx))
    return out

# ──────────────────── DEBOUNCE ────────────────────

def debounce_list(lst, window=3, min_count=2):
    n=len(lst); out=lst.copy()
    for i,e in enumerate(lst):
        if e in ("turn_left","turn_right"):
            cnt = sum(1 for j in range(max(0,i-window), min(n,i+window+1))
                      if lst[j]==e)
            if cnt<min_count:
                out[i]="drive"
    return out

def debounce_signals(sigs, window=3):
    n=len(sigs); out=[None]*n
    for i,s in enumerate(sigs):
        if s in ("red","green"):
            cnt = sum(1 for j in range(max(0,i-window), min(n,i+window+1))
                      if sigs[j]==s)
            if cnt>=2:
                out[i]=s
    return out

# ──────────────────── NARRATIVE ────────────────────

def build_narrative(evts, sigs, signs):
    parts=[]; last_sig=None

    def add(p): parts.append(p)

    for i,(e,s,sg) in enumerate(zip(evts,sigs,signs)):
        # start at red
        if i==0 and s=="red":
            add("turned right from the signal")
            last_sig="red"
            continue

        # sign OCR
        if sg:
            txt,cx = sg[0]
            side = "left-hand side" if cx<0.5 else "right-hand side"
            add(f"a shop was visible on the {side}")

        # turn
        if e=="turn_right": add("then turned right")
        if e=="turn_left":  add("then turned left")

        # green after red
        if last_sig=="red" and s=="green":
            add("the vehicle proceeded through another green signal")
            last_sig="green"

        # continued drive
        if e=="drive" and not parts[-1].startswith("then"):
            add("continued straight for a while")

    sent = parts[0].capitalize()
    for p in parts[1:]:
        sent += " and " + p
    sent += "."
    return sent
