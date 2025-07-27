import os
import cv2
import urllib.request
import torch
import easyocr
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

STATIC_YOLO = {
    "traffic light", "stop sign", "street sign", "traffic sign",
    "bench", "fire hydrant", "parking meter", "clock", "potted plant",
}

def extract_frames(path, fps=1):
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(".jpg"):
                img = cv2.imread(os.path.join(path, fn))
                if img is not None:
                    yield img
        return
    cap = cv2.VideoCapture(path)
    nat = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(nat / fps))
    idx, ok, img = 0, *cap.read()
    while ok:
        if idx % step:
            pass
        else:
            yield img
        ok, img = cap.read()
        idx += 1
    cap.release()

def detect_event_for_frame(prev, cur, dx_thresh=3.0, stop_thresh=0.2):
    if prev is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev,cur,None,0.5,3,15,3,5,1.2,0)
    dxm = flow[...,0].mean()
    mag = np.linalg.norm(flow,axis=2).mean()
    if mag < stop_thresh:
        return "stop"
    if dxm > dx_thresh:
        return "turn_right"
    if dxm < -dx_thresh:
        return "turn_left"
    return "drive"

def debounce_lane_changes(events, window=3):
    out = list(events)
    n = len(events)
    for i,e in enumerate(events):
        if e in ("turn_left","turn_right"):
            cnt = sum(
                1 for j in range(max(0,i-window), min(n,i+window+1))
                if events[j]==e
            )
            if cnt < 2:
                out[i] = "drive"
    return out

def detect_signal_color(frame, yolo_model, conf=0.25):
    r = yolo_model(frame, conf=conf, verbose=False)[0]
    boxes = []
    for b in r.boxes:
        cls = yolo_model.model.names[int(b.cls[0])]
        if cls == "traffic light":
            x1,y1,x2,y2 = map(int,b.xyxy[0].cpu().numpy())
            boxes.append((x1,y1,x2,y2))
    if not boxes:
        return None
    areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes]
    x1,y1,x2,y2 = boxes[int(np.argmax(areas))]
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv,(0,50,50),(10,255,255))
    m2 = cv2.inRange(hsv,(160,50,50),(180,255,255))
    red = cv2.bitwise_or(m1,m2)
    green = cv2.inRange(hsv,(40,50,50),(85,255,255))
    rc = int(red.sum()/255); gc = int(green.sum()/255)
    if max(rc,gc) < 50:
        return None
    return "red" if rc>gc else "green"

def debounce_signals(states, window=2):
    out = list(states)
    n = len(states)
    for i,s in enumerate(states):
        if s in ("red","green"):
            cnt = sum(
                1 for j in range(max(0,i-window), min(n,i+window+1))
                if states[j]==s
            )
            if cnt < 2:
                out[i] = None
    return out

def fetch(name,d,url,fname):
    os.makedirs(d,exist_ok=True)
    dst = os.path.join(d,fname)
    if url and not os.path.exists(dst):
        urllib.request.urlretrieve(url,dst)
    return dst

def get_landmark_models(device):
    pt = fetch("yolov8n","models",
               "https://github.com/ultralytics/assets/releases/"
               "download/v0.0.0/yolov8n.pt","yolov8n.pt")
    yolo = YOLO(pt).to(device).half()
    ocr = easyocr.Reader(["en"],gpu=device.startswith("cuda"))
    return yolo,ocr

def detect_landmarks_for_frame(img,yolo,ocr,conf=0.25):
    r = yolo(img,conf=conf,verbose=False)[0]
    labels,texts = [],[]
    for b in r.boxes:
        cls = yolo.model.names[int(b.cls[0])]
        if cls not in STATIC_YOLO:
            continue
        x1,y1,x2,y2 = map(int,b.xyxy[0].cpu().numpy())
        crop = img[y1:y2,x1:x2]
        t = " ".join(ocr.readtext(crop,detail=0))
        labels.append(f"{cls}[{t}]" if t else cls)
        if t:
            texts.append(t)
    return ", ".join(labels) or "none"," ".join(texts)

def get_scene_model(device):
    from torchvision import models
    ck = fetch("places365","models",
               "http://places2.csail.mit.edu/models_places365/"
               "resnet18_places365.pth.tar","resnet18_places365.pth.tar")
    m = models.resnet18(num_classes=365)
    sd = torch.load(ck,map_location="cpu")["state_dict"]
    m.load_state_dict({k.replace("module.",""):v for k,v in sd.items()})
    m.to(device).half().eval()
    cats="categories_places365.txt"
    if not os.path.exists(cats):
        os.system("wget -q https://raw.githubusercontent.com/csailvision/"
                  "places365/master/categories_places365.txt")
    classes = [l.strip().split()[0][3:] for l in open(cats)]
    return m,classes

def classify_scene_for_frame(img,model,classes):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((256,256)),transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    pil = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    inp = tf(pil).unsqueeze(0).to(
        next(model.parameters()).device,
        dtype=next(model.parameters()).dtype
    )
    with torch.no_grad():
        p = torch.nn.functional.softmax(model(inp),1)
    return classes[int(p.argmax())]

def get_caption_models(device):
    repo="Salesforce/blip-image-captioning-base"
    proc=BlipProcessor.from_pretrained(repo)
    mod=BlipForConditionalGeneration.from_pretrained(repo).to(device)
    return proc,mod

def generate_caption_for_frame(img,proc,mod,lm):
    pil = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    if max(pil.size)>512:
        pil.thumbnail((512,512),Image.LANCZOS)
    ins = proc(images=pil,text=f"Scene contains: {lm}.",
               return_tensors="pt").to(mod.device)
    with torch.no_grad():
        ids = mod.generate(**ins,max_new_tokens=30)
    return proc.batch_decode(ids,skip_special_tokens=True)[0].strip()

def summarise_journey(events,lm,cap,scn,ocr,signals):
    rows=[]
    for i,e in enumerate(events):
        rows.append({
            "step":i+1,
            "event":e,
            "signal":signals[i] or "",
            "scene":scn[i],
            "description":f"{cap[i]}. Landmark: {lm[i]}. OCR: '{ocr[i]}'"
        })
    return rows

def generate_long_summary(events, signals, *a, **k):
    """
    events:  list of "drive", "turn_left", "passed <X>", etc.
    signals: parallel list of None, "red" or "green"
    """
    parts = []
    stopped = False

    for ev, sig in zip(events, signals):
        # 1) signal events
        if sig == "red" and not stopped:
            parts.append("stopped at the red light")
            stopped = True
            continue
        if sig == "green" and stopped:
            parts.append("when it turned green, I drove on")
            stopped = False
            continue

        # 2) motion / passes
        if ev == "drive":
            # skip raw drives when we have nothing else
            continue
        if ev.startswith("passed "):
            loc = ev[len("passed "):]
            parts.append(f"passed {loc}")
            continue
        if ev == "turn_left":
            parts.append("turned left")
            continue
        if ev == "turn_right":
            parts.append("took a slight right")
            continue
        if ev == "stop":
            # optional: treat as a brief pause
            parts.append("paused briefly")
            continue

    # ensure it starts with driving
    if parts and not parts[0].startswith("stopped"):
        parts.insert(0, "drove straight")

    # join
    journey = parts[0].capitalize()
    for p in parts[1:]:
        journey += " and " + p

    # finish
    if not journey.endswith("."):
        journey += "."

    return journey

