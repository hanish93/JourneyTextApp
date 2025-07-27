import os
import cv2
import urllib.request
import torch
import easyocr
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

# ─── STATIC CONFIG ────────────────────────────────────────────────────────
STATIC_YOLO = {
    "traffic light", "stop sign", "street sign", "traffic sign",
    "bench", "fire hydrant", "parking meter", "clock", "potted plant",
}

# ─── FRAME EXTRACTION ─────────────────────────────────────────────────────
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
        if idx % step == 0:
            yield img
        ok, img = cap.read()
        idx += 1
    cap.release()

# ─── MOTION DETECTION (LANE‑CHANGE) ───────────────────────────────────────
def detect_event_for_frame(prev, cur, dx_thresh=3.0, stop_thresh=0.2):
    if prev is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(
        prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    dxm = flow[...,0].mean()
    mag = np.linalg.norm(flow, axis=2).mean()
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
    for i in range(n):
        e = events[i]
        if e in ("turn_left","turn_right"):
            cnt = sum(
                1 for j in range(max(0,i-window), min(n,i+window+1))
                if events[j]==e
            )
            if cnt < 2:
                out[i] = "drive"
    return out

# ─── SIGNAL‑COLOR DETECTION VIA HSV ───────────────────────────────────────
def detect_signal_color(frame, yolo_model, conf=0.25):
    """
    Finds any 'traffic light' box, takes the largest one,
    converts to HSV, and compares red vs green pixel counts.
    Returns 'red', 'green', or None.
    """
    # 1) detect all traffic‑light bboxes
    r = yolo_model(frame, conf=conf, verbose=False)[0]
    tl_boxes = [
        box.xyxy[0].cpu().numpy().astype(int)
        for box in r.boxes
        if yolo_model.model.names[int(box.cls[0])] == "traffic light"
    ]
    if not tl_boxes:
        return None

    # 2) pick the largest box by area
    areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in tl_boxes]
    idx = int(np.argmax(areas))
    x1,y1,x2,y2 = tl_boxes[idx]

    # 3) crop & convert
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 4) red mask (two hue ranges) and green mask
    lower1, upper1 = np.array([0,50,50]), np.array([10,255,255])
    lower2, upper2 = np.array([160,50,50]), np.array([180,255,255])
    red1 = cv2.inRange(hsv, lower1, upper1)
    red2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(red1, red2)
    green_mask = cv2.inRange(hsv, np.array([40,50,50]), np.array([85,255,255]))

    rcount = int(red_mask.sum()/255)
    gcount = int(green_mask.sum()/255)
    if max(rcount,gcount) < 50:   # too few pixels => uncertain
        return None
    return "red" if rcount>gcount else "green"

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

# ─── LANDMARK + OCR ───────────────────────────────────────────────────────
def fetch(name, d, url, fname):
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, fname)
    if url and not os.path.exists(dst):
        urllib.request.urlretrieve(url, dst)
    return dst

def get_landmark_models(device):
    pt = fetch(
        "yolov8n","models",
        "https://github.com/ultralytics/assets/releases/"
        "download/v0.0.0/yolov8n.pt","yolov8n.pt"
    )
    yolo = YOLO(pt).to(device).half()
    ocr = easyocr.Reader(["en"], gpu=device.startswith("cuda"))
    return yolo, ocr

def detect_landmarks_for_frame(img, yolo, ocr, conf=0.25):
    r = yolo(img, conf=conf, verbose=False)[0]
    labels, texts = [], []
    for b in r.boxes:
        cls = yolo.model.names[int(b.cls[0])]
        if cls not in STATIC_YOLO:
            continue
        x1,y1,x2,y2 = map(int,b.xyxy[0])
        crop = img[y1:y2, x1:x2]
        t = " ".join(ocr.readtext(crop, detail=0))
        labels.append(f"{cls}[{t}]" if t else cls)
        if t: texts.append(t)
    return ", ".join(labels) or "none", " ".join(texts)

# ─── SCENE CLASSIFIER ────────────────────────────────────────────────────
def get_scene_model(device):
    import torchvision.models as models
    ck = fetch(
        "places365","models",
        "http://places2.csail.mit.edu/models_places365/"
        "resnet18_places365.pth.tar","resnet18_places365.pth.tar"
    )
    m = models.resnet18(num_classes=365)
    sd = torch.load(ck, map_location="cpu")["state_dict"]
    m.load_state_dict({k.replace("module.",""):v for k,v in sd.items()})
    m.to(device).half().eval()
    cats = "categories_places365.txt"
    if not os.path.exists(cats):
        os.system(
            "wget -q https://raw.githubusercontent.com/csailvision/"
            "places365/master/categories_places365.txt"
        )
    classes = [l.strip().split()[0][3:] for l in open(cats)]
    return m, classes

def classify_scene_for_frame(img, model, classes):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inp = tf(pil).unsqueeze(0).to(
        next(model.parameters()).device,
        dtype=next(model.parameters()).dtype
    )
    with torch.no_grad():
        p = torch.nn.functional.softmax(model(inp),1)
    return classes[int(p.argmax())]

# ─── BLIP CAPTIONING ─────────────────────────────────────────────────────
def get_caption_models(device):
    repo = "Salesforce/blip-image-captioning-base"
    proc = BlipProcessor.from_pretrained(repo)
    mod  = BlipForConditionalGeneration.from_pretrained(repo).to(device)
    return proc, mod

def generate_caption_for_frame(img, proc, mod, lm):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if max(pil.size)>512:
        pil.thumbnail((512,512), Image.LANCZOS)
    ins = proc(images=pil, text=f"Scene contains: {lm}.",
               return_tensors="pt").to(mod.device)
    with torch.no_grad():
        out = mod.generate(**ins, max_new_tokens=30)
    return proc.batch_decode(out, skip_special_tokens=True)[0].strip()

# ─── SUMMARY BUILDERS ───────────────────────────────────────────────────
def summarise_journey(events, lm, cap, scn, ocr, sigs):
    """
    Now includes the per‑frame signal color in the output.
    """
    rows=[]
    for i, e in enumerate(events):
        sig = sigs[i] or ""
        rows.append({
            "step": i+1,
            "event": e,
            "scene": scn[i],
            "signal": sig,
            "description": f"{cap[i]}. Landmark: {lm[i]}. OCR: '{ocr[i]}'"
        })
    return rows
