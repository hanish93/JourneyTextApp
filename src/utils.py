import os
import cv2
import urllib.request
import torch
import easyocr
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# ─── STATIC CONFIG ───────────────────────────────────────────────────────
STATIC_YOLO = {
    "traffic light", "stop sign", "street sign", "traffic sign",
    "bench", "fire hydrant", "parking meter", "clock", "potted plant",
}
# Words to strip out of BLIP captions
DYNAMIC = {"car", "person", "truck", "bus", "motorcycle", "bicycle", "dog"}

# ─── FRAME EXTRACTION ────────────────────────────────────────────────────
def extract_frames(path, fps=1):
    """Yield one frame per second from a video, or all .jpg in a folder."""
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

# ─── MOTION DETECTION ───────────────────────────────────────────────────
def detect_event_for_frame(prev_gray, cur_gray, dx_thresh=3.0, stop_thresh=0.2):
    """
    Optical‐flow motion. Returns one of:
      'drive', 'stop', 'turn_left', 'turn_right'
    Uses a higher dx_thresh and same stop_thresh.
    """
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
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

# ─── DOWNLOAD HELPER ────────────────────────────────────────────────────
def fetch(name, d, url, fname):
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, fname)
    if url and not os.path.exists(dst):
        urllib.request.urlretrieve(url, dst)
    return dst

# ─── YOLO + OCR FOR LANDMARKS ───────────────────────────────────────────
def get_landmark_models(device):
    yolo_pt = fetch(
        "yolov8n", "models",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8n.pt"
    )
    obj = YOLO(yolo_pt).to(device).half()
    sign_pt = "models/yolov8_signs.pt"
    sign = YOLO(sign_pt).to(device).half() if os.path.exists(sign_pt) else None
    ocr = easyocr.Reader(["en","it"], gpu=device.startswith("cuda"))
    return (obj, sign), ocr

def detect_landmarks_for_frame(img, model, ocr, conf=0.25):
    r = model(img, conf=conf, verbose=False)[0]
    if not r.boxes:
        return "none", ""
    labels, texts = [], []
    for b in r.boxes:
        cls = model.model.names[int(b.cls[0])]
        if cls not in STATIC_YOLO:
            continue
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        crop = img[y1:y2, x1:x2]
        t = " ".join(ocr.readtext(crop, detail=0))
        if t:
            texts.append(t)
            labels.append(f"{cls}[{t}]")
        else:
            labels.append(cls)
    return ", ".join(labels), " ".join(texts)

# ─── PLACES365 SCENE CLASSIFIER ─────────────────────────────────────────
def get_scene_model(device):
    from torchvision import models
    ck = fetch(
        "places365", "models",
        "http://places2.csail.mit.edu/models_places365/"
        "resnet18_places365.pth.tar",
        "resnet18_places365.pth.tar"
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
        p = torch.nn.functional.softmax(model(inp), 1)
    return classes[int(p.argmax())]

# ─── BLIP CAPTIONING ────────────────────────────────────────────────────
def get_caption_models(device):
    repo = "Salesforce/blip-image-captioning-base"
    proc = BlipProcessor.from_pretrained(repo)
    mod  = BlipForConditionalGeneration.from_pretrained(repo).to(device)
    return proc, mod

def generate_caption_for_frame(img, proc, mod, lm):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if max(pil.size) > 512:
        pil.thumbnail((512,512), Image.LANCZOS)
    ins = proc(images=pil, text=f"Scene contains: {lm}.", return_tensors="pt")
    ins = ins.to(mod.device)
    with torch.no_grad():
        ids = mod.generate(**ins, max_new_tokens=30)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ─── PURE‑PYTHON JOURNEY SUMMARY ────────────────────────────────────────
def summarise_journey(events, lm, cap, scn, ocr):
    return [
        {
            "step": i+1,
            "event": events[i],
            "scene": scn[i],
            "description": f"{cap[i]}. Landmark: {lm[i]}. OCR: '{ocr[i]}'"
        }
        for i in range(len(events))
    ]

def generate_long_summary(events, *args, **kwargs):
    """
    Build one deterministic first‑person sentence from your event list.
    """
    # 1) drop noise
    sigs = [e for e in events if e not in ("drive","stop")]

    # 2) collapse duplicates
    clean = []
    for e in sigs:
        if not clean or clean[-1] != e:
            clean.append(e)

    # 3) start
    if "the signal turned green" in clean:
        idx = clean.index("the signal turned green")
        summary = "I drove straight after the light turned green"
        rem = clean[idx+1:]
    else:
        summary = "I drove straight"
        rem = clean

    # 4) walk
    i = 0
    while i < len(rem):
        e = rem[i]
        if e.startswith("passed "):
            shops = []
            while i < len(rem) and rem[i].startswith("passed "):
                shops.append(rem[i][len("passed "):])
                i += 1
            if len(shops)==1:
                summary += f" and passed {shops[0]}"
            else:
                summary += " and passed " + ", ".join(shops[:-1]) + f" and {shops[-1]}"
        elif e == "turn_right":
            summary += " and took a slight right"
            while i < len(rem) and rem[i]=="turn_right":
                i += 1
        elif e == "turn_left":
            summary += " and turned left"
            while i < len(rem) and rem[i]=="turn_left":
                i += 1
        else:
            i += 1

    summary += " and continued straight."
    return summary[0].upper() + summary[1:]
