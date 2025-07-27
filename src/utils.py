import os
import cv2
import urllib.request
import json
import torch
import easyocr
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
)

# ─── static configs ─────────────────────────────────────────────────────
STATIC_YOLO = {
    "traffic light", "stop sign", "street sign", "traffic sign",
    "bench", "fire hydrant", "parking meter", "clock", "potted plant",
}
DYNAMIC = {"car","person","truck","bus","motorcycle","bicycle","dog"}

# ─── helper to decide if OCR text is worth keeping ─────────────────────
def salient(txt):
    words = txt.split()
    return len(words) >= 2 or (words and words[0][0].isupper())

def kind_of(txt):
    l = txt.lower()
    if any(x in l for x in ["shop","store","express","mart","market"]):
        return "shop"
    if any(x in l for x in [
        "office","tower","building","center","plaza","hotel"
    ]):
        return "building"
    return "other"

# ─── extract a frame‑per‑second or read a folder of JPGs ───────────────
def extract_frames(path, fps=1):
    # directory of images?
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(".jpg"):
                img = cv2.imread(os.path.join(path, fn))
                if img is not None:
                    yield img
        return

    # otherwise treat as video
    cap = cv2.VideoCapture(path)
    nat = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(nat / fps))
    idx, ok, img = 0, *cap.read()
    print("[Frames] Starting extraction …")
    while ok:
        if idx % step == 0:
            yield img
        ok, img = cap.read()
        idx += 1
    cap.release()
    print("[Frames] Done.")

# ─── simple motion detection → drive/stop/turn ─────────────────────────
def detect_event_for_frame(prev, cur, dx=1.5, stop_thr=0.2):
    if prev is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(
        prev, cur, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    dxm = flow[...,0].mean()
    mag = np.linalg.norm(flow, axis=2).mean()
    if mag < stop_thr:
        return "stop"
    if dxm > dx:
        return "turn_right"
    if dxm < -dx:
        return "turn_left"
    return "drive"

# ─── download helper ───────────────────────────────────────────────────
def fetch(name, d, url, fname):
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, fname)
    if url and not os.path.exists(dst):
        urllib.request.urlretrieve(url, dst)
    return dst

# ─── YOLO + EasyOCR for landmarks ──────────────────────────────────────
def get_landmark_models(device):
    yolo_path = fetch(
        "yolov8n", "models",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8n.pt"
    )
    obj = YOLO(yolo_path).to(device).half()
    sign_path = "models/yolov8_signs.pt"
    sign = YOLO(sign_path).to(device).half() if os.path.exists(sign_path) else None
    ocr = easyocr.Reader(["en","it"], gpu=device.startswith("cuda"))
    return (obj, sign), ocr

def detect_landmarks_for_frame(img, model, ocr, conf=0.25):
    r = model(img, conf=conf, verbose=False)[0]
    if not r.boxes:
        return "none", ""
    labels, texts = [], []
    for box in r.boxes:
        cls = model.model.names[int(box.cls[0])]
        if cls not in STATIC_YOLO:
            continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        txt = " ".join(ocr.readtext(crop, detail=0))
        if salient(txt):
            labels.append(f"{cls}[{txt}]")
            texts.append(txt)
        else:
            labels.append(cls)
    return ", ".join(labels), " ".join(texts)

# ─── PLACES365 scene classifier ────────────────────────────────────────
def get_scene_model(device):
    import torchvision.models as models
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
    # categories:
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
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inp = tf(pil).unsqueeze(0).to(
        next(model.parameters()).device,
        dtype=next(model.parameters()).dtype
    )
    with torch.no_grad():
        p = torch.nn.functional.softmax(model(inp), 1)
    return classes[int(p.argmax())]

# ─── BLIP captioning ──────────────────────────────────────────────────
def get_caption_models(device):
    repo = "Salesforce/blip-image-captioning-base"
    proc = BlipProcessor.from_pretrained(repo)
    mod = BlipForConditionalGeneration.from_pretrained(repo).to(device)
    return proc, mod

def generate_caption_for_frame(img, proc, mod, landmarks):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if max(pil.size) > 512:
        pil.thumbnail((512,512), Image.LANCZOS)
    ins = proc(images=pil,
               text=f"Scene contains: {landmarks}.",
               return_tensors="pt").to(mod.device)
    with torch.no_grad():
        ids = mod.generate(**ins, max_new_tokens=30)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ─── Flan‑T5 long summary ─────────────────────────────────────────────
# src/utils.py

from transformers import pipeline as hf_pipeline
import torch

def generate_long_summary(events, *args, **kwargs):
    """
    Take the cleaned event list (one bullet per frame)
    and produce exactly one first‑person sentence summary.
    """
    # Load Flan‑T5‑Large on GPU if available
    device_map = "auto" if torch.cuda.is_available() else None
    summariser = hf_pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device_map=device_map,
    )

    # Build a tiny bullet list of just the events
    bullets = "\n".join(f"- {e}" for e in events)

    prompt = (
        "Write one concise first‑person sentence describing this drive, "
        "based only on these events:\n"
        f"{bullets}\n\nSummary:"
    )

    out = summariser(prompt, max_new_tokens=60, do_sample=False)[0]["generated_text"]
    return out.strip()


# ─── Table helper ─────────────────────────────────────────────────────
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
