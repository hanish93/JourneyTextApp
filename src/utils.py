# utils.py  — lightweight video‑to‑diary helpers
import os, cv2, urllib.request, torch, easyocr, numpy as np
from itertools import islice
from PIL import Image
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
)

# ─────────── static sets ────────────────────────────────────────────────
STATIC_YOLO = {
    "traffic light","stop sign","street sign","traffic sign",
    "bench","fire hydrant","parking meter","clock","potted plant",
}
DYNAMIC_WORDS = {
    "car","person","truck","bus","motorcycle","bicycle","dog"
}

salient = lambda t: (t.split() and (len(t.split())>=2 or t[0].isupper()))
kind_of = lambda t: ("shop" if any(k in t.lower()
                    for k in["shop","store","mart","market","express"])
                    else "building" if any(k in t.lower()
                    for k in["registration","office","tower","hotel","plaza",
                             "building","center","carter"])
                    else "other")

# ─────────── small helpers ─────────────────────────────────────────────
def fetch(dir_, url, fname):
    os.makedirs(dir_, exist_ok=True)
    path = os.path.join(dir_, fname)
    if url and not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path

# ─────────── frame extractor (1 fps) ───────────────────────────────────
def extract_frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30
    step   = max(1, round(native / fps))
    idx, ok, frame = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield frame
        ok, frame = cap.read(); idx += 1
    cap.release()

# ─────────── basic motion label ────────────────────────────────────────
def detect_event(prev, cur, dx_turn=1.5, flow_stop=0.20):
    if prev is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None, .5,3,15,3,5,1.2,0)
    dx   = flow[...,0].mean(); mag = np.linalg.norm(flow, axis=2).mean()
    if mag < flow_stop:   return "stop"
    if dx  > dx_turn:     return "turn_right"
    if dx  < -dx_turn:    return "turn_left"
    return "drive"

# ─────────── detectors & OCR ───────────────────────────────────────────
def load_detectors(device):
    obj = YOLO(fetch("models",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8n.pt")).to(device).half()
    ocr = easyocr.Reader(["en"], gpu=device.startswith("cuda"))
    return obj, ocr

def get_landmarks(frame, model, ocr, conf=0.25):
    res = model(frame, verbose=False, conf=conf)[0]
    names = []
    for b in res.boxes:
        cls = model.model.names[int(b.cls[0])]
        if cls not in STATIC_YOLO: continue
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        txt = " ".join(ocr.readtext(frame[y1:y2, x1:x2], detail=0))
        if txt:
            names.append(txt)
    return names

# ─────────── caption model ─────────────────────────────────────────────
def load_captioner(device):
    repo = "Salesforce/blip-image-captioning-base"
    return (BlipProcessor.from_pretrained(repo),
            BlipForConditionalGeneration.from_pretrained(repo).to(device))

def caption_frame(frame, proc, mod, hint=""):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if max(img.size) > 512:
        img.thumbnail((512,512), Image.Resampling.LANCZOS)
    ins = proc(images=img, text=hint, return_tensors="pt").to(mod.device)
    with torch.no_grad():
        gen = mod.generate(**ins, max_new_tokens=30)
    return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

# ─────────── CPU‑side summariser (BART) ────────────────────────────────
tok_sum = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
mod_sum = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").cpu()

def summarise_journey(captions, whitelist, max_caps=40):
    captions = list(islice(captions, max_caps))       # keep ≤40 to stay <1 024 tok
    guard = ("Use only these place names: "+", ".join(whitelist)+".\n") if whitelist else \
            "No place names were detected; do NOT invent any.\n"
    prompt = (
        "Write 3‑4 plain sentences in first person describing the drive. "
        "Do NOT add headings. Ignore people and vehicles.\n"
        + guard + "\n"
        + "\n".join("- "+c for c in captions) + "\n\nSummary:"
    )
    inputs = tok_sum(prompt, return_tensors="pt")      # stays on CPU
    with torch.no_grad():
        ids = mod_sum.generate(**inputs,
                               max_new_tokens=120,
                               do_sample=False)
    return tok_sum.decode(ids[0], skip_special_tokens=True).split("Summary:")[-1].strip()
