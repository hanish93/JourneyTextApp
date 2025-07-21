# utils.py  — tiny video‑to‑diary helpers
import os, cv2, urllib.request, torch, easyocr, numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoModelForSeq2SeqLM, AutoTokenizer
)

# permanent objects we care about
STATIC_YOLO = {
    "traffic light","stop sign","street sign","traffic sign",
    "bench","fire hydrant","parking meter","clock","potted plant",
}

# ───────────────── download utility ────────────────────────────────────
def fetch(dst_dir: str, url: str, fname: str):
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, fname)
    if url and not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path

# ───────────────── frame extractor (1 fps) ─────────────────────────────
def extract_frames(video_path, fps: int = 1):
    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30
    step   = max(1, round(native / fps))
    idx, ok, img = 0, *cap.read()
    while ok:
        if idx % step == 0:      # keep 1‑per‑sec by default
            yield img
        ok, img = cap.read(); idx += 1
    cap.release()

# ───────────────── simple motion tag ───────────────────────────────────
def detect_event(prev, cur, dx_turn=1.5, flow_stop=0.20):
    if prev is None: return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None, .5,3,15,3,5,1.2,0)
    dx   = flow[...,0].mean();  mag = np.linalg.norm(flow,axis=2).mean()
    if mag < flow_stop: return "stop"
    if dx  > dx_turn : return "turn_right"
    if dx  < -dx_turn: return "turn_left"
    return "drive"

# ───────────────── detectors / OCR ─────────────────────────────────────
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
        x1,y1,x2,y2 = map(int,b.xyxy[0])
        txt = " ".join(ocr.readtext(frame[y1:y2,x1:x2], detail=0))
        if txt: names.append(txt)
    return names

# ───────────────── caption model ───────────────────────────────────────
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
        ids = mod.generate(**ins, max_new_tokens=30)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ───────────────── summariser (Bart‑CNN, no halluc.) ───────────────────
tok_sum = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
mod_sum = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn",
            device_map="auto"
          )

def summarise_journey(sentences, whitelist):
    guard = ("Use only these place names: "+
             ", ".join(whitelist)+".\n") if whitelist else \
            "Do not mention any place names.\n"
    prompt = ("Write 3‑4 plain sentences in first person describing the drive. "
              "Ignore people and vehicles. "
              "No headings.\n"+guard+"\n"
              + "\n".join("- "+s for s in sentences) + "\n\nSummary:")
    inp = tok_sum(prompt, return_tensors="pt").to(mod_sum.device)
    with torch.no_grad():
        ids = mod_sum.generate(**inp, max_new_tokens=120, do_sample=False)
    return tok_sum.decode(ids[0], skip_special_tokens=True).split("Summary:")[-1].strip()
