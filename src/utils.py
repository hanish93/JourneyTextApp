# utils.py  — 2025‑07‑20
import os, cv2, urllib.request, json, shutil, re
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import easyocr
from ultralytics import YOLO
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BitsAndBytesConfig,
)

# ───────────────────────────────── frame extraction ──────────────────────────
def extract_frames(video_path, fps: int = 1):
    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30
    step   = max(1, round(native / fps))
    idx, ok, img = 0, *cap.read()
    print("[Frames] Starting extraction …")
    while ok:
        if idx % step == 0:
            yield img
        ok, img = cap.read()
        idx += 1
    cap.release()
    print("[Frames] Done.")

# ───────────────────────────────── event detector ────────────────────────────
def detect_event_for_frame(prev_gray, current_gray, stride=1, dx_turn=1.5, flow_stop=0.20) -> str:
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, .5,3,15,3,5,1.2,0)
    dx   = flow[..., 0].mean()
    mag  = np.linalg.norm(flow, axis=2).mean()

    if   mag < flow_stop: evt = "stop"
    elif dx >  dx_turn :  evt = "turn_right"
    elif dx < -dx_turn :  evt = "turn_left"
    else:                evt = "drive"
    return evt

# ───────────────────────────── helper: download / cache ──────────────────────
def get_or_download_model(name, local_dir, url=None, hf_repo=None, cfg="config.json"):
    if hf_repo:
        return hf_repo
    os.makedirs(local_dir, exist_ok=True)
    if not os.path.exists(os.path.join(local_dir, cfg)) and url:
        print(f"[Model] Downloading {name} …")
        urllib.request.urlretrieve(url, os.path.join(local_dir, os.path.basename(url)))
    return local_dir

# ───────────────────────────── landmark detector ─────────────────────────────
def get_landmark_models(device):
    logo_w = "models/yolov8_logo.pt"
    coco_w = "models/yolov8n.pt"
    if not os.path.exists(logo_w):
        get_or_download_model(
            "yolov8n", os.path.dirname(coco_w),
            url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            cfg="yolov8n.pt"
        )
    model = YOLO(logo_w if os.path.exists(logo_w) else coco_w).to(device)
    ocr   = easyocr.Reader(["en"], gpu=device.startswith("cuda"))
    return model, ocr

def detect_landmarks_for_frame(frame, model, ocr, conf=0.25):
    r = model(frame, verbose=False, conf=conf)[0]
    if len(r.boxes) == 0:
        return "none"
    boxes = []
    for b in r.boxes:
        cls = model.model.names[int(b.cls[0])]
        conf = float(b.conf[0])
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        txt  = " ".join(t[1] for t in ocr.readtext(crop))
        boxes.append(f"{cls} ({conf:.2f}) [{txt}]" if txt else f"{cls} ({conf:.2f})")
    return ", ".join(boxes)

# ──────────────────────────── caption generator (BLIP‑2) ─────────────────────
def get_caption_models(device):
    repo = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(repo)
    model = BlipForConditionalGeneration.from_pretrained(repo).to(device)
    return processor, model

def generate_caption_for_frame(frame, processor, model, landmark=None):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if max(img.size) > 512:
        res = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        img.thumbnail((512, 512), res)
    hint = f"Scene contains: {landmark}." if landmark else ""
    ins  = processor(images=img, text=hint, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**ins, max_new_tokens=30)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ───────────────────────── long‑form narrative generator ─────────────────────
def generate_long_summary(events, landmarks, captions):
    from transformers import pipeline
    repo = "sshleifer/distilbart-cnn-12-6"
    summarizer = pipeline("summarization", model=repo)

    ctx_max, out_tok = 1024, 256
    input_text = " ".join([f"{e}: {c}. Landmark: {l}" for e, c, l in zip(events, captions, landmarks)])

    # Summarize the text
    summary = summarizer(input_text, max_length=out_tok, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# ──────────────────────────── JSON‑friendly helpers ──────────────────────────
def summarise_journey(events, landmarks, captions):
    return [
        {"step": i+1, "event": e, "description": f"{c}. Landmark: {l}"}
        for i,(e,l,c) in enumerate(zip(events, landmarks, captions))
    ]

def save_training_data(out_dir, video, frames, events, landmarks, captions):
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.splitext(os.path.basename(video))[0] + ".json")
    with open(dst, "w") as f:
        json.dump({
            "video": video,
            "events": events,
            "landmarks": landmarks,
            "captions":   captions,
        }, f, indent=2)
    print(f"[TrainData] saved → {dst}")
