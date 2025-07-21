# utils.py  —  High‑signal video‑to‑diary helpers
import os, cv2, urllib.request, torch, easyocr, numpy as np
from itertools import islice
from PIL import Image
from ultralytics import YOLO
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
)

# Permanent street objects
KEEP_CLASSES = {
    "traffic light","stop sign","street sign","traffic sign",
    "bench","fire hydrant","parking meter","clock","potted plant"
}
DYNAMIC = {"car","truck","bus","motorcycle","bicycle","person","dog"}

# ─── Downloader helper ────────────────────────────────────────────────
def fetch(dst_dir, url, fname):
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, fname)
    if url and not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path

# ─── Frame sampler (1 fps) ────────────────────────────────────────────
def sample_frames(video, fps=1):
    cap=cv2.VideoCapture(video); nat=cap.get(cv2.CAP_PROP_FPS) or 30
    step=max(1,round(nat/fps)); idx,ok,img=0,*cap.read()
    while ok:
        if idx%step==0: yield img
        ok,img=cap.read(); idx+=1
    cap.release()

# ─── Simple motion verb ───────────────────────────────────────────────
def motion(prev,cur,dx_thr=1.5,stop_thr=0.2):
    if prev is None: return"drive"
    flow=cv2.calcOpticalFlowFarneback(prev,cur,None,.5,3,15,3,5,1.2,0)
    dx=flow[...,0].mean(); mag=np.linalg.norm(flow,axis=2).mean()
    if mag<stop_thr:     return"stop"
    if dx> dx_thr:       return"turn_right"
    if dx<-dx_thr:       return"turn_left"
    return"drive"

# ─── YOLO detector + OCR ──────────────────────────────────────────────
def load_detectors(dev):
    obj=YOLO(fetch("models",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8n.pt")).to(dev).half()
    ocr=easyocr.Reader(["en"],gpu=dev.startswith("cuda"))
    return obj,ocr

def static_landmarks(frame,obj,ocr,conf=0.25):
    res=obj(frame,conf=conf,verbose=False)[0]
    names=[]
    for b in res.boxes:
        cls=obj.model.names[int(b.cls[0])]
        if cls not in KEEP_CLASSES: continue
        x1,y1,x2,y2=map(int,b.xyxy[0])
        txt=" ".join(ocr.readtext(frame[y1:y2,x1:x2],detail=0))
        if txt: names.append(txt)
    return names

# ─── ShareCaptioner‑Video (HF pipeline) ───────────────────────────────
def load_captioner(dev):
    cap=pipeline("image-to-text",
                 model="Lin-Chen/ShareCaptioner-Video",
                 device_map="auto" if dev.startswith("cuda") else None)
    return cap

# ─── Text summariser (Flan‑T5 large on CPU) ───────────────────────────
tok_sum=AutoTokenizer.from_pretrained("google/flan-t5-large")
mod_sum=AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").cpu()

def summarise(frames_caps, whitelist, max_caps=40):
    caps=list(islice(frames_caps,max_caps))
    guard=("Use only these place names: "+", ".join(sorted(whitelist))+".\n") if whitelist else \
           "No place names were detected; do NOT invent any.\n"
    prompt=("Write 3‑4 simple sentences in first person summarising the drive. "
            "Ignore people/vehicles; no headings.\n"+guard+
            "\n".join("- "+c for c in caps)+"\n\nSummary:")
    ids=mod_sum.generate(**tok_sum(prompt,return_tensors="pt"),
                         max_new_tokens=120,do_sample=False)
    return tok_sum.decode(ids[0],skip_special_tokens=True).split("Summary:")[-1].strip()
