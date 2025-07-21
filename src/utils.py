# utils.py  — stable diary helpers (YOLO + InstructBLIP + FLAN‑T5)
import os, cv2, urllib.request, torch, easyocr, numpy as np
from itertools import islice
from PIL import Image
from ultralytics import YOLO
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
)

KEEP = {"traffic light","stop sign","street sign","traffic sign",
        "bench","fire hydrant","parking meter","clock","potted plant"}
DYNAMIC = {"car","truck","bus","motorcycle","bicycle","person","dog"}

# — Downloader helper —
def fetch(d,url,f): os.makedirs(d,exist_ok=True); p=os.path.join(d,f)
  ; 0 if url and os.path.exists(p) else urllib.request.urlretrieve(url,p); return p

# — Frame sampler (1 fps) —
def frames(vid,fps=1):
    c=cv2.VideoCapture(vid); nat=c.get(cv2.CAP_PROP_FPS)or30
    step=max(1,round(nat/fps)); i,ok,img=0,*c.read()
    while ok:
        if i%step==0: yield img
        ok,img=c.read(); i+=1
    c.release()

# — Motion verb —
def move(prev,cur,dx=1.5,stop=0.2):
    if prev is None:return"drive"
    f=cv2.calcOpticalFlowFarneback(prev,cur,None,.5,3,15,3,5,1.2,0)
    dxm=f[...,0].mean(); mag=np.linalg.norm(f,axis=2).mean()
    return"stop"if mag<stop else"turn_right"if dxm>dx else"turn_left"if dxm<-dx else"drive"

# — Detectors & OCR —
def load_det(dev):
    y=YOLO(fetch("models",
      "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt","y.pt")
      ).to(dev).half()
    o=easyocr.Reader(["en"],gpu=dev.startswith("cuda"))
    return y,o

def landmarks(img,y,o,conf=0.25):
    r=y(img,conf=conf,verbose=False)[0]; names=[]
    for b in r.boxes:
        cls=y.model.names[int(b.cls[0])]
        if cls not in KEEP:continue
        x1,y1,x2,y2=map(int,b.xyxy[0])
        t=" ".join(o.readtext(img[y1:y2,x1:x2],detail=0))
        if t:names.append(t)
    return names

# — Captioner: InstructBLIP T5‑XL —
def load_cap(dev):
    return pipeline("image-to-text",
        model="Salesforce/instructblip-flan-t5-xl",
        device_map="auto" if dev.startswith("cuda") else None)

def cap_img(img,cap_pipe,hint=""):
    pil=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    if max(pil.size)>640: pil.thumbnail((640,640),Image.Resampling.LANCZOS)
    res=cap_pipe(pil,max_new_tokens=30,
                 generate_kwargs={"temperature":0.2,"repetition_penalty":1.1},
                 prompt=hint or None)[0]["generated_text"]
    return res

# — Summariser: FLAN‑T5‑large on CPU —
tok=AutoTokenizer.from_pretrained("google/flan-t5-large")
summ=AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").cpu()

def diary(lines,whitelist,max_lines=40):
    lines=list(islice(lines,max_lines))
    guard=("Only these place names may appear: "+", ".join(sorted(whitelist))+".\n")\
           if whitelist else "No place names detected—do NOT invent any.\n"
    prompt=("Write 3‑4 simple first‑person sentences about the drive. "
            "Ignore people/vehicles; no headings.\n"+guard+
            "\n".join("- "+l for l in lines)+"\n\nSummary:")
    ids=summ.generate(**tok(prompt,return_tensors="pt"),
                      max_new_tokens=120,do_sample=False)
    return tok.decode(ids[0],skip_special_tokens=True).split("Summary:")[-1].strip()
