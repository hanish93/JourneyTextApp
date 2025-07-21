# utils.py  —  High‑signal video‑to‑diary helpers
import os, cv2, urllib.request, torch, easyocr, numpy as np
from itertools import islice
from PIL import Image
from ultralytics import YOLO
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
)

KEEP = {
    "traffic light", "stop sign", "street sign", "traffic sign",
    "bench", "fire hydrant", "parking meter", "clock", "potted plant",
}
DYNAMIC = {"car", "truck", "bus", "motorcycle", "bicycle", "person", "dog"}

# ─── simple downloader helper (safe indentation) ───────────────────────
def fetch(dst_dir: str, url: str, fname: str) -> str:
    """
    Ensure `dst_dir/fname` exists; download from `url` if missing.
    Returns the full path.
    """
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, fname)
    if url and not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


# ─── Frame sampler (1 fps) ────────────────────────────────────────────
def frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    nat = cap.get(cv2.CAP_PROP_FPS) or 30          # ← space before/after “or”
    step = max(1, round(nat / fps))

    idx, ok, img = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield img
        ok, img = cap.read()
        idx += 1
    cap.release()


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

# ─── captioner: InstructBLIP‑FLAN‑T5‑XL ────────────────────────────────
def load_cap(dev):
    """
    Returns a HF pipeline that accepts {'image': PIL, 'text': prompt}
    and produces a caption.
    """
    return pipeline(
        "image-text-to-text",
        model="Salesforce/instructblip-flan-t5-xl",
        device_map="auto" if dev.startswith("cuda") else None,
    )


# ─── caption a single frame with InstructBLIP ──────────────────────────
# ─── caption a single frame ────────────────────────────────────────────
def cap_img(img, cap_pipe, hint: str = "") -> str:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if max(pil.size) > 640:
        pil.thumbnail((640, 640), Image.Resampling.LANCZOS)

    prompt = hint if hint else "Describe the scene briefly."
    # image‑text‑to‑text expects a dict with both fields
    res = cap_pipe({"image": pil, "text": prompt}, max_new_tokens=30)[0]
    return res["generated_text"]



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
