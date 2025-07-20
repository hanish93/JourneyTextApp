# utils.py  — 2025‑07‑20
import os, cv2, urllib.request, json, shutil, re
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import easyocr
from ultralytics import YOLO
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
)

# ───────────────────────────────── frame extraction ──────────────────────────
def extract_frames(video_path, fps: int = 1):
    frames, cap = [], cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30
    step   = max(1, round(native / fps))
    idx, ok, img = 0, *cap.read()
    print("[Frames] Starting extraction …")
    while ok:
        if idx % step == 0:
            frames.append(img)
            print(f"[Frames] {len(frames)}")
        ok, img = cap.read()
        idx += 1
    cap.release()
    print(f"[Frames] Done. Total = {len(frames)}")
    return frames

# ───────────────────────────────── event detector ────────────────────────────
def detect_events(frames, stride=1, dx_turn=1.5, flow_stop=0.20) -> list[str]:
    if len(frames) < 2:
        return ["drive"] * len(frames)

    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    events, prev = ["drive"] * len(frames), gray[0]

    for i in tqdm(range(1, len(gray), stride), desc="[Events]"):
        flow = cv2.calcOpticalFlowFarneback(prev, gray[i], None, .5,3,15,3,5,1.2,0)
        dx   = flow[..., 0].mean()
        mag  = np.linalg.norm(flow, axis=2).mean()

        if   mag < flow_stop: evt = "stop"
        elif dx >  dx_turn :  evt = "turn_right"
        elif dx < -dx_turn :  evt = "turn_left"
        else:                evt = "drive"

        for k in range(stride):
            events[min(i - k, len(events)-1)] = evt
        prev = gray[i]

    return events

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
def detect_landmarks(frames, device, conf=0.25):
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

    names = []
    for i, f in enumerate(tqdm(frames, desc="[Landmarks]", unit="frame")):
        r = model(f, verbose=False, conf=conf)[0]
        if len(r.boxes) == 0:
            names.append("none"); continue
        boxes = []
        for b in r.boxes:
            cls = model.model.names[int(b.cls[0])]
            conf = float(b.conf[0])
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            crop = f[y1:y2, x1:x2]
            txt  = " ".join(t[1] for t in ocr.readtext(crop))
            boxes.append(f"{cls} ({conf:.2f}) [{txt}]" if txt else f"{cls} ({conf:.2f})")
        names.append(", ".join(boxes))
    return names

# ──────────────────────────── caption generator (BLIP‑2) ─────────────────────
def generate_captions(frames, device, landmarks=None, events=None):
    repo = "Salesforce/blip2-flan-t5-xl"
    processor = Blip2Processor.from_pretrained(repo)
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = Blip2ForConditionalGeneration.from_pretrained(
        repo, quantization_config=bnb_cfg, device_map="auto"
    ).eval()

    caps = []
    for i, f in enumerate(tqdm(frames, desc="[BLIP2]", unit="frame")):
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if max(img.size) > 512:
            res = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            img.thumbnail((512, 512), res)
        hint = f"Scene contains: {landmarks[i]}." if landmarks else ""
        ins  = processor(images=img, text=hint, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**ins, max_new_tokens=30)
        caps.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())

    if events is None:
        events = ["drive"] * len(frames)
    narrative = generate_long_summary(events, landmarks or ["none"]*len(frames), caps)
    return caps, narrative

# ───────────────────────── long‑form narrative generator ─────────────────────
def generate_long_summary(events, landmarks, captions):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    repo = "facebook/bart-large-cnn"
    tok  = AutoTokenizer.from_pretrained(repo)
    lm   = AutoModelForCausalLM.from_pretrained(repo)

    ctx_max, out_tok = 1024, 256
    limit = ctx_max - out_tok

    def pack(ev, lmks, caps):
        import json
        return (
            "Summarise this car journey in 250‑400 words.\n\n"
            f"Events: {json.dumps(ev)}\n"
            f"Landmarks: {json.dumps(lmks)}\n"
            f"Captions: {json.dumps(caps)}\n\nSummary:"
        )

    ev, lmks, caps = events[:], landmarks[:], captions[:]
    while len(tok(pack(ev, lmks, caps)).input_ids) > limit:
        longest = max((len(ev),"ev"), (len(lmks),"lm"), (len(caps),"cp"))[1]
        if   longest == "ev":  ev   = ev  [: int(len(ev  )*0.9)]
        elif longest == "lm":  lmks = lmks[: int(len(lmks)*0.9)]
        else:                  caps = caps[: int(len(caps)*0.9)]

    gen = pipeline("text-generation", model=lm, tokenizer=tok,
                   device=0 if torch.cuda.is_available() else -1)
    text = gen(pack(ev,lmks,caps), max_new_tokens=out_tok, do_sample=True, top_p=0.9)[0]["generated_text"]
    return text.split("Summary:",1)[-1].strip()

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
