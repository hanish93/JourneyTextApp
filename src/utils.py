# src/utils.py  —  High‑signal video‑to‑diary helpers

import os
import cv2
import urllib.request
import torch
import easyocr
import numpy as np
from itertools import islice
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

KEEP = {
    "traffic light", "stop sign", "street sign", "traffic sign",
    "bench", "fire hydrant", "parking meter", "clock", "potted plant",
}
DYNAMIC = {"car", "truck", "bus", "motorcycle", "bicycle", "person", "dog"}


def fetch(dst_dir: str, url: str, fname: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, fname)
    if url and not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    nat = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(nat / fps))
    idx, ok, img = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield img
        ok, img = cap.read()
        idx += 1
    cap.release()


def move(prev, cur, dx=1.5, stop=0.2):
    if prev is None:
        return "drive"
    f = cv2.calcOpticalFlowFarneback(prev, cur, None, .5, 3, 15, 3, 5, 1.2, 0)
    dxm = f[..., 0].mean()
    mag = np.linalg.norm(f, axis=2).mean()
    if mag < stop:
        return "stop"
    if dxm > dx:
        return "turn_right"
    if dxm < -dx:
        return "turn_left"
    return "drive"


def load_det(dev, custom_model_path=None):
    if custom_model_path:
        y = YOLO(custom_model_path).to(dev).half()
    else:
        pt = fetch(
            "models",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "yolov8n.pt",
        )
        y = YOLO(pt).to(dev).half()
    ocr_reader = easyocr.Reader(["en"], gpu=dev.startswith("cuda"))
    return y, ocr_reader


def signage_names(img, ocr, conf_thresh=0.4):
    raw = ocr.readtext(img, detail=1)
    picks = []
    for _, text, prob in raw:
        text = text.strip()
        if prob >= conf_thresh and len(text) >= 3 and any(c.isalpha() for c in text):
            picks.append(text)
    return list(dict.fromkeys(picks))


def landmarks(img, yolo_model, ocr_reader, conf=0.25):
    r = yolo_model(img, conf=conf, verbose=False)[0]
    names = []
    # YOLO boxes → OCR
    for b in r.boxes:
        cls = yolo_model.model.names[int(b.cls[0])]
        if cls not in KEEP:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        txt = " ".join(ocr_reader.readtext(img[y1:y2, x1:x2], detail=0))
        if txt:
            names.append(txt)
    # Full‑frame signage
    names += signage_names(img, ocr_reader)
    return list(dict.fromkeys(names))


def load_cap(dev):
    return pipeline(
        "image-text-to-text",
        model="Salesforce/instructblip-flan-t5-xl",
        device_map="auto" if dev.startswith("cuda") else None,
    )


def cap_img(img, cap_pipe, hint: str = "") -> str:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if max(pil.size) > 640:
        pil.thumbnail((640, 640), Image.Resampling.LANCZOS)
    prompt = hint or "Describe the scene briefly."
    out = cap_pipe({"images": pil, "text": prompt}, max_new_tokens=30)[0]
    return out["generated_text"]


# Summariser (FLAN‑T5‑large)
tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
summ = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").cpu()


def diary(lines, whitelist, max_lines=40):
    lines = list(islice(lines, max_lines))
    if whitelist:
        guard = "Only these place names may appear: " + ", ".join(sorted(whitelist)) + ".\n"
    else:
        guard = "No place names detected—do NOT invent any.\n"

    prompt = (
        "Write 3‑4 simple first‑person sentences about the drive. "
        "Ignore people/vehicles; no headings.\n"
        + guard
        + "\n".join(f"- {l}" for l in lines)
        + "\n\nSummary:"
    )
    ids = summ.generate(**tok(prompt, return_tensors="pt"), max_new_tokens=120, do_sample=False)
    out = tok.decode(ids[0], skip_special_tokens=True)
    return out.split("Summary:")[-1].strip()
