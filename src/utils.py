# utils.py  – persistent‑landmark journey helper
import os, cv2, urllib.request, json, math, numpy as np, torch, easyocr
from PIL import Image
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer,
)

# ─────────── constant sets ──────────────────────────────────────────────
STATIC_YOLO_CLASSES = {
    "traffic light", "stop sign", "street sign", "traffic sign", "bus stop",
    "bench", "fire hydrant", "parking meter", "clock", "potted plant",
}
DYNAMIC_WORDS = {"car", "person", "truck", "bus", "motorcycle", "bicycle", "dog"}

# ─────────── small helpers ──────────────────────────────────────────────
def salient(txt: str) -> bool:
    words = txt.split()
    return len(words) >= 2 or (words and words[0][0].isupper())

def kind_of(txt: str) -> str:
    low = txt.lower()
    if any(k in low for k in ["shop", "store", "mart", "market", "express"]):
        return "shop"
    if any(k in low for k in ["registration", "office", "tower", "hotel",
                              "plaza", "building", "center", "carter"]):
        return "building"
    return "other"

def preprocess_crop(crop):
    h, w = crop.shape[:2]
    if h < 32:  # upscale tiny crops for OCR
        scale = 32 / h
        crop = cv2.resize(crop, (int(w * scale), 32), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

# ─────────── video frame utilities ──────────────────────────────────────
def extract_frames(video_path, fps: int = 1):
    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(native / fps))
    idx, ok, img = 0, *cap.read()
    print("[Frames] Starting extraction …")
    while ok:
        if idx % step == 0:
            yield img
        ok, img = cap.read()
        idx += 1
    cap.release()
    print("[Frames] Done.")

def detect_event_for_frame(prev_gray, cur_gray, dx_turn=1.5, flow_stop=0.20):
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, .5, 3, 15, 3, 5, 1.2, 0)
    dx   = flow[..., 0].mean()
    mag  = np.linalg.norm(flow, axis=2).mean()
    if mag < flow_stop:      return "stop"
    if dx   > dx_turn:       return "turn_right"
    if dx   < -dx_turn:      return "turn_left"
    return "drive"

# ─────────── lightweight downloader ─────────────────────────────────────
def fetch(dir_, url, fname):
    os.makedirs(dir_, exist_ok=True)
    path = os.path.join(dir_, fname)
    if url and not os.path.exists(path):
        print(f"[Model] Downloading {fname} …")
        urllib.request.urlretrieve(url, path)
    return path

# ─────────── detector + OCR setup ───────────────────────────────────────
def get_landmark_models(device):
    obj = YOLO(
        fetch("models",
              "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
              "yolov8n.pt")
    ).to(device).half()

    sign_path = "models/yolov8_signs.pt"
    sign = YOLO(sign_path).to(device).half() if os.path.exists(sign_path) else None

    ocr = easyocr.Reader(["en"], gpu=device.startswith("cuda"))
    return (obj, sign), ocr

def detect_landmarks_for_frame(frame, model, ocr, conf=0.25):
    result = model(frame, verbose=False, conf=conf)[0]
    if not result.boxes:
        return "none", ""
    labels, ocr_texts = [], []
    for b in result.boxes:
        cls_name = model.model.names[int(b.cls[0])]
        if cls_name not in STATIC_YOLO_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        t = " ".join(txt for txt in ocr.readtext(frame[y1:y2, x1:x2], detail=0))
        if t:
            ocr_texts.append(t)
            labels.append(f"{cls_name} [{t}]")
        else:
            labels.append(cls_name)
    return (", ".join(labels) if labels else "none"), " ".join(ocr_texts)

# ─────────── scene classification ───────────────────────────────────────
def get_scene_model(device):
    from torchvision import models, transforms
    path = fetch("models",
                 "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
                 "resnet18_places365.pth.tar")
    model = models.resnet18(num_classes=365)
    sd = torch.load(path, map_location="cpu")["state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    model.to(device).half().eval()

    cat_file = "categories_places365.txt"
    if not os.access(cat_file, os.W_OK):
        os.system("wget -q https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt")
    classes = [l.strip().split(" ")[0][3:] for l in open(cat_file)]

    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, classes, tf

def classify_scene_for_frame(frame, model, classes, tf):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inp = tf(img).unsqueeze(0).to(next(model.parameters()).device,
                                  dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(inp), 1)
    return classes[int(probs.argmax())]

# ─────────── caption model & generation ─────────────────────────────────
def get_caption_models(device):
    repo = "Salesforce/blip-image-captioning-base"
    return (
        BlipProcessor.from_pretrained(repo),
        BlipForConditionalGeneration.from_pretrained(repo).to(device)
    )

def generate_caption_for_frame(frame, proc, mod, landmark=None):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if max(img.size) > 512:
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    inp = proc(
        images=img,
        text=f"Scene contains: {landmark}." if landmark else "",
        return_tensors="pt"
    ).to(mod.device)
    with torch.no_grad():
        ids = mod.generate(**inp, max_new_tokens=30)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ─────────── long‑form summary with whitelist guard ─────────────────────
def generate_long_summary(events, landmarks, captions, scenes, ocr_texts, sign_stats):
    repo = "mistralai/Mistral-7B-Instruct-v0.2"
    model = AutoModelForCausalLM.from_pretrained(
        repo, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16
    )
    tok = AutoTokenizer.from_pretrained(repo)

    # per‑frame log lines
    lines = []
    for i in range(len(events)):
        clean_cap = " ".join(w for w in captions[i].split() if w.lower() not in DYNAMIC_WORDS)
        lines.append(
            f"Frame {i+1}: Scene={scenes[i]}, Event={events[i]}, "
            f"Caption={clean_cap}, Landmark={landmarks[i]}, OCR='{ocr_texts[i]}'"
        )

    # whitelist (top‑2 per bucket)
    whitelist = []
    for bucket in sign_stats.values():
        whitelist.extend(t for t, _ in sorted(bucket.items(),
                                              key=lambda kv: (-kv[1][0], -kv[1][1]))[:2])

    guard = (
        f"Use **only** these place names (and no others): {', '.join(whitelist)}.\n"
        if whitelist else
        "Do **not** mention any place names, cities, regions or countries.\n"
    )

    prompt = (
        "Summarize the journey below in a clear first‑person diary style. "
        "Ignore transient objects (people, vehicles). "
        "Do not add any headings such as 'Day 1'. "
        + guard +
        "---\n" + "\n".join(lines) + "\n---\nJourney Summary:\n"
    )

    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=160, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True).split("Journey Summary:")[-1].strip()

# ─────────── simple table helper ────────────────────────────────────────
def summarise_journey(ev, lm, cap, scn, ocr):
    return [
        {
            "step": i + 1,
            "event": ev[i],
            "scene": scn[i],
            "description": f"{cap[i]}. Landmark: {lm[i]}. OCR: '{ocr[i]}'",
        }
        for i in range(len(ev))
    ]
