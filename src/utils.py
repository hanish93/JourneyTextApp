# utils.py  —  persistent‑landmark diary helpers
import os, cv2, urllib.request, torch, easyocr, numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer,
)

STATIC_YOLO_CLASSES = {
    "traffic light", "stop sign", "street sign", "traffic sign",
    "bus stop", "bench", "fire hydrant", "parking meter",
    "clock", "potted plant",
}
DYNAMIC_WORDS = {
    "car", "person", "truck", "bus", "motorcycle", "bicycle", "dog"
}

salient  = lambda t: (t.split()
                      and (len(t.split()) >= 2 or t[0].isupper()))
kind_of  = lambda t: ("shop" if any(k in t.lower() for k in
                    ["shop","store","mart","market","express"])
                    else "building" if any(k in t.lower() for k in
                    ["registration","office","tower","hotel","plaza",
                     "building","center","carter"])
                    else "other")

# ─────────── frame handling ─────────────────────────────────────────────
def extract_frames(path, fps=1):
    cap = cv2.VideoCapture(path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(native / fps))
    idx, ok, img = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield img
        ok, img = cap.read(); idx += 1
    cap.release()

def detect_event_for_frame(prev, cur, dx=1.5, stop=0.2):
    if prev is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None, .5, 3, 15, 3, 5, 1.2, 0)
    dxm = flow[..., 0].mean(); mag = np.linalg.norm(flow, axis=2).mean()
    if mag < stop:    return "stop"
    if dxm > dx:      return "turn_right"
    if dxm < -dx:     return "turn_left"
    return "drive"

# ─────────── model fetch helper ─────────────────────────────────────────
def fetch(dir_, url, fname):
    os.makedirs(dir_, exist_ok=True)
    path = os.path.join(dir_, fname)
    if url and not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path

# ─────────── detectors & OCR ────────────────────────────────────────────
def get_landmark_models(device):
    obj = YOLO(fetch("models",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8n.pt")).to(device).half()
    ocr = easyocr.Reader(["en"], gpu=device.startswith("cuda"))
    return (obj, None), ocr          # sign model unused → None

def detect_landmarks_for_frame(frame, model, ocr, conf=0.25):
    res = model(frame, verbose=False, conf=conf)[0]
    if not res.boxes:
        return "none", ""
    labels, txts = [], []
    for b in res.boxes:
        cls = model.model.names[int(b.cls[0])]
        if cls not in STATIC_YOLO_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        t = " ".join(ocr.readtext(frame[y1:y2, x1:x2], detail=0))
        if t:
            txts.append(t); labels.append(f"{cls} [{t}]")
        else:
            labels.append(cls)
    return (", ".join(labels) or "none"), " ".join(txts)

# ─────────── scene classifier ───────────────────────────────────────────
def get_scene_model(device):
    from torchvision import models, transforms
    ckpt = fetch("models",
        "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
        "resnet18_places365.pth.tar")
    model = models.resnet18(num_classes=365)
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    model.to(device).half().eval()
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    classes = [l.strip().split(" ")[0][3:]
               for l in open("categories_places365.txt")]
    return model, classes, tf

def classify_scene_for_frame(frame, model, classes, tf):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inp = tf(img).unsqueeze(0).to(next(model.parameters()).device,
                                  dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        p = torch.nn.functional.softmax(model(inp), 1)
    return classes[int(p.argmax())]

# ─────────── caption generator ──────────────────────────────────────────
def get_caption_models(device):
    repo = "Salesforce/blip-image-captioning-base"
    return (BlipProcessor.from_pretrained(repo),
            BlipForConditionalGeneration.from_pretrained(repo).to(device))

def generate_caption_for_frame(frame, proc, mod, landmark=None):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if max(img.size) > 512:
        img.thumbnail((512,512), Image.Resampling.LANCZOS)
    ins = proc(images=img,
               text=(f"Scene contains: {landmark}." if landmark else ""),
               return_tensors="pt").to(mod.device)
    with torch.no_grad():
        ids = mod.generate(**ins, max_new_tokens=30)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ─────────── summary (strict whitelist, no headings) ────────────────────
def generate_long_summary(events, landmarks, captions, scenes, ocr_txts, sign_stats):
    repo = "mistralai/Mistral-7B-Instruct-v0.2"
    model = AutoModelForCausalLM.from_pretrained(
        repo, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16
    )
    tok = AutoTokenizer.from_pretrained(repo)

    # frame logs
    lines = []
    for i in range(len(events)):
        cap_clean = " ".join(
            w for w in captions[i].split() if w.lower() not in DYNAMIC_WORDS
        )
        lines.append(
            f"Frame {i+1}: Scene={scenes[i]}, Event={events[i]}, "
            f"Caption={cap_clean}, Landmark={landmarks[i]}, OCR='{ocr_txts[i]}'"
        )

    # whitelist
    whitelist = []
    for bag in sign_stats.values():
        whitelist.extend(
            t for t,_ in sorted(bag.items(),
                                key=lambda kv: (-kv[1][0], -kv[1][1]))[:2]
        )

    if whitelist:
        guard = ("Only these place names may be used: "
                 + ", ".join(whitelist) + ".\n")
    else:
        guard = ("No place names were detected; do NOT invent any.\n"
                 "No notable landmarks were detected.\n")

    prompt = (
        "Write 3‑4 simple sentences describing the journey, in first person. "
        "Do NOT add headings such as 'Day 1'. "
        "Ignore transient objects (people, vehicles). "
        + guard
        + "---\n" + "\n".join(lines) + "\n---\nSummary:\n"
    )

    out = model.generate(**tok(prompt, return_tensors="pt").to(model.device),
                         max_new_tokens=120,
                         do_sample=False)          # deterministic
    return tok.decode(out[0], skip_special_tokens=True).split("Summary:")[-1].strip()

def summarise_journey(ev,lm,cap,scn,ocr):
    return [
        {"step":i+1,"event":ev[i],"scene":scn[i],
         "description":f"{cap[i]} | {lm[i]} | OCR '{ocr[i]}'"}
        for i in range(len(ev))
    ]
