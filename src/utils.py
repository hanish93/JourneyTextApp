import os, cv2, urllib.request, json, torch, easyocr, numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
)

# ───────────── static configurations ──────────────────────────────────────
STATIC_YOLO_CLASSES = {
    "traffic light", "stop sign", "street sign", "traffic sign", "bus stop",
    "bench", "fire hydrant", "parking meter", "clock", "potted plant"
}
DYNAMIC_WORDS = {"car", "person", "truck", "bus", "motorcycle", "bicycle", "dog"}

def salient(txt: str) -> bool:
    words = txt.split()
    if len(words) >= 2:
        return True
    return words and words[0][0].isupper()

def kind_of(txt: str) -> str:
    low = txt.lower()
    if any(w in low for w in ["shop", "store", "mart", "market", "express"]):
        return "shop"
    if any(w in low for w in [
        "registration", "office", "tower", "hotel", "plaza", "building", "center",
        "carter"]):
        return "building"
    return "other"

# ───────────── frame extraction & motion ──────────────────────────────────
def extract_frames(video_path, fps=1):
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

def detect_event_for_frame(prev_gray, current_gray, dx_turn=1.5, flow_stop=0.20):
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, .5,3,15,3,5,1.2,0)
    dx   = flow[...,0].mean()
    mag  = np.linalg.norm(flow, axis=2).mean()
    if   mag < flow_stop: evt = "stop"
    elif dx >  dx_turn :  evt = "turn_right"
    elif dx < -dx_turn :  evt = "turn_left"
    else:                 evt = "drive"
    return evt

# ───────────── basic downloader ───────────────────────────────────────────
def get_or_download_model(name, local_dir, url=None, cfg="config.json"):
    os.makedirs(local_dir, exist_ok=True)
    if url and not os.path.exists(os.path.join(local_dir, cfg)):
        print(f"[Model] Downloading {name} …")
        urllib.request.urlretrieve(url, os.path.join(local_dir, os.path.basename(url)))
    return local_dir

# ───────────── detectors & OCR ────────────────────────────────────────────
def get_landmark_models(device):
    coco_w = "models/yolov8n.pt"
    get_or_download_model(
        "yolov8n", os.path.dirname(coco_w),
        url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        cfg="yolov8n.pt"
    )
    obj_model = YOLO(coco_w).to(device).half()

    sign_w = "models/yolov8_signs.pt"
    sign_model = YOLO(sign_w).to(device).half() if os.path.exists(sign_w) else None

    ocr = easyocr.Reader(["en","it"], gpu=device.startswith("cuda"))
    return (obj_model, sign_model), ocr

def detect_landmarks_for_frame(frame, model, ocr, conf=0.25):
    result = model(frame, verbose=False, conf=conf)[0]
    if not result.boxes:
        return "none", ""
    labels, ocr_texts = [], []
    for b in result.boxes:
        cls_name = model.model.names[int(b.cls[0])]
        if cls_name not in STATIC_YOLO_CLASSES:
            continue
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        txt = " ".join(t[1] for t in ocr.readtext(frame[y1:y2, x1:x2]))
        if txt:
            ocr_texts.append(txt)
            labels.append(f"{cls_name} [{txt}]")
        else:
            labels.append(cls_name)
    if not labels:
        return "none", ""
    return ", ".join(labels), " ".join(ocr_texts)

# ───────────── scene classifier ──────────────────────────────────────────
def get_scene_model(device):
    from torchvision import models
    path = "models/resnet18_places365.pth.tar"
    if not os.path.exists(path):
        get_or_download_model(
            "resnet18_places365", os.path.dirname(path),
            url="http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
            cfg="resnet18_places365.pth.tar"
        )
    model = models.resnet18(num_classes=365)
    st = torch.load(path, map_location="cpu")["state_dict"]
    model.load_state_dict({k.replace("module.",""):v for k,v in st.items()})
    model.to(device).half().eval()

    cat_file = "categories_places365.txt"
    if not os.access(cat_file, os.W_OK):
        os.system("wget https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt")
    classes = [l.strip().split(" ")[0][3:] for l in open(cat_file)]
    return model, classes

def classify_scene_for_frame(frame, model, classes):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((256,256)), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inp = tf(img).unsqueeze(0).to(next(model.parameters()).device,
                                  dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        prob = torch.nn.functional.softmax(model(inp),1)
    return classes[prob.argmax()]

# ───────────── caption model ─────────────────────────────────────────────
def get_caption_models(device):
    repo = "Salesforce/blip-image-captioning-base"
    return (
        BlipProcessor.from_pretrained(repo),
        BlipForConditionalGeneration.from_pretrained(repo).to(device)
    )

def generate_caption_for_frame(frame, proc, mod, landmark=None):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if max(img.size) > 512:
        img.thumbnail((512,512), Image.Resampling.LANCZOS if hasattr(Image,"Resampling") else Image.LANCZOS)
    ins = proc(
        images=img,
        text=f"Scene contains: {landmark}." if landmark else "",
        return_tensors="pt").to(mod.device)
    with torch.no_grad():
        ids = mod.generate(**ins, max_new_tokens=30)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ───────────── long‑form summary ─────────────────────────────────────────
def generate_long_summary(events, landmarks, captions, scenes, ocr_texts, sign_stats):
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16
    )
    summarizer = lambda p: model.generate(**AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2")(
            p, return_tensors="pt").to(model.device),
        max_new_tokens=180, do_sample=True, temperature=0.7, top_p=0.9)

    long_text = []
    for i in range(len(events)):
        cap_clean = " ".join(w for w in captions[i].split() if w.lower() not in DYNAMIC_WORDS)
        long_text.append(
            f"Frame {i+1}: Scene={scenes[i]}, Event={events[i]}, "
            f"Caption={cap_clean}, Landmark={landmarks[i]}, OCR='{ocr_texts[i]}'"
        )
    trailer = []
    for k, bag in sign_stats.items():
        ranked = sorted(bag.items(), key=lambda kv:(-kv[1][0],-kv[1][1]))[:2]  # top‑2
        if ranked:
            trailer.append(f"Key {k}s: " + ", ".join(t for t,_ in ranked))
    prompt = (
        "Summarize the following journey in an engaging travel‑diary style. "
        "Mention only persistent landmarks (shops, buildings, signs) and avoid transient objects.\n---\n"
        + "\n".join(long_text) + ("\n"+"\n".join(trailer) if trailer else "") +
        "\n---\nJourney Summary:\n"
    )
    gen = summarizer(prompt)
    txt = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").batch_decode(gen, skip_special_tokens=True)[0]
    return txt.split("Journey Summary:")[-1].strip()

# ───────────── table helper (unchanged) ──────────────────────────────────
def summarise_journey(events, landmarks, captions, scenes, ocr_texts):
    return [
        {"step":i+1, "event":events[i], "scene":scenes[i],
         "description":f"{captions[i]}. Landmark: {landmarks[i]}. OCR: '{ocr_texts[i]}'"}
        for i in range(len(events))
    ]
