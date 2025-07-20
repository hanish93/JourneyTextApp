import os, cv2, urllib.request, json
import numpy as np, collections, torch, easyocr
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ───────────── text‑filter helpers ──────────────────────────────────────────
def salient(txt: str) -> bool:
    words = txt.split()
    if len(words) >= 2:
        return True                     # at least two words
    return words and words[0][0].isupper()

def kind_of(txt: str) -> str:
    low = txt.lower()
    if any(w in low for w in ["shop", "store", "mart", "market", "express"]):
        return "shop"
    if any(w in low for w in ["tower", "hotel", "plaza", "building", "center", "office"]):
        return "building"
    return "other"

# ───────────── frame extraction / motion / helpers ─────────────────────────
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

def detect_event_for_frame(prev_gray, current_gray, stride=1, dx_turn=1.5, flow_stop=0.20) -> str:
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, .5,3,15,3,5,1.2,0)
    dx   = flow[..., 0].mean()
    mag  = np.linalg.norm(flow, axis=2).mean()
    if   mag < flow_stop: evt = "stop"
    elif dx >  dx_turn :  evt = "turn_right"
    elif dx < -dx_turn :  evt = "turn_left"
    else:                 evt = "drive"
    return evt

def get_or_download_model(name, local_dir, url=None, hf_repo=None, cfg="config.json"):
    if hf_repo:
        return hf_repo
    os.makedirs(local_dir, exist_ok=True)
    if not os.path.exists(os.path.join(local_dir, cfg)) and url:
        print(f"[Model] Downloading {name} …")
        urllib.request.urlretrieve(url, os.path.join(local_dir, os.path.basename(url)))
    return local_dir

# ───────────── object & sign detectors + OCR ───────────────────────────────
def get_landmark_models(device):
    logo_w = "models/yolov8_logo.pt"
    coco_w = "models/yolov8n.pt"
    if not os.path.exists(logo_w):
        get_or_download_model(
            "yolov8n",
            os.path.dirname(coco_w),
            url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            cfg="yolov8n.pt",
        )
    obj_model = YOLO(logo_w if os.path.exists(logo_w) else coco_w).to(device)
    obj_model.model.half()

    sign_w = "models/yolov8_signs.pt"            # optional fine‑tuned weights
    sign_model = None
    if os.path.exists(sign_w):
        sign_model = YOLO(sign_w).to(device)
        sign_model.model.half()

    ocr = easyocr.Reader(["en", "it"], gpu=device.startswith("cuda"))
    return (obj_model, sign_model), ocr

# ───────────── scene classification  ───────────────────────────────────────
def get_scene_model(device):
    from torchvision import models
    model_file = "models/resnet18_places365.pth.tar"
    if not os.path.exists(model_file):
        get_or_download_model(
            "resnet18_places365",
            os.path.dirname(model_file),
            url="http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
            cfg="resnet18_places365.pth.tar",
        )
    model = models.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda s, l: s)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device).half().eval()

    cat_file = "categories_places365.txt"
    if not os.access(cat_file, os.W_OK):
        os.system("wget https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt")
    classes = [line.strip().split(" ")[0][3:] for line in open(cat_file)]
    return model, classes

def classify_scene_for_frame(frame, model, classes):
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inp = tfm(img).unsqueeze(0).to(next(model.parameters()).device,
                                   dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        logit = model(inp)
    idx = torch.nn.functional.softmax(logit,1).data.squeeze().sort(0,True)[1]
    return classes[idx[0]]

# ───────────── object‑box OCR helper ───────────────────────────────────────
def detect_landmarks_for_frame(frame, model, ocr, conf=0.25):
    r = model(frame, verbose=False, conf=conf)[0]
    if len(r.boxes) == 0:
        return "none", ""
    boxes, ocr_texts = [], []
    for b in r.boxes:
        cls = model.model.names[int(b.cls[0])]
        # Skip vehicle-related classes
        if cls.lower() in ["car", "bus", "truck", "motorcycle", "bicycle", "vehicle"]:
            continue
        # Focus on relevant classes
        if cls.lower() not in [
            "traffic light", "traffic sign", "sign", "stop sign", "street sign",
            "store", "shop", "market", "mart", "hoarding", "billboard", "plaza", "building", "light"
        ]:
            continue
        c   = float(b.conf[0])
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        txt = " ".join(t[1] for t in ocr.readtext(frame[y1:y2, x1:x2]))
        if txt:
            ocr_texts.append(txt)
        boxes.append(f"{cls} ({c:.2f}) [{txt}]" if txt else f"{cls} ({c:.2f})")
    if not boxes:
        return "none", ""
    return ", ".join(boxes), " ".join(ocr_texts)

# ───────────── caption model  ──────────────────────────────────────────────
def get_caption_models(device):
    repo = "Salesforce/blip-image-captioning-base"
    proc = BlipProcessor.from_pretrained(repo)
    mod  = BlipForConditionalGeneration.from_pretrained(repo).to(device)
    return proc, mod

def generate_caption_for_frame(frame, processor, model, landmark=None):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if max(img.size) > 512:
        res = Image.Resampling.LANCZOS if hasattr(Image,"Resampling") else Image.LANCZOS
        img.thumbnail((512,512), res)
    hint = f"Scene contains: {landmark}." if landmark else ""
    ins  = processor(images=img, text=hint, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**ins, max_new_tokens=30)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ───────────── long‑form summary  ──────────────────────────────────────────
def generate_long_summary(events, landmarks, captions, scenes, ocr_texts, sign_stats):
    from transformers import pipeline
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    repo = "mistralai/Mistral-7B-Instruct-v0.2"
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(repo,
            device_map="auto", quantization_config=bnb, torch_dtype=torch.float16)
    tok   = AutoTokenizer.from_pretrained(repo)
    summarizer = pipeline("text-generation", model=model, tokenizer=tok)

    long_text = "\n".join([
        f"Frame {i+1}: Scene={scenes[i]}, Event={events[i]}, Caption={captions[i]}, "
        f"Landmark={landmarks[i]}, OCR='{ocr_texts[i]}'"
        for i in range(len(scenes))
    ])

    # ------- add top‑K salient sign texts ----------------------------------
    trailer = ""
    top_txt = []
    for kind, bag in sign_stats.items():
        ranked = sorted(bag.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))[:3]
        if ranked:
            top_txt.append(f"Key {kind}s: " + ", ".join(t for t,_ in ranked))
    if top_txt:
        trailer = "\n" + "\n".join(top_txt)

    prompt = (
        "You are an expert journey summarizer. Given the following driving events, detected landmarks, and scene captions, "
        "write a detailed, engaging, and coherent long-form summary (200-1000 words) describing the journey. List only high confidence country/region names, "
        "Include navigation, notable landmarks, road/traffic/weather conditions, and overall impressions.\n\n"
        "---\n" + long_text + trailer + "\n---\nJourney Summary:\n"
    )

    resp = summarizer(prompt, max_new_tokens=150, temperature=0.7, top_p=0.9)[0]
    torch.cuda.empty_cache()
    return resp["generated_text"].split("Journey Summary:")[1].strip()

# ───────────── misc helpers (unchanged) ────────────────────────────────────
def summarise_journey(events, landmarks, captions, scenes, ocr_texts):
    return [
        {
            "step": i + 1,
            "event": events[i],
            "scene": scenes[i],
            "description": f"{captions[i]}. Landmark: {landmarks[i]}. OCR: '{ocr_texts[i]}'",
        }
        for i in range(len(events))
    ]

def save_training_data(out_dir, video, frames, events, landmarks, captions):
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.splitext(os.path.basename(video))[0] + ".json")
    with open(dst, "w") as f:
        json.dump(
            {"video": video, "events": events, "landmarks": landmarks, "captions": captions},
            f, indent=2
        )
    print(f"[TrainData] saved → {dst}")
