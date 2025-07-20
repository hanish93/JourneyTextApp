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
    AutoModelForCausalLM,
    AutoTokenizer,
)

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

def detect_event_for_frame(prev_gray, current_gray, stride=1, dx_turn=1.5, flow_stop=0.20) -> str:
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx = flow[..., 0].mean()
    mag = np.linalg.norm(flow, axis=2).mean()
    if mag < flow_stop:
        evt = "stop"
    elif dx > dx_turn:
        evt = "turn_right"
    elif dx < -dx_turn:
        evt = "turn_left"
    else:
        evt = "drive"
    return evt

def get_or_download_model(name, local_dir, url=None, hf_repo=None, cfg="config.json"):
    if hf_repo:
        return hf_repo
    os.makedirs(local_dir, exist_ok=True)
    if not os.path.exists(os.path.join(local_dir, cfg)) and url:
        print(f"[Model] Downloading {name} …")
        urllib.request.urlretrieve(url, os.path.join(local_dir, os.path.basename(url)))
    return local_dir

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
    model = YOLO(logo_w if os.path.exists(logo_w) else coco_w).to(device)
    model.model.half()
    ocr = easyocr.Reader(["en", "it"], gpu=device.startswith("cuda"))
    return model, ocr

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
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.half()
    model.eval()
    file_name_category = "categories_places365.txt"
    if not os.access(file_name_category, os.W_OK):
        synset_url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
        os.system("wget " + synset_url)
    classes = []
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(" ")[0][3:])
    return model, classes

def classify_scene_for_frame(frame, model, classes):
    from torchvision import transforms
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_img = transform(img).unsqueeze(0).to(next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        logit = model(input_img)
    h_x = torch.nn.functional.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    return classes[idx[0]]

def detect_landmarks_for_frame(frame, model, ocr, conf=0.25):
    r = model(frame, verbose=False, conf=conf)[0]
    if len(r.boxes) == 0:
        return "none", ""
    boxes = []
    ocr_texts = []
    for b in r.boxes:
        cls = model.model.names[int(b.cls[0])]
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        read_results = ocr.readtext(crop)
        txt = " ".join([t[1] for t in read_results])
        if txt:
            ocr_texts.append(txt)
        boxes.append(f"{cls} ({conf:.2f}) [{txt}]" if txt else f"{cls} ({conf:.2f})")
    return ", ".join(boxes), " ".join(ocr_texts)

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
    ins = processor(images=img, text=hint, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**ins, max_new_tokens=30)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def generate_long_summary(events, landmarks, captions, scenes, ocr_texts):
    from transformers import pipeline
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    repo = "mistralai/Mistral-7B-Instruct-v0.2"
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(repo)
    summarizer = pipeline("text-generation", model=model, tokenizer=tok)
    long_text = "\n".join(
        [
            f"Frame {i+1}: Scene is {scenes[i]}. Event: {events[i]}. Caption: {captions[i]}. Landmark: {landmarks[i]}. Visible text: '{ocr_texts[i]}'"
            for i in range(len(scenes))
        ]
    )
    prompt = (
        "Summarize the following journey from a video in a travel-diary style.\nUse the frame-by-frame details to build a compelling narrative.\nMention key scenes, activities, and landmarks.\n---\n"
        + long_text
        + "\n---\nJourney Summary:\n"
    )
    response = summarizer(
        prompt,
        max_new_tokens=150,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
    )
    torch.cuda.empty_cache()
    return response[0]["generated_text"].split("Journey Summary:")[1].strip()

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
            {
                "video": video,
                "events": events,
                "landmarks": landmarks,
                "captions": captions,
            },
            f,
            indent=2,
        )
    print(f"[TrainData] saved → {dst}")
