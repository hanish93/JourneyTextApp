import cv2, os, urllib.request
from ultralytics import YOLO
import torch
import easyocr
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def extract_frames(video_path, fps=1):
    """
    Extract frames from the input video at the specified FPS.
    Returns a list of frames (as numpy arrays).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    success, image = cap.read()
    print(f"[Frames] Starting frame extraction...")
    while success:
        if int(count % round(frame_rate / fps)) == 0:
            frames.append(image)
            print(f"[Frames] Extracted frame {len(frames)}")
        success, image = cap.read()
        count += 1
    cap.release()
    print(f"[Frames] Extraction complete. Total frames: {len(frames)}")
    return frames

def detect_events(frames,
                  stride: int = 1,
                  dx_turn: float = 1.5,
                  flow_stop: float = 0.20) -> list[str]:
    """
    Heuristic event detector using dense optical flow.

    Returns one label per frame:
    'drive' | 'turn_left' | 'turn_right' | 'stop'
    """
    if len(frames) < 2:
        return ["drive"] * len(frames)

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    events = ["drive"] * len(frames)
    prev = gray_frames[0]

    for i in range(1, len(gray_frames), stride):
        flow = cv2.calcOpticalFlowFarneback(
            prev, gray_frames[i],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        dx  = flow[..., 0].mean()                    # horizontal component
        mag = np.linalg.norm(flow, axis=2).mean()    # total magnitude

        if mag < flow_stop:
            evt = "stop"
        elif dx >  dx_turn:
            evt = "turn_right"
        elif dx < -dx_turn:
            evt = "turn_left"
        else:
            evt = "drive"

        # label this frame and the skipped ones (if stride > 1)
        for k in range(stride):
            idx = min(i - k, len(events) - 1)
            events[idx] = evt

        prev = gray_frames[i]

    return events

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import json
import shutil

def get_or_download_model(model_name, local_dir, download_url=None, hf_repo=None, config_file="config.json"):
    """
    Checks if a model exists locally. If not, downloads it (from URL or HuggingFace repo).
    Returns the local directory path to the model.
    """
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
    config_path = os.path.join(local_dir, config_file)
    if not os.path.exists(config_path):
        if download_url:
            # Download from direct URL
            print(f"[Model] Downloading model from {download_url} to {local_dir}")
            urllib.request.urlretrieve(download_url, os.path.join(local_dir, os.path.basename(download_url)))
        elif hf_repo:
            # Download from HuggingFace repo
            print(f"[Model] Downloading HuggingFace model {hf_repo} to {local_dir}")
            if os.path.isdir(local_dir):
                shutil.rmtree(local_dir)
            _ = Blip2Processor.from_pretrained(hf_repo, cache_dir=local_dir)
            _ = Blip2ForConditionalGeneration.from_pretrained(hf_repo, cache_dir=local_dir)
    return local_dir

def save_training_data(training_data_dir, video_path, frames, events, landmarks, captions):
    """
    Save frames, events, landmarks, and captions for a video as training data.
    Frames are not saved as images to save space, but you can extend this to save images if needed.
    """
    os.makedirs(training_data_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_json = os.path.join(training_data_dir, base + '.json')
    data = {
        "video": video_path,
        "events": events,
        "landmarks": landmarks,
        "captions": captions
    }
    with open(out_json, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[TrainData] Saved training data to {out_json}")

def get_blip2_model_and_processor(model_dir, device):
    """
    Utility to load BLIP2 processor and model from a local directory (for training or inference).
    Ensures the model is downloaded if not present.
    """
    repo = "Salesforce/blip2-flan-t5-xl"
    local_dir = get_or_download_model(repo, model_dir, hf_repo=repo)
    processor = Blip2Processor.from_pretrained(local_dir)
    model = Blip2ForConditionalGeneration.from_pretrained(local_dir).to(device)
    return processor, model

def detect_landmarks(frames, device):
    import os, urllib.request, cv2, easyocr
    from ultralytics import YOLO
    from tqdm import tqdm

    logo_path = os.path.join("models", "yolov8_logo.pt")
    coco_path = os.path.join("models", "yolov8n.pt")

    if os.path.exists(logo_path):
        model_path = logo_path
    else:
        model_path = coco_path
        model_dir = os.path.dirname(model_path)
        yolov8_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        get_or_download_model("yolov8n", model_dir, download_url=yolov8_url, config_file="yolov8n.pt")
        model_path = coco_path

    model = YOLO(model_path).to(device)
    reader = easyocr.Reader(["en"], gpu=device.startswith("cuda"))

    names = []
    for frame in tqdm(frames, unit="frame"):
        r = model(frame, verbose=False)[0]
        if len(r.boxes) == 0:
            names.append("none")
            continue
        b = r.boxes[0]
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls = model.model.names[int(b.cls[0])]
        crop = frame[y1:y2, x1:x2]
        txt = " ".join(t[1] for t in reader.readtext(crop))
        names.append(f"{cls} {txt}".strip())

    return names


# utils.py
def generate_captions(frames, device, landmarks=None):
    import os, shutil, cv2, torch
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    repo = "Salesforce/blip2-flan-t5-xl"
    local_dir = os.path.join("models", repo.split("/")[-1])
    local_dir = get_or_download_model(repo, local_dir, hf_repo=repo)
    processor = Blip2Processor.from_pretrained(local_dir)
    model = Blip2ForConditionalGeneration.from_pretrained(local_dir).to(device)

    captions = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hint = f"Scene contains: {landmarks[i]}. " if landmarks and i < len(landmarks) else ""
        inputs = processor(images=img, text=hint, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=30)
        captions.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return captions



def summarise_journey(events, landmarks, captions):
    summary = []
    for i, (event, landmark, caption) in enumerate(zip(events, landmarks, captions)):
        summary.append({
            "step": i+1,
            "event": event,
            "description": f"{caption} Landmark: {landmark}."
        })
    return summary

def generate_long_summary(events, landmarks, captions):
    """
    Generate a long-form narrative summary (200-1000 words) describing the journey using a language model.
    Uses events, landmarks, captions, and other data to construct a prompt for the model.
    Handles model context window by truncating input if needed.
    """
    from transformers import pipeline, AutoTokenizer
    import os
    import json

    model_name = os.environ.get("JOURNEY_SUMMARY_MODEL", "gpt2")
    local_dir = os.path.join("models", model_name.replace("/", "_"))
    if not os.path.exists(local_dir):
        AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    max_context = getattr(tokenizer, 'model_max_length', 1024)
    max_new_tokens = 256
    # Reserve space for output tokens
    max_prompt_tokens = max_context - max_new_tokens

    # Prepare the prompt base
    prompt_base = (
        "You are an expert journey summarizer. Given the following driving events, detected landmarks, and scene captions, "
        "write a detailed, engaging, and coherent long-form summary (200-1000 words) describing the journey. "
        "Include navigation, notable landmarks, road/traffic/weather conditions, and overall impressions.\n\n"
    )

    # Truncate events, landmarks, captions to fit within max_prompt_tokens
    def truncate_json_list(data, max_items):
        if len(data) > max_items:
            return data[:max_items] + ["..."]
        return data

    # Start with a reasonable guess, then shrink if needed
    max_items = 50
    while True:
        prompt = (
            prompt_base +
            f"Events: {json.dumps(truncate_json_list(events, max_items))}\n"
            f"Landmarks: {json.dumps(truncate_json_list(landmarks, max_items))}\n"
            f"Captions: {json.dumps(truncate_json_list(captions, max_items))}\n\n"
            "Summary:"
        )
        n_tokens = len(tokenizer.encode(prompt))
        if n_tokens <= max_prompt_tokens or max_items == 1:
            break
        max_items -= 1

    generator = pipeline("text-generation", model=model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        num_return_sequences=1,
        truncation=True
    )
    summary = output[0]["generated_text"][len(prompt):].strip()
    # Truncate to 1000 words if needed
    words = summary.split()
    if len(words) > 1000:
        summary = " ".join(words[:1000]) + "..."
    return summary
