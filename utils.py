import cv2, os, urllib.request
from ultralytics import YOLO
import torch
import easyocr
from tqdm import tqdm

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

def detect_events(frames):
    """
    Detect events (e.g., drive, turn, stop) for each frame.
    Placeholder: In a real implementation, use temporal models or heuristics.
    """
    print(f"[Events] Detecting events for {len(frames)} frames...")
    events = ["drive", "turn_right", "stop"][:len(frames)]
    print(f"[Events] Events detected: {events}")
    return events

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import json

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
    """
    processor = Blip2Processor.from_pretrained(model_dir)
    model = Blip2ForConditionalGeneration.from_pretrained(model_dir).to(device)
    return processor, model

def detect_landmarks(frames, device):
    """
    Detect shop logos / signs using a fine‑grained YOLO model,
    then run OCR on each detection to capture the actual text.
    Returns a list of landmark strings per frame.
    """
    # ---- 2.1  Load a logo‑aware weight -----------------------------
    model_path = os.path.join('models', 'yolov8_logo.pt')
    if not os.path.exists(model_path):
        print("[YOLO] Downloading logo‑detection weights…")
        url = "https://universe.roboflow.com/...",
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(url, model_path)
    model = YOLO(model_path).to(device)
    reader = easyocr.Reader(['en'], gpu=device.startswith('cuda'))

    landmark_names = []
    print(f"[YOLO] Running detection on {len(frames)} frames…")
    for idx, frame in enumerate(tqdm(frames, unit='frame')):
        results = model(frame, verbose=False)
        r = results[0]

        if len(r.boxes) == 0:
            landmark_names.append("none")
            continue

        # ---- 2.2  Choose the highest‑confidence box ---------------
        box = r.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = model.model.names[int(box.cls[0])]

        # ---- 2.3  OCR on that crop --------------------------------
        crop = frame[y1:y2, x1:x2]
        ocr_txt = " ".join([t[1] for t in reader.readtext(crop)])
        final_name = f"{cls_name} {ocr_txt}".strip()
        landmark_names.append(final_name)

    return landmark_names

# utils.py
def generate_captions(frames, device, landmarks=None):
    """
    Use BLIP‑2 to caption each frame.
    If `landmarks` is supplied, prepend a text hint
    (e.g., “Scene contains: Tesco logo”) so the model
    is more likely to mention it.
    """
    print(f"[BLIP2] Loading BLIP2 model on {device}...")
    blip2_local_dir = os.path.join('models', 'blip2-opt-2.7b')
    processor = Blip2Processor.from_pretrained(blip2_local_dir)
    model      = Blip2ForConditionalGeneration.from_pretrained(blip2_local_dir).to(device)

    captions = []
    for idx, frame in enumerate(frames):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ---------- NEW LINES ----------
        hint = ""
        if landmarks and idx < len(landmarks):
            hint = f"Scene contains: {landmarks[idx]}. "
        # --------------------------------

        prompt_inputs = processor(images=img,
                                  text=hint,        # supply the hint
                                  return_tensors="pt").to(device)

        with torch.no_grad():
            ids = model.generate(**prompt_inputs, max_new_tokens=30)

        caption = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        captions.append(caption)
        print(f"[BLIP2] Frame {idx+1}: {caption}")

    print("[BLIP2] Caption generation complete.")
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
