import cv2, os, urllib.request
from ultralytics import YOLO
import torch
import easyocr
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

# --- Vivid detail detection using CLIP or similar model ---
def detect_vivid_details(frames, device):
    """
    Detect vivid scene attributes (weather, time of day, ambiance, etc.) for each frame.
    Uses a pre-trained model (e.g., CLIP) for zero-shot classification.
    Returns a list of vivid detail strings, one per frame.
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        from PIL import Image
        import cv2
        import numpy as np
        from tqdm import tqdm
    except ImportError:
        print("[VividDetails] Required packages not found. Please install transformers, torch, and tqdm.")
        return ["unknown details"] * len(frames)

    # Define candidate vivid detail prompts
    detail_prompts = [
        "sunny day", "rainy weather", "foggy", "nighttime", "dawn", "dusk", "urban street", "country road", "mountainous", "forest", "busy traffic", "empty road", "snowy", "autumn colors", "spring blossoms", "summer heat", "winter scene", "calm", "tense", "joyful", "mysterious"
    ]
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)

    vivid_details = []
    for frame in tqdm(frames, desc="[VividDetails] Detecting", unit="frame"):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=detail_prompts, images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits_per_image = model(**inputs).logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        best_idx = int(np.argmax(probs))
        vivid_details.append(detail_prompts[best_idx])
    return vivid_details

def extract_frames(video_path, fps=1):
    """
    Extract frames from the input video at the specified FPS.
    Returns a list of frames (as numpy arrays).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
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
    from tqdm import tqdm
    if len(frames) < 2:
        return ["drive"] * len(frames)

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    events = ["drive"] * len(frames)
    prev = gray_frames[0]

    for i in tqdm(range(1, len(gray_frames), stride), desc="[Events] Detecting", unit="frame"):
        try:
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
        except Exception as e:
            print(f"[Events] Error processing frame {i}: {e}")
            continue

    return events

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import json
import shutil

def get_or_download_model(model_name, local_dir, download_url=None, hf_repo=None, config_file="config.json"):
    """
    Checks if a model exists locally. If not, downloads it (from URL or HuggingFace repo).
    Returns the local directory path to the model, or the HuggingFace repo name if using HuggingFace.
    """
    if hf_repo:
        # For HuggingFace models, just return the repo name and let transformers manage the cache
        return hf_repo
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
    config_path = os.path.join(local_dir, config_file)
    if not os.path.exists(config_path):
        if download_url:
            # Download from direct URL
            print(f"[Model] Downloading model from {download_url} to {local_dir}")
            urllib.request.urlretrieve(download_url, os.path.join(local_dir, os.path.basename(download_url)))
    return local_dir

def save_training_data(training_data_dir, video_path, frames, events, landmarks, captions):
    """
    Save frames, events, landmarks, and the full journey summary as training data.
    Frames are not saved as images to save space, but you can extend this to save images if needed.
    """
    os.makedirs(training_data_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_json = os.path.join(training_data_dir, base + '.json')
    data = {
        "video": video_path,
        "events": events,
        "landmarks": landmarks,
        "journey_summary": captions if isinstance(captions, str) else "\n".join(captions)
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

def detect_landmarks(frames, device, conf_threshold=0.25):
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
    for idx, frame in enumerate(tqdm(frames, unit="frame")):
        if frame is None or not hasattr(frame, 'shape'):
            print(f"[Landmark Detection] Warning: Frame {idx} is invalid or empty.")
            names.append("none")
            continue
        r = model(frame, verbose=False, conf=conf_threshold)[0]
        if len(r.boxes) == 0:
            print(f"[Landmark Detection] No boxes detected in frame {idx}.")
            names.append("none")
            continue
        # Collect all detected classes with confidence and OCR if possible
        detected_info = []
        for box in r.boxes:
            cls = model.model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            txt = " ".join(t[1] for t in reader.readtext(crop))
            if txt:
                detected_info.append(f"{cls} ({conf:.2f}) [{txt}]")
            else:
                detected_info.append(f"{cls} ({conf:.2f})")
        detected_str = ", ".join(detected_info)
        print(f"[Landmark Detection] Frame {idx} detected: {detected_str}")
        names.append(detected_str if detected_str else "none")

    print(f"[Landmark Detection] Landmarks detected for {len([n for n in names if n != 'none'])} out of {len(frames)} frames.")
    return names


# utils.py
def generate_captions(frames, device, landmarks=None, events=None):
    """
    Generate a complete journey summary using the frames, landmarks, and events.
    This now returns a single narrative summary instead of per-frame captions.
    """
    import os, shutil, cv2, torch
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from tqdm import tqdm

    print(f"[BLIP2] Using device: {device}")
    repo = "Salesforce/blip2-flan-t5-xl"
    model_id = get_or_download_model(repo, None, hf_repo=repo)
    processor = Blip2Processor.from_pretrained(model_id, use_fast=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
    ).eval()

    per_frame_captions = []
    for i, frame in enumerate(tqdm(frames, desc="[BLIP2] Generating captions", unit="frame")):
        try:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Downscale image to max 512x512 for memory efficiency
            max_dim = 512
            if img.width > max_dim or img.height > max_dim:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            hint = f"Scene contains: {landmarks[i]}. " if landmarks and i < len(landmarks) else ""
            inputs = processor(images=img, text=hint, return_tensors="pt").to(device)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=30)
            caption = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            per_frame_captions.append(caption)
            if device.startswith("cuda"):
                import torch
                torch.cuda.empty_cache()
        except RuntimeError as e:
            import torch
            if "out of memory" in str(e) and device.startswith("cuda"):
                print(f"[BLIP2] CUDA OOM at frame {i}, retrying on CPU...")
                torch.cuda.empty_cache()
                try:
                    model_cpu = model.to("cpu")
                    inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
                    with torch.no_grad():
                        ids = model_cpu.generate(**inputs_cpu, max_new_tokens=30)
                    caption = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
                    per_frame_captions.append(caption)
                except Exception as e2:
                    print(f"[BLIP2] Error on CPU for frame {i}: {e2}")
                    per_frame_captions.append("")
                model.to(device)  # Move back to original device
            else:
                print(f"[BLIP2] Error generating caption for frame {i}: {e}")
                per_frame_captions.append("")
        except Exception as e:
            print(f"[BLIP2] Error generating caption for frame {i}: {e}")
            per_frame_captions.append("")

    if events is None:
        events = ["drive"] * len(frames)
    journey_text = generate_long_summary(events, landmarks or ["none"]*len(frames), per_frame_captions)
    return per_frame_captions, journey_text



def summarise_journey(events, landmarks, captions, vivid_details=None):
    summary = []
    vivid_details = vivid_details or ["unknown details"] * len(events)
    for i, (event, landmark, caption, vivid) in enumerate(zip(events, landmarks, captions, vivid_details)):
        vivid_desc = (
            f"As the journey unfolds, step {i+1} features a moment of '{event}'. "
            f"{caption} "
            f"Notable sights: {landmark if landmark != 'none' else 'no major landmarks detected'}. "
            f"Scene details: {vivid}. "
            f"The atmosphere is vibrant and immersive, drawing the traveler deeper into the experience."
        )
        summary.append({
            "step": i+1,
            "event": event,
            "description": vivid_desc
        })
    return summary

def generate_long_summary(events, landmarks, captions, vivid_details=None):
    """
    Create a 250‑400‑word journey narrative.
    Ensures the prompt never exceeds the model’s max positional length.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import os, json, torch

    vivid_details = vivid_details or ["unknown details"] * len(events)

    # --- 1. choose an inexpensive LM (runs in 2‑3 GB VRAM) ------------
    repo = os.getenv("JOURNEY_SUMMARY_MODEL", "facebook/bart-large-cnn")
    cache_dir = os.path.join("models", repo.replace("/", "_"))
    tok  = AutoTokenizer.from_pretrained(repo, cache_dir=cache_dir)
    lm   = AutoModelForCausalLM.from_pretrained(repo, cache_dir=cache_dir)

    # model limit (Bart‑large‑cnn = 1024)
    ctx_max      = getattr(lm.config, "max_position_embeddings", 1024)
    out_tokens   = 256                         # we’ll ask for this many new tokens
    prompt_limit = ctx_max - out_tokens

    # --- 2. pack the JSON chunks, truncate until tokens fit ------------
    base_hdr = (
        "You are an expert journey summarizer and storyteller. "
        "Compose a vivid, immersive, and emotionally engaging 250‑400‑word narrative of the route. "
        "Use rich, sensory language to bring the journey to life, describing sights, sounds, and feelings. "
        "Highlight key events, landmarks, and vivid scene details, and make the reader feel as if they are experiencing the journey firsthand.\n\n"
    )

    def pack(ev, lmks, caps, vivid):
        return (
            base_hdr +
            f"Events: {json.dumps(ev)}\n"
            f"Landmarks: {json.dumps(lmks)}\n"
            f"Captions: {json.dumps(caps)}\n"
            f"VividDetails: {json.dumps(vivid)}\n\nSummary:"
        )

    ev, lmks, caps, vivid = events[:], landmarks[:], captions[:], vivid_details[:]
    while True:
        prompt = pack(ev, lmks, caps, vivid)
        if len(tok(prompt).input_ids) <= prompt_limit:
            break
        # drop 10 % from the longest list until it fits
        largest = max((len(ev), "ev"), (len(lmks), "lm"), (len(caps), "cap"), (len(vivid), "vivid"))[1]
        if   largest == "ev":    ev    = ev[   : max(1, int(len(ev)*0.9)) ]
        elif largest == "lm":    lmks  = lmks[ : max(1, int(len(lmks)*0.9))]
        elif largest == "cap":   caps  = caps[ : max(1, int(len(caps)*0.9))]
        else:                    vivid = vivid[: max(1, int(len(vivid)*0.9))]

    # --- 3. generate ---------------------------------------------------
    device = 0 if torch.cuda.is_available() else -1
    gen = pipeline("text-generation", model=lm, tokenizer=tok, device=device)
    text = gen(
        prompt,
        max_new_tokens=out_tokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        no_repeat_ngram_size=4,
    )[0]["generated_text"]

    summary = text[len(prompt):].strip()
    # final safety trim to 1000 words
    words = summary.split()
    return " ".join(words[:1000]) + ("…" if len(words) > 1000 else "")

