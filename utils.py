import cv2, os, urllib.request
from ultralytics import YOLO
import torch
import easyocr
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import BitsAndBytesConfig

def extract_frames(video_path, fps=1, keyframe_threshold=0.1):
    """
    Extract frames from the input video at the specified FPS.
    Returns a list of frames (as numpy arrays) and a list of booleans indicating keyframes.
    A frame is considered a keyframe if the Mean Squared Error (MSE) between
    it and the previous frame is above a certain threshold.
    """
    frames = []
    is_keyframe = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
    count = 0
    success, image = cap.read()
    prev_frame = None
    print(f"[Frames] Starting frame extraction...")

    while success:
        if int(count % round(frame_rate / fps)) == 0:
            if prev_frame is not None:
                # Calculate MSE between current and previous frame
                mse = np.mean((image - prev_frame) ** 2)
                if mse > keyframe_threshold:
                    is_keyframe.append(True)
                else:
                    is_keyframe.append(False)
            else:
                # First frame is always a keyframe
                is_keyframe.append(True)

            frames.append(image)
            prev_frame = image
            print(f"[Frames] Extracted frame {len(frames)} (Keyframe: {is_keyframe[-1]})")

        success, image = cap.read()
        count += 1

    cap.release()
    print(f"[Frames] Extraction complete. Total frames: {len(frames)}, Keyframes: {sum(is_keyframe)}")
    return frames, is_keyframe

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


def detect_landmarks(frames, device, is_keyframe, conf_threshold=0.25):
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
        if not is_keyframe[idx]:
            names.append("no_change")
            continue
        try:
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
        except Exception as e:
            print(f"[Landmark Detection] Error processing frame {idx}: {e}")
            names.append("error")

    print(f"[Landmark Detection] Landmarks detected for {len([n for n in names if n not in ['none', 'no_change']])} out of {sum(is_keyframe)} keyframes.")
    return names


from model_utils import generate_captions, generate_long_summary

def summarise_journey(events, landmarks, captions):
    """
    Generate a step-by-step summary of the journey.
    """
    summary = []
    # Ensure all lists have the same length to avoid IndexError
    min_len = min(len(events), len(landmarks), len(captions))
    if min_len == 0:
        return []

    for i in range(min_len):
        event = events[i]
        landmark = landmarks[i]
        caption = captions[i]

        # Skip entries where an error occurred during processing
        if "error" in [event, landmark, caption] or "no_change" in [event, landmark, caption]:
            continue

        summary.append({
            "step": i + 1,
            "event": event,
            "description": f"{caption} Landmark: {landmark}."
        })
    return summary

import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def clean_caption(caption):
    """Clean a single caption string."""
    if not caption or caption in ["error", "no_change"]:
        return None
    caption = re.sub(r'[^\w\s.,!?-]', '', caption)
    caption = re.sub(r'(\b\w+\b)(?:\s+\1\b)+', r'\1', caption)  # remove repeated words
    caption = caption.strip()
    if len(caption) < 8 or len(caption.split()) < 2:
        return None
    if len(set(caption.lower().split())) == 1:
        return None
    return caption

def clean_landmarks(landmarks):
    """Clean and deduplicate a list of landmark strings."""
    cleaned_landmarks = []
    seen_lm = set()
    for l in landmarks:
        if not l or l in ["error", "none", "no_change"]:
            continue
        l = l.strip()
        if l and l not in seen_lm:
            cleaned_landmarks.append(l)
            seen_lm.add(l)
    return cleaned_landmarks

def format_summary_prompt(events, landmarks, captions):
    """Format the prompt for the summarization model."""
    prompt = (
        "You are an expert journey summarizer. Write a detailed, engaging, and coherent long-form summary (250-400 words) describing the journey. "
        "Include navigation, notable landmarks, road/traffic/weather conditions, and overall impressions.\n\n"
        "Driving events: " + ', '.join(events[:10]) + ".\n"
        "Notable landmarks: " + '; '.join(landmarks) + ".\n"
        "Scene highlights:\n- " + '\n- '.join(captions) + "\n\nSummary:"
    )
    return prompt


