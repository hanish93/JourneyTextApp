# app.py – print‑only edition
import torch
import cv2
import logging
from .utils import (
    extract_frames,
    detect_event_for_frame,
    get_landmark_models,
    detect_landmarks_for_frame,
    get_caption_models,
    generate_caption_for_frame,
    get_scene_model,
    classify_scene_for_frame,
    generate_long_summary,
    summarise_journey,
)

def load_models(device):
    """Load all the models needed for the pipeline."""
    print("[Models] Loading...")
    landmark_model, landmark_ocr = get_landmark_models(device)
    caption_processor, caption_model = get_caption_models(device)
    scene_model, scene_classes = get_scene_model(device)
    print("[Models] Done.")
    return {
        "landmark": (landmark_model, landmark_ocr),
        "caption": (caption_processor, caption_model),
        "scene": (scene_model, scene_classes),
    }

def process_frames(video_path, models):
    """Process each frame of the video and extract features."""
    events, landmarks, captions, scenes, ocr_texts = [], [], [], [], []
    prev_gray = None

    (landmark_model, landmark_ocr) = models["landmark"]
    (caption_processor, caption_model) = models["caption"]
    (scene_model, scene_classes) = models["scene"]

    for frame in extract_frames(video_path):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Event Detection
        event = detect_event_for_frame(prev_gray, gray)
        events.append(event)

        # 2. Landmark Detection
        landmark, ocr_text = detect_landmarks_for_frame(frame, landmark_model, landmark_ocr)
        landmarks.append(landmark)
        ocr_texts.append(ocr_text)

        # 3. Caption Generation
        caption = generate_caption_for_frame(frame, caption_processor, caption_model, landmark)
        captions.append(caption)

        # 4. Scene Classification
        scene = classify_scene_for_frame(frame, scene_model, scene_classes)
        scenes.append(scene)

        prev_gray = gray

    return events, landmarks, captions, scenes, ocr_texts

def run_pipeline(video_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {video_path} (device: {device}) ===\n")

    models = load_models(device)
    events, landmarks, captions, scenes, ocr_texts = process_frames(video_path, models)

    # 4. step table
    steps = summarise_journey(events, landmarks, captions, scenes, ocr_texts)
    for s in steps:
        print(f"[{s['step']:03}] {s['event']:<11} | Scene: {s['scene']:<20} | {s['description']}")

    # 5. narrative
    long_story = generate_long_summary(events, landmarks, captions, scenes, ocr_texts)
    print("\n―――――  Long‑form summary  ―――――\n")
    print(long_story)
    print("\n―――――――――――――――――――――――――――――\n")

if __name__ == "__main__":        # allows: python app.py --video …
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    run_pipeline(p.parse_args().video)
