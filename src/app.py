import torch
import cv2
import logging
from utils import (
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
    events, landmarks, captions, scenes, ocr_texts, brand_texts = [], [], [], [], [], [], []
    prev_gray = None
    landmark_model, landmark_ocr = models["landmark"]
    caption_processor, caption_model = models["caption"]
    scene_model, scene_classes = models["scene"]
    for frame in extract_frames(video_path):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        event = detect_event_for_frame(prev_gray, gray)
        events.append(event)
        with torch.no_grad():
            landmark, ocr_text, brand_text = detect_landmarks_for_frame(frame, landmark_model, landmark_ocr)
            landmarks.append(landmark)
            ocr_texts.append(ocr_text)
            brand_texts.append(brand_text)
            caption = generate_caption_for_frame(frame, caption_processor, caption_model, landmark, ocr_text)
            captions.append(caption)
            scene = classify_scene_for_frame(frame, scene_model, scene_classes)
            scenes.append(scene)
        prev_gray = gray
    torch.cuda.empty_cache()
    return events, landmarks, captions, scenes, ocr_texts, brand_texts

def run_pipeline(video_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {video_path} (device: {device}) ===\n")
    models = load_models(device)
    events, landmarks, captions, scenes, ocr_texts, brand_texts = process_frames(video_path, models)
    steps = summarise_journey(events, landmarks, captions, scenes, ocr_texts, brand_texts)
    for s in steps:
        print(f"[{s['step']:03}] {s['event']:<11} | Scene: {s['scene']:<20} | {s['description']}")
    torch.cuda.empty_cache()
    long_story = generate_long_summary(events, landmarks, captions, scenes, ocr_texts, brand_texts)
    print("\n―――――  Long‑form summary  ―――――\n")
    print(long_story)
    print("\n―――――――――――――――――――――――――――――\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    run_pipeline(p.parse_args().video)
