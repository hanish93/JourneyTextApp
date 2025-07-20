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
    generate_long_summary,
    summarise_journey,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def run_pipeline(video_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Starting journey summary for {video_path} (device: {device})")

    # Load models once
    logging.info("Loading landmark detection models...")
    landmark_model, landmark_ocr = get_landmark_models(device)
    logging.info("Loading caption generation models...")
    caption_processor, caption_model = get_caption_models(device)

    events, landmarks, captions = [], [], []
    prev_gray = None
    frame_count = 0

    logging.info("Extracting frames and processing...")
    for frame in extract_frames(video_path):
        frame_count += 1
        logging.info(f"Processing frame {frame_count}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Event Detection
        event = detect_event_for_frame(prev_gray, gray)
        events.append(event)
        logging.info(f"  Event detected: {event}")

        # 2. Landmark Detection
        landmark = detect_landmarks_for_frame(frame, landmark_model, landmark_ocr)
        landmarks.append(landmark)
        logging.info(f"  Landmark detected: {landmark}")

        # 3. Caption Generation
        caption = generate_caption_for_frame(frame, caption_processor, caption_model, landmark)
        captions.append(caption)
        logging.info(f"  Caption generated: {caption}")

        prev_gray = gray

    logging.info(f"Processed {frame_count} frames.")

    # 4. step table
    steps = summarise_journey(events, landmarks, captions)
    print(f"\n=== Journey summary for {video_path} (device: {device}) ===\n")
    print(f"Frames processed: {frame_count}")
    print("\nStep-by-step journey:")
    for s in steps:
        print(f"[{s['step']:03}] {s['event']:<11} | {s['description']}")

    # 5. narrative
    logging.info("Generating long-form summary...")
    long_story = generate_long_summary(events, landmarks, captions)
    print("\n―――――  Long‑form summary  ―――――\n")
    print(long_story)
    print("\n―――――――――――――――――――――――――――――\n")
    print(f"Summary generated using device: {device}, frames: {frame_count}")

if __name__ == "__main__":        # allows: python app.py --video …
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    run_pipeline(p.parse_args().video)
