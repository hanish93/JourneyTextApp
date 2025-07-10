import os
from utils import extract_frames, detect_events, detect_landmarks, generate_captions, summarise_journey

def run_pipeline(video_path, device=None):
    """
    Main pipeline for journey summarisation.
    Steps:
    1. Extract frames from video
    2. Detect events
    3. Detect landmarks
    4. Generate captions
    5. Summarise journey
    6. Save output
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Pipeline] Using device: {device}")

    # 1. Extract frames from video
    print(f"[Pipeline] Extracting frames from {video_path}...")
    frames = extract_frames(video_path)
    print(f"[Pipeline] Extracted {len(frames)} frames.")

    # 2. Detect events (turns, stops, etc.)
    print(f"[Pipeline] Detecting events...")
    events = detect_events(frames)
    print(f"[Pipeline] Detected {len(events)} events.")

    # 3. Detect landmarks in frames
    print(f"[Pipeline] Detecting landmarks using YOLO...")
    landmarks = detect_landmarks(frames, device)
    print(f"[Pipeline] Landmarks detected for {len(landmarks)} frames.")

    # 4. Generate captions for each frame
    print(f"[Pipeline] Generating captions using BLIP2...")
    captions = generate_captions(frames, device)
    print(f"[Pipeline] Captions generated for {len(captions)} frames.")

    # 5. Summarise journey into structured JSON
    print(f"[Pipeline] Summarising journey...")
    summary = summarise_journey(events, landmarks, captions)

    # 5b. Generate long-form narrative summary
    print(f"[Pipeline] Generating long-form summary...")
    from utils import generate_long_summary
    long_summary = generate_long_summary(events, landmarks, captions)

    # 6. Save output (steps + long summary)
    output_obj = {
        "steps": summary,
        "long_summary": long_summary
    }
    os.makedirs('JourneyText/outputs', exist_ok=True)
    output_path = os.path.join('JourneyText/outputs', os.path.basename(video_path) + '_summary.json')
    with open(output_path, 'w') as f:
        import json
        json.dump(output_obj, f, indent=2)
    print(f"[Pipeline] Summary saved to {output_path}")

    # 7. Save training data for this video
    from utils import save_training_data
    save_training_data(video_path, frames, events, landmarks, captions)

    # 8. Trigger model training after each new video
    print("[Pipeline] Training BLIP2 model on new data...")
    import subprocess
    subprocess.run(["python", "JourneyText/train.py"])
    print("[Pipeline] Model training complete.")
