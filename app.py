import os
import json
import subprocess
from utils import (
    extract_frames,
    detect_events,
    detect_landmarks,
    summarise_journey,
    save_training_data,
)
from model_utils import generate_captions, generate_long_summary


def run_pipeline(video_path, device=None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pipeline] Using device: {device}")

    try:
        print(f"[Pipeline] Extracting frames from {video_path}...")
        frames, is_keyframe = extract_frames(video_path)
        print(f"[Pipeline] Extracted {len(frames)} frames, including {sum(is_keyframe)} keyframes.")

        print("[Pipeline] Detecting events...")
        events = detect_events(frames)
        print(f"[Pipeline] Detected {len(events)} events.")

        print("[Pipeline] Detecting landmarks using YOLO...")
        landmarks = detect_landmarks(frames, device, is_keyframe=is_keyframe)
        print(f"[Pipeline] Landmarks detected for {len(landmarks)} frames.")

        print("[Pipeline] Generating captions using BLIP2...")
        captions, journey_text = generate_captions(frames, device, landmarks=landmarks, is_keyframe=is_keyframe)
        print(f"[Pipeline] Captions generated for {len(captions)} frames.")

        print("[Pipeline] Summarising journey...")
        summary = summarise_journey(events, landmarks, captions)

        for step in summary:
            print(
                f"[Step {step['step']:03}] {step['event']:<11} | {step['description']}"
            )

        print("[Pipeline] Generating long-form summary...")
        long_summary = generate_long_summary(events, landmarks, captions)

        output = {"steps": summary, "long_summary": long_summary}
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join(
            "outputs", os.path.basename(video_path) + "_summary.json"
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[Pipeline] Summary saved to {out_path}")

        training_dir = os.path.dirname(video_path)
        save_training_data(
            training_dir, video_path, frames, events, landmarks, captions
        )

        print("[Pipeline] Training BLIP2 model on new data...")
        subprocess.run(
            [
                "python",
                "train.py",
                "--training_data_dir",
                training_dir,
                "--model_dir",
                "models/blip2-flan-t5-xl",
            ],
            check=True,
        )
        print("[Pipeline] Model training complete.")
    except Exception as e:
        print(f"[Pipeline] An error occurred during the pipeline: {e}")
        # Optionally, save a partial or error summary
        output = {"error": str(e)}
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join(
            "outputs", os.path.basename(video_path) + "_error.json"
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[Pipeline] Error summary saved to {out_path}")
