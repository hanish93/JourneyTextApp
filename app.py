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
    import tempfile
    import cv2

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pipeline] Using device: {device}")

    try:
        print(f"[Pipeline] Extracting frames from {video_path}...")

        all_is_keyframe = []
        all_events = []
        all_landmarks = []
        all_captions = []
        frame_paths = []

        with tempfile.TemporaryDirectory() as temp_dir:
            frame_generator = extract_frames(video_path, batch_size=10)
            frame_count = 0

            for frames_batch, keyframe_batch in frame_generator:
                for i, frame in enumerate(frames_batch):
                    frame_filename = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    frame_paths.append(frame_filename)
                    frame_count += 1

                all_is_keyframe.extend(keyframe_batch)

                print("[Pipeline] Detecting events...")
                events = detect_events(frames_batch)
                all_events.extend(events)
                print(f"[Pipeline] Detected {len(events)} events.")

                print("[Pipeline] Detecting landmarks using YOLO...")
                landmarks = detect_landmarks(frames_batch, device, is_keyframe=keyframe_batch)
                all_landmarks.extend(landmarks)
                print(f"[Pipeline] Landmarks detected for {len(landmarks)} frames.")

                print("[Pipeline] Generating captions using BLIP2...")
                caption = generate_captions(frames_batch, device, is_keyframe=keyframe_batch)
                all_captions.append(caption)
                print(f"[Pipeline] Caption generated for batch.")

            print(f"[Pipeline] Extracted {frame_count} frames, including {sum(all_is_keyframe)} keyframes.")

            print("[Pipeline] Summarising journey...")
            summary = summarise_journey(all_events, all_landmarks, all_captions)

            for step in summary:
                print(
                    f"[Step {step['step']:03}] {step['event']:<11} | {step['description']}"
                )

            print("[Pipeline] Generating long-form summary...")
            long_summary = generate_long_summary(all_events, all_landmarks, all_captions)

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
                training_dir, video_path, frame_paths, all_events, all_landmarks, all_captions
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
        print(f"An error occurred during the pipeline: {e}")
        # Optionally, save a partial or error summary
        output = {"error": str(e)}
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join(
            "outputs", os.path.basename(video_path) + "_error.json"
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Error summary saved to {out_path}")
