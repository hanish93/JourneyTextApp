# app.py – print‑only edition
import torch
from utils import (
    extract_frames,  detect_events, detect_landmarks,
    generate_captions, summarise_journey
)

def run_pipeline(video_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {video_path} (device: {device}) ===\n")

    # 1. frames
    frames = extract_frames(video_path)

    # 2. temporal events
    events = detect_events(frames)

    # 3. spatial landmarks
    landmarks = detect_landmarks(frames, device)

    # 4. captions + long narrative
    captions, long_story = generate_captions(
        frames, device, landmarks, events
    )

    # 5. step table
    steps = summarise_journey(events, landmarks, captions)
    for s in steps:
        print(f"[{s['step']:03}] {s['event']:<11} | {s['description']}")

    # 6. narrative
    print("\n―――――  Long‑form summary  ―――――\n")
    print(long_story)
    print("\n―――――――――――――――――――――――――――――\n")

if __name__ == "__main__":        # allows: python app.py --video …
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    run_pipeline(p.parse_args().video)
