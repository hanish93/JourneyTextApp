# app.py  — 2025‑07‑20
import os, json, subprocess, torch
from utils import (
    extract_frames, detect_events, detect_landmarks,
    generate_captions, summarise_journey, save_training_data
)

def run_pipeline(video):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Pipeline] device = {device}")

    frames = extract_frames(video)
    events = detect_events(frames)
    landmarks = detect_landmarks(frames, device)

    captions, long_text = generate_captions(frames, device, landmarks, events)
    steps = summarise_journey(events, landmarks, captions)

    for s in steps:
        print(f"[Step {s['step']:03}] {s['event']:<11} | {s['description']}")

    out = {"steps": steps, "long_summary": long_text}
    os.makedirs("outputs", exist_ok=True)
    op = os.path.join("outputs", os.path.basename(video) + "_summary.json")
    json.dump(out, open(op, "w"), indent=2)
    print(f"[Pipeline] summary → {op}")

    save_training_data(os.path.dirname(video), video, frames, events, landmarks, captions)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    run_pipeline(p.parse_args().video)
