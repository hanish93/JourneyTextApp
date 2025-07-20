# app.py – print‑only edition
"""End‑to‑end pipeline that
1. extracts frames from a dash‑cam style video,
2. detects the driving event (drive / stop / turn_left / turn_right),
3. recognises landmarks, vehicles, buildings and road details in each frame,
4. produces a short caption for every frame with BLIP‑2, and
5. generates both a step‑table and a long‑form narrative.

It now **also** persists a fully‑structured JSON artefact next to the video so
that `test.py`, `train.py`, or any downstream task can pick it up.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cv2
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)


from .utils import (
    detect_event_for_frame,
    detect_landmarks_for_frame,
    extract_frames,
    generate_caption_for_frame,
    generate_long_summary,
    get_caption_models,
    get_landmark_models,
    summarise_journey,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry‑point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(video_path: str, device: str | None = None) -> None:
    """Run the full journey‑summary pipeline on *video_path*.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    device : str | None, optional
        Explicit device override ("cpu", "cuda", "mps" …).  If *None* we pick
        **cuda** when available, otherwise **cpu**.
    """

    # ── env / device ────────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Starting journey summary for %s (device: %s)", video_path, device)

    # ── load models once (expensive ops) ────────────────────────────────────
    logging.info("Loading landmark detection models …")
    landmark_model, landmark_ocr = get_landmark_models(device)

    logging.info("Loading caption generation models …")
    caption_processor, caption_model = get_caption_models(device)

    # ── run over the video frame‑by‑frame ───────────────────────────────────
    events: list[str] = []
    landmarks: list[dict] = []
    captions: list[str] = []
    prev_gray = None
    frame_count = 0

    logging.info("Extracting frames and processing …")
    for frame in extract_frames(video_path):
        frame_count += 1
        logging.info("Processing frame %d", frame_count)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1 ⟩ event detection -------------------------------------------------
        event = detect_event_for_frame(prev_gray, gray)
        events.append(event)
        logging.info("  Event detected: %s", event)

        # 2 ⟩ landmark detection --------------------------------------------
        landmark = detect_landmarks_for_frame(frame, landmark_model, landmark_ocr)
        landmarks.append(landmark)
        logging.info("  Landmark detected: %s", landmark)

        # 3 ⟩ caption generation --------------------------------------------
        caption = generate_caption_for_frame(frame, caption_processor, caption_model, landmark)
        captions.append(caption)
        logging.info("  Caption generated: %s", caption)

        prev_gray = gray

    logging.info("Processed %d frames.", frame_count)

    # ── human‑readable output ----------------------------------------------
    steps = summarise_journey(events, landmarks, captions)

    print(f"\n=== Journey summary for {video_path} (device: {device}) ===\n")
    print(f"Frames processed: {frame_count}")
    print("\nStep‑by‑step journey:")
    for s in steps:
        print(f"[{s['step']:03}] {s['event']:<11} | {s['description']}")

    logging.info("Generating long‑form summary …")
    long_story = generate_long_summary(events, landmarks, captions, use_gpu=False)

    print("\n―――――  Long‑form summary  ―――――\n")
    print(long_story)
    print("\n―――――――――――――――――――――――――――――\n")
    print(f"Summary generated using device: {device}, frames: {frame_count}")

    # ── machine‑readable artefact (for test.py / train.py) ------------------
    summary_path = Path(f"{video_path}_summary.json")
    try:
        with summary_path.open("w", encoding="utf‑8") as fp:
            json.dump(
                {
                    "video": video_path,
                    "frames_processed": frame_count,
                    "events": events,
                    "landmarks": landmarks,
                    "captions": captions,
                    "long_summary": long_story,
                },
                fp,
                indent=2,
                ensure_ascii=False,
            )
        logging.info("Saved structured summary → %s", summary_path)
    except Exception as exc:
        logging.error("Failed to write summary JSON: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# CLI helper (allows `python app.py --video <file>`)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Console journey summariser")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--device",
        default=None,
        help="Force computation device (cpu / cuda / mps).  Defaults to auto‑detect.",
    )
    args = parser.parse_args()

    run_pipeline(args.video, args.device)
