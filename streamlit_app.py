import streamlit as st
import os
import json
import subprocess
import tempfile
from utils import (
    extract_frames,
    detect_events,
    detect_landmarks,
    generate_captions,
    summarise_journey,
    generate_long_summary,
    save_training_data,
)

def run_pipeline_st(video_path, device=None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Using device: {device}")

    try:
        st.write("Extracting frames...")
        progress_bar = st.progress(0)
        log_container = st.empty()
        log_container.text("Starting frame extraction...")

        frames, is_keyframe = extract_frames(video_path)
        log_container.text(f"Extracted {len(frames)} frames, including {sum(is_keyframe)} keyframes.")
        progress_bar.progress(10)

        st.write("Detecting events...")
        events = detect_events(frames)
        log_container.text(f"Detected {len(events)} events.")
        progress_bar.progress(20)

        st.write("Detecting landmarks using YOLO...")
        landmarks = detect_landmarks(frames, device, is_keyframe=is_keyframe)
        log_container.text(f"Landmarks detected for {len(landmarks)} frames.")
        progress_bar.progress(40)

        st.write("Generating captions using BLIP2...")
        captions, journey_text = generate_captions(frames, device, is_keyframe=is_keyframe)
        log_container.text(f"Captions generated for {len(captions)} frames.")
        progress_bar.progress(70)

        st.write("Summarising journey...")
        summary = summarise_journey(events, landmarks, captions)
        progress_bar.progress(90)

        st.write("Generating long-form summary...")
        long_summary = generate_long_summary(events, landmarks, captions)
        progress_bar.progress(100)

        output = {"steps": summary, "long_summary": long_summary}

        return output

    except Exception as e:
        st.error(f"An error occurred during the pipeline: {e}")
        return {"error": str(e)}

st.title("Video Summarization App")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("Summarize Video"):
        with st.spinner("Summarizing..."):
            summary = run_pipeline_st(video_path)

        if "error" not in summary:
            st.subheader("Summary")
            st.write(summary.get("long_summary", "No long summary available."))

            st.subheader("Journey Steps")
            for step in summary.get("steps", []):
                st.write(f"**Step {step['step']}**: {step['event']} - {step['description']}")
        else:
            st.error(f"Failed to generate summary: {summary['error']}")
