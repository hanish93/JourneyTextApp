#!/usr/bin/env python3
import time
import os
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from app import run_pipeline

# Supported video formats for processing
SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv'}
# Directory to watch for new videos
ASSETS_DIR = os.path.join('JourneyText', 'assets')

def is_supported_video(path):
    _, ext = os.path.splitext(path)
    return ext.lower() in SUPPORTED_FORMATS

class VideoHandler(FileSystemEventHandler):
    """
    Handles new file creation/move events in the assets directory.
    If a supported video file is detected, triggers the journey summarisation pipeline.
    """
    def process(self, path):
        if not os.path.isfile(path):
            return
        if is_supported_video(path):
            print(f"[Watcher] New video detected: {path}")
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[Watcher] Using device: {device}")
                run_pipeline(path, device)
            except Exception as e:
                print(f"[Watcher] Error processing {path}: {e}")
        else:
            print(f"[Watcher] Ignored unsupported file: {path}")

    def on_created(self, event):
        if not event.is_directory:
            self.process(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self.process(event.dest_path)

def main():
    # Ensure the assets directory exists
    os.makedirs(ASSETS_DIR, exist_ok=True)
    event_handler = VideoHandler()
    observer = Observer()
    observer.schedule(event_handler, ASSETS_DIR, recursive=False)
    observer.start()
    print(f"[Watcher] Watching for new videos in {ASSETS_DIR}...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
