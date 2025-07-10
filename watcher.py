import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from app import run_pipeline

# Supported video formats for processing
SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv'}
# Directory to watch for new videos
ASSETS_DIR = os.path.join('JourneyText', 'assets')

class VideoHandler(FileSystemEventHandler):
    """
    Handles new file creation events in the assets directory.
    If a supported video file is detected, triggers the journey summarisation pipeline.
    """
    def on_created(self, event):
        # Ignore directories
        if event.is_directory:
            return
        _, ext = os.path.splitext(event.src_path)
        # Check if the file is a supported video format
        if ext.lower() in SUPPORTED_FORMATS:
            print(f"[Watcher] New video detected: {event.src_path}")
            try:
                import torch
                # Auto-select device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[Watcher] Using device: {device}")
                # Run the pipeline for the new video
                run_pipeline(event.src_path, device)
            except Exception as e:
                print(f"[Watcher] Error processing {event.src_path}: {e}")
        else:
            print(f"[Watcher] Ignored unsupported file: {event.src_path}")

def main():
    # Ensure the assets directory exists
    os.makedirs(ASSETS_DIR, exist_ok=True)
    event_handler = VideoHandler()
    observer = Observer()
    # Watch the assets directory for new files
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
