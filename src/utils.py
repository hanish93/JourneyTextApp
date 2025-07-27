# src/utils.py

import os
import cv2

def frames(video_path: str, fps: int = 1):
    """
    Yield frames from a video at approximately `fps` frames per second.
    Or, if `video_path` is a directory of JPGs, yield those in sorted order.
    """
    if os.path.isdir(video_path):
        # directory of JPGs
        for fname in sorted(os.listdir(video_path)):
            if fname.lower().endswith(".jpg"):
                yield cv2.imread(os.path.join(video_path, fname))
        return

    # otherwise assume it's a video file
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(original_fps / fps)))
    idx = 0
    ok, frame = cap.read()
    while ok:
        if idx % step == 0:
            yield frame
        ok, frame = cap.read()
        idx += 1
    cap.release()
