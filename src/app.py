import os, cv2, torch
from glob import glob
from ultralytics import YOLO
from .utils import detect_signal_color, generate_full_summary

# your 5 labeled frames:
FRAME_LABELS = {
    7:  "Tesco Express",
    10: "CREMA",
    77: "Townhall",
    96: "Vue",
    116:"Wool Pack Hub"
}

def extract_frames(path):
    if os.path.isdir(path):
        files = sorted(glob(os.path.join(path,"*.jpg")))
        for fn in files:
            img = cv2.imread(fn)
            if img is not None:
                yield img
        return
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(round(fps))
    idx, ok, frame = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield frame
        ok, frame = cap.read()
        idx += 1
    cap.release()

def run_pipeline(src):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLOv8n on {device}…")
    model = YOLO("yolov8n").to(device).half()
    print("Done.\n")

    frames = list(extract_frames(src))
    total = len(frames)

    # 1) collect all signal colors
    signals = [detect_signal_color(f, model) for f in frames]

    # 2) note your “passed” frames
    passes = sorted(FRAME_LABELS.keys())

    # 3) emit per-frame debug table
    print("Frame │ Signal │ Note")
    print("──────┼────────┼────────────────")
    for i,sig in enumerate(signals, start=1):
        note = FRAME_LABELS[i] if i in FRAME_LABELS else ""
        print(f"{i:5d} │ {sig or 'none':6} │ {note}")

    # 4) final summary
    print("\n―― Final summary ――――\n")
    summary = generate_full_summary(passes, signals, FRAME_LABELS, total)
    print(summary)
    print("\n――――――――――――――――\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help=".mp4 or frames folder")
    args = p.parse_args()
    run_pipeline(args.input)
