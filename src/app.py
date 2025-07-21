import cv2, torch, pathlib, logging
from utils import (
    extract_frames, detect_event,
    load_detectors, get_landmarks,
    load_captioner, caption_frame,
    summarise_journey
)

def process_clip(path, models, device, fps=1):
    ev, caps = [], []
    whitelist = set()

    for f in extract_frames(path, fps=fps):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        ev.append(detect_event(process_clip.prev_gray, gray))
        process_clip.prev_gray = gray

        names = get_landmarks(f, models["det"], models["ocr"])
        whitelist.update(names)
        caps.append(caption_frame(
            f, models["cap_proc"], models["cap_mod"],
            " ".join(names) if names else ""
        ))

    sentences = [f"I {e.replace('_',' ')} and {c.lower()}" for e,c in zip(ev,caps)]
    return summarise_journey(sentences, sorted(whitelist))

process_clip.prev_gray = None   # static attribute

def run(target):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det, ocr = load_detectors(device)
    cap_proc, cap_mod = load_captioner(device)
    models = dict(det=det, ocr=ocr, cap_proc=cap_proc, cap_mod=cap_mod)

    p = pathlib.Path(target)
    files = [p] if p.is_file() else sorted(p.glob("*.mp4"))

    for vid in files:
        process_clip.prev_gray = None  # reset per video
        print("\n———", vid, "———")
        print(process_clip(str(vid), models, device))

# legacy alias for cli.py
run_pipeline = run

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    arg = argparse.ArgumentParser()
    arg.add_argument("--video", required=True,
                     help="Path to .mp4 or directory of .mp4 files")
    run(arg.parse_args().video)
