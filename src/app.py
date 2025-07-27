import os, cv2, torch, logging
from glob import glob

from utils import (
    extract_frames,
    detect_event_for_frame,
    debounce_lane_changes,
    detect_signal_color,
    debounce_signals,
    get_landmark_models,
    detect_landmarks_for_frame,
    get_caption_models,
    generate_caption_for_frame,
    get_scene_model,
    classify_scene_for_frame,
    summarise_journey,
    generate_long_summary,
)

# ─── YOUR 5 MANUAL FRAMES ────────────────────────────────────────────────
FRAME_WHITELIST = [7,10,77,96,116]
FRAME_LABELS   = [
    "Tesco Express","CREMA","Townhall","Vue","Wool Pack Hub"
]
# ────────────────────────────────────────────────────────────────────────

def load_models(device):
    print("[Models] Loading…")
    yolo, ocr = get_landmark_models(device)
    cap_proc, cap_mod = get_caption_models(device)
    scene_mod, scene_cls = get_scene_model(device)
    print("[Models] Done.")
    return {
        "yolo": yolo, "ocr": ocr,
        "cap_proc": cap_proc, "cap_mod": cap_mod,
        "scene_mod": scene_mod, "scene_cls": scene_cls
    }

def process_frames(src, M):
    raw_motions, sigs = [], []
    lm, cap, scn, ocr = [], [], [], []
    prev_gray = None

    if os.path.isdir(src):
        files = sorted(glob(os.path.join(src,"*.jpg")))
        it = (cv2.imread(f) for f in files)
    else:
        it = extract_frames(src)

    for idx, frame in enumerate(it, start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # — manual frames —
        if idx in FRAME_WHITELIST:
            lbl = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            raw_motions.append(f"passed {lbl}")
            sigs.append(None)
            lm.append("none"); cap.append("…"); scn.append("…"); ocr.append("")
            prev_gray = gray
            continue

        # — motion —
        evt = detect_event_for_frame(prev_gray, gray)
        raw_motions.append(evt)
        prev_gray = gray

        # — signal color via HSV —
        color = detect_signal_color(frame, M["yolo"])
        sigs.append(color)

        # — landmarks + OCR —
        labels, txt = detect_landmarks_for_frame(frame, M["yolo"], M["ocr"])
        lm.append(labels); ocr.append(txt)

        # — BLIP caption & scene —
        cap.append(generate_caption_for_frame(frame, M["cap_proc"], M["cap_mod"], labels))
        scn.append(classify_scene_for_frame(frame, M["scene_mod"], M["scene_cls"]))

    # — debounce both streams —
    motions = debounce_lane_changes(raw_motions)
    signals = debounce_signals(sigs)

    return motions, signals, lm, cap, scn, ocr

def run_pipeline(src):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {src} (device={dev}) ===\n")
    M = load_models(dev)

    motions, signals, lm, cap, scn, ocr = process_frames(src, M)

    # per‑frame table
    for row in summarise_journey(motions, lm, cap, scn, ocr, signals):
        print(
            f"[{row['step']:03d}] {row['event']:<20} | "
            f"Signal={row['signal'] or 'none':<5} | "
            f"Scene={row['scene']:<18} | {row['description']}"
        )

    # final single sentence
    print("\n―――――  Final summary  ―――――\n")
    print(generate_long_summary(motions))
    print("\n――――――――――――――――――――\n")

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore",category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser()
    p.add_argument("-i","--input",required=True,
                   help="Path to .mp4 or folder of .jpg frames")
    args = p.parse_args()
    run_pipeline(args.input)
