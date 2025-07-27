import os, cv2, torch, logging
from glob import glob

from .utils import (
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
    generate_long_summary
)

# ─── YOUR MANUAL FRAMES ───────────────────────────────────────────────────
FRAME_WHITELIST = [7, 10, 77, 96, 116]
FRAME_LABELS   = [
    "Tesco Express",
    "CREMA",
    "Townhall",
    "Vue",
    "Wool Pack Hub"
]
# ──────────────────────────────────────────────────────────────────────────

def load_models(device):
    print("[Models] Loading…")
    yolo, ocr = get_landmark_models(device)
    cap_proc, cap_mod = get_caption_models(device)
    scene_mod, scene_cls = get_scene_model(device)
    print("[Models] Done.\n")
    return {
        "yolo": yolo,
        "ocr": ocr,
        "cap_proc": cap_proc,
        "cap_mod": cap_mod,
        "scene_mod": scene_mod,
        "scene_cls": scene_cls,
    }

def process_frames(src, M):
    raw, sigs, lm, cap, scn, ocr = [], [], [], [], [], []
    prev_gray = None

    # choose frames
    if os.path.isdir(src):
        paths = sorted(glob(os.path.join(src, "*.jpg")))
        it = (cv2.imread(p) for p in paths)
    else:
        it = extract_frames(src)

    for idx, img in enumerate(it, start=1):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # — manual frame?
        if idx in FRAME_WHITELIST:
            label = FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            raw.append(f"passed {label}")
            sigs.append(None)
            lm.append("—")
            cap.append("—")
            scn.append("—")
            ocr.append("")
            prev_gray = gray
            continue

        # — detect motion —
        ev = detect_event_for_frame(prev_gray, gray)
        raw.append(ev)
        prev_gray = gray

        # — detect signal color —
        col = detect_signal_color(img, M["yolo"])
        sigs.append(col)

        # — detect landmarks & OCR —
        labs, txt = detect_landmarks_for_frame(img, M["yolo"], M["ocr"])
        lm.append(labs)
        ocr.append(txt)

        # — BLIP caption & scene classification —
        cap.append(generate_caption_for_frame(img, M["cap_proc"], M["cap_mod"], labs))
        scn.append(classify_scene_for_frame(img, M["scene_mod"], M["scene_cls"]))

    # — debounce both streams —
    motions = debounce_lane_changes(raw)
    signals = debounce_signals(sigs)

    return motions, signals, lm, cap, scn, ocr

def run_pipeline(src):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {src} (device={dev}) ===\n")
    M = load_models(dev)

    motions, signals, lm, cap, scn, ocr = process_frames(src, M)

    # — print a per‐frame table —
    print("STEP │ EVENT               │ SIGNAL  │ SCENE               │ CAPTION / LANDMARK / OCR")
    print("─────┼─────────────────────┼─────────┼─────────────────────┼────────────────────────────")
    for i, (ev, sig, scene, caption, labs, txt) in enumerate(
        zip(motions, signals, scn, cap, lm, ocr), start=1
    ):
        manual = "⚑" if ev.startswith("passed ") else " "
        sig_str = sig or "none"
        print(
            f"{i:3d}  │ {manual}{ev:<19} │ {sig_str:<7} │ {scene:<19} │ "
            f"{caption} / {labs} / '{txt}'"
        )

    # — final one‐line summary —
    print("\n――――――――  Final summary  ―――――――\n")
    print(generate_long_summary(motions))
    print("\n――――――――――――――――――――\n")

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Journey summariser")
    p.add_argument(
        "-i", "--input", required=True,
        help="Path to .mp4 video or folder of .jpg frames"
    )
    args = p.parse_args()
    run_pipeline(args.input)
