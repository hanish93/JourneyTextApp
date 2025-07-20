# src/app.py  – journey‑diary pipeline
import torch, cv2, logging, pathlib
from utils import (
    extract_frames, detect_event_for_frame,
    get_landmark_models, detect_landmarks_for_frame,
    get_caption_models, generate_caption_for_frame,
    get_scene_model, classify_scene_for_frame,
    generate_long_summary, summarise_journey,
    salient, kind_of
)

OUT_DIR = pathlib.Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


# ───────────── model loader (GPU / CPU) ──────────────────────────────────
def load_models(device: str):
    (obj, sign), ocr = get_landmark_models(device)
    cap_proc, cap_mod = get_caption_models(device)
    scene_mod, scene_classes, scene_tf = get_scene_model(device)   # ← 3 values
    return {
        "obj": obj, "sign": sign, "ocr": ocr,
        "cap_proc": cap_proc, "cap_mod": cap_mod,
        "scene_mod": scene_mod, "scene_classes": scene_classes,
        "scene_tf": scene_tf,                         # include the transform
    }


# ───────────── process a single video ───────────────────────────────────
def run_single(video_path: str, M: dict, device: str):
    print(f"\n=== Journey summary for {video_path} (device: {device}) ===")

    ev, lm, cap, scn, ocr_txt = [], [], [], [], []
    sign_stats = {"shop": {}, "building": {}, "other": {}}
    prev = None

    for frm in extract_frames(video_path):
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        ev.append(detect_event_for_frame(prev, gray))

        # landmark / OCR
        lmk, txt = detect_landmarks_for_frame(frm, M["obj"], M["ocr"])
        lm.append(lmk)
        ocr_txt.append(txt)

        # caption + scene
        cap.append(generate_caption_for_frame(frm, M["cap_proc"], M["cap_mod"], lmk))
        scn.append(
            classify_scene_for_frame(
                frm, M["scene_mod"], M["scene_classes"], M["scene_tf"]
            )
        )

        # optional sign detector
        if M["sign"]:
            for b in M["sign"](frm, conf=0.15, verbose=False)[0].boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                s_txt = " ".join(t[1] for t in M["ocr"].readtext(frm[y1:y2, x1:x2]))
                if not salient(s_txt):
                    continue
                bucket = kind_of(s_txt)
                entry = sign_stats[bucket].setdefault(s_txt, [0, 0.0])
                entry[0] += 1  # frames seen
                entry[1] += float(b.conf[0])

        prev = gray

    # generate long summary & save
    summary = generate_long_summary(ev, lm, cap, scn, ocr_txt, sign_stats)
    out_path = OUT_DIR / f"{pathlib.Path(video_path).stem}_journey.txt"
    out_path.write_text(summary)
    print(f"[Saved] → {out_path}")


# ───────────── public runner (file or folder) ───────────────────────────
def run(video_or_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device)

    # accept single file or directory
    if pathlib.Path(video_or_dir).is_dir():
        videos = sorted(str(p) for p in pathlib.Path(video_or_dir).glob("*.mp4"))
        if not videos:
            print("No .mp4 files found in", video_or_dir)
            return
    else:
        videos = [video_or_dir]

    for vid in videos:
        run_single(vid, models, device)


# backward‑compat alias
run_pipeline = run


# ───────────── CLI entry point ──────────────────────────────────────────
if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Generate journey diaries from video(s)")
    parser.add_argument("--video", required=True,
                        help="Path to a .mp4 file or directory containing .mp4 files")
    args = parser.parse_args()
    run(args.video)
