import torch, cv2, logging, time, os, pathlib
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

# ---------- load models once per process --------------------------------
def load_models(device: str):
    (obj, sign), ocr = get_landmark_models(device)
    cap_proc, cap_mod = get_caption_models(device)
    scene_mod, scene_classes = get_scene_model(device)
    return {
        "obj": obj, "sign": sign, "ocr": ocr,
        "cap_proc": cap_proc, "cap_mod": cap_mod,
        "scene_mod": scene_mod, "scene_classes": scene_classes,
    }

# ---------- run on a single video ---------------------------------------
def run_single(video_path: str, models, device: str):
    print(f"\n=== Journey summary for {video_path} (device: {device}) ===\n")

    ev, lm, cap, scn, ocr_txt = [], [], [], [], []
    sign_stats = {"shop": {}, "building": {}, "other": {}}
    prev = None

    for frm in extract_frames(video_path):
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        ev.append(detect_event_for_frame(prev, gray))

        with torch.no_grad():
            lmk, txt = detect_landmarks_for_frame(frm, models["obj"], models["ocr"])
            lm.append(lmk)
            ocr_txt.append(txt)
            cap.append(generate_caption_for_frame(frm, models["cap_proc"], models["cap_mod"], lmk))
            scn.append(classify_scene_for_frame(frm, models["scene_mod"], models["scene_classes"]))
            if models["sign"]:
                for b in models["sign"](frm, conf=0.15, verbose=False)[0].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    s_txt = " ".join(t[1] for t in models["ocr"].readtext(frm[y1:y2, x1:x2]))
                    if not salient(s_txt):
                        continue
                    bucket = kind_of(s_txt)
                    entry = sign_stats[bucket].setdefault(s_txt, [0, 0.0])
                    entry[0] += 1
                    entry[1] += float(b.conf[0])

        prev = gray

    summary = generate_long_summary(ev, lm, cap, scn, ocr_txt, sign_stats)

    # print short log to console
    print(summary[:300] + ("…" if len(summary) > 300 else ""))
    # write full summary to disk
    out_file = OUT_DIR / (pathlib.Path(video_path).stem + "_journey.txt")
    out_file.write_text(summary)
    print(f"[Saved] → {out_file}")

# ---------- public API ---------------------------------------------------
def run(video_or_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device)

    if os.path.isdir(video_or_dir):
        videos = sorted(
            str(p) for p in pathlib.Path(video_or_dir).glob("*.mp4")
        )
        if not videos:
            print("No .mp4 files found inside", video_or_dir)
            return
    else:
        videos = [video_or_dir]

    for vid in videos:
        run_single(vid, models, device)

# keep the legacy name so older scripts still import run_pipeline
run_pipeline = run

# ---------- CLI ----------------------------------------------------------
if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description="Generate diary summaries for videos or a folder")
    p.add_argument("--video", required=True,
                   help="Path to a .mp4 file *or* a directory containing .mp4 files")
    args = p.parse_args()
    run(args.video)
