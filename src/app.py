# src/app.py  –  diary‑generator pipeline
import torch, cv2, logging, time
from utils import (
    extract_frames, detect_event,
    get_models, detect_static,
    get_scene_model, classify_scene,
    get_caption_models, caption,
    summarise, journey_table,
    salient, kind_of, preprocess_crop
)

# ───────────── load all neural models once ───────────────────────────────
def load_all(device: str):
    (obj, sign), ocr = get_models(device)
    scene_mod, scene_cls, scene_tf = get_scene_model(device)
    cap_proc, cap_mod = get_caption_models(device)
    return {
        "obj": obj, "sign": sign, "ocr": ocr,
        "scene_mod": scene_mod, "scene_cls": scene_cls, "scene_tf": scene_tf,
        "cap_proc": cap_proc, "cap_mod": cap_mod,
    }

# ───────────── per‑video processing ──────────────────────────────────────
def process_video(video_path: str, M: dict):
    events, landmarks, captions, scenes, ocr_texts = [], [], [], [], []
    sign_stats = {"shop": {}, "building": {}, "other": {}}
    conf_scores = []

    # collect adaptive threshold statistics on first 50 frames
    first_frames = list(extract_frames(video_path))[:50]
    for f in first_frames:
        result = M["obj"](f, verbose=False, conf=0.1)[0]
        conf_scores.extend([float(c) for c in result.boxes.conf.cpu()])
    adaptive_conf = (np.median(conf_scores) * 0.60) if conf_scores else 0.25

    prev_gray = None
    start_time = time.time()

    for frame in extract_frames(video_path):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        events.append(detect_event(prev_gray, gray))

        # persistent landmark detection
        lm, txt = detect_static(frame, M["obj"], M["ocr"], adaptive_conf)
        landmarks.append(lm)
        ocr_texts.append(txt)

        # caption + scene
        captions.append(caption(frame, M["cap_proc"], M["cap_mod"], lm))
        scenes.append(classify_scene(frame, M["scene_mod"], M["scene_cls"], M["scene_tf"]))

        # sign‑specific detections (optional model)
        if M["sign"]:
            sbxs = M["sign"](frame, conf=0.15, verbose=False)[0]
            for b in sbxs.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cropped = preprocess_crop(frame[y1:y2, x1:x2])
                txt_sign = " ".join(M["ocr"].readtext(cropped, detail=0))
                if not salient(txt_sign):
                    continue
                bucket = kind_of(txt_sign)
                entry = sign_stats[bucket].setdefault(txt_sign, [0, 0.0])
                entry[0] += 1                   # frames_seen
                entry[1] += float(b.conf[0])    # cumulative confidence

        prev_gray = gray

    processing_secs = time.time() - start_time
    return events, landmarks, captions, scenes, ocr_texts, sign_stats, processing_secs

# ───────────── main orchestration function ───────────────────────────────
def run(video_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {video_path} (device: {device}) ===\n")

    models = load_all(device)
    ev, lm, cap, scn, ocr, stats, secs = process_video(video_path, models)

    for row in journey_table(ev, lm, cap, scn, ocr):
        print(f"[{row['step']:03}] {row['event']:<11} | Scene: {row['scene']:<20} | {row['description']}")

    print("\n―――――  Long‑form summary  ―――――\n")
    print(summarise(ev, lm, cap, scn, ocr, stats, secs))
    print("\n―――――――――――――――――――――――――――――\n")

# backward‑compat alias so scripts that import run_pipeline keep working
run_pipeline = run

# ───────────── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Generate a journey diary from a video")
    parser.add_argument("--video", required=True, help="Path to the video file")
    args = parser.parse_args()

    run(args.video)
