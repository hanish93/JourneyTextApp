import torch, cv2, collections
from utils import (
    extract_frames,
    detect_event_for_frame,
    get_landmark_models,
    detect_landmarks_for_frame,
    get_caption_models,
    generate_caption_for_frame,
    get_scene_model,
    classify_scene_for_frame,
    generate_long_summary,
    summarise_journey,
    salient, kind_of,
)

def load_models(device):
    print("[Models] Loading...")
    (obj_model, sign_model), ocr = get_landmark_models(device)
    caption_proc, caption_mod    = get_caption_models(device)
    scene_mod, scene_classes     = get_scene_model(device)
    print("[Models] Done.")
    return {
        "obj":  obj_model,
        "sign": sign_model,
        "ocr":  ocr,
        "caption": (caption_proc, caption_mod),
        "scene": (scene_mod, scene_classes),
    }

def process_frames(video_path, models):
    events, landmarks, captions, scenes, ocr_texts = [], [], [], [], []
    sign_stats = {"shop": {}, "building": {}, "other": {}}

    prev_gray = None
    obj_model = models["obj"]
    sign_model = models["sign"]
    ocr = models["ocr"]
    caption_proc, caption_mod = models["caption"]
    scene_mod, scene_classes  = models["scene"]

    for frame in extract_frames(video_path):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        events.append(detect_event_for_frame(prev_gray, gray))

        with torch.no_grad():
            # main object / landmark detector
            landmark, ocr_text = detect_landmarks_for_frame(frame, obj_model, ocr)
            landmarks.append(landmark)
            ocr_texts.append(ocr_text)

            # caption & scene
            captions.append(generate_caption_for_frame(frame, caption_proc, caption_mod, landmark))
            scenes.append(classify_scene_for_frame(frame, scene_mod, scene_classes))

            # sign‑specific detector (if weights present)
            if sign_model:
                sbxs = sign_model(frame, conf=0.15, verbose=False)[0]
                for sb in sbxs:
                    x1,y1,x2,y2 = map(int, sb.xyxy[0])
                    txt = " ".join(t[1] for t in ocr.readtext(frame[y1:y2, x1:x2]))
                    if not salient(txt):
                        continue
                    kind = kind_of(txt)
                    entry = sign_stats[kind].setdefault(txt, [0,0.0])
                    entry[0] += 1                   # frames_seen
                    entry[1] += float(sb.conf[0])   # cumulative conf

        prev_gray = gray

    torch.cuda.empty_cache()
    return events, landmarks, captions, scenes, ocr_texts, sign_stats

def run_pipeline(video_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {video_path} (device: {device}) ===\n")
    models = load_models(device)

    results = process_frames(video_path, models)
    events, landmarks, captions, scenes, ocr_texts, sign_stats = results

    for s in summarise_journey(events, landmarks, captions, scenes, ocr_texts):
        print(f"[{s['step']:03}] {s['event']:<11} | Scene: {s['scene']:<20} | {s['description']}")

    torch.cuda.empty_cache()
    long_story = generate_long_summary(events, landmarks, captions, scenes, ocr_texts, sign_stats)
    print("\n―――――  Long‑form summary  ―――――\n")
    print(long_story)
    print("\n―――――――――――――――――――――――――――――\n")

if __name__ == "__main__":
    import argparse, warnings, logging
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    run_pipeline(p.parse_args().video)
