import torch, cv2, logging, collections
from utils import (
    extract_frames, detect_event_for_frame,
    get_landmark_models, detect_landmarks_for_frame,
    get_caption_models, generate_caption_for_frame,
    get_scene_model, classify_scene_for_frame,
    generate_long_summary, summarise_journey,
    salient, kind_of
)

def load_models(device):
    print("[Models] Loading...")
    (obj_model, sign_model), ocr = get_landmark_models(device)
    cap_proc, cap_mod            = get_caption_models(device)
    scene_mod, scene_classes     = get_scene_model(device)
    print("[Models] Done.")
    return {
        "obj":obj_model, "sign":sign_model, "ocr":ocr,
        "caption":(cap_proc, cap_mod),
        "scene":(scene_mod, scene_classes)
    }

def process_frames(video_path, m):
    events, landmarks, captions, scenes, ocr_texts = [], [], [], [], []
    sign_stats = {"shop":{}, "building":{}, "other":{}}
    prev_gray = None
    for frame in extract_frames(video_path):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        events.append(detect_event_for_frame(prev_gray, gray))
        with torch.no_grad():
            lm, ocr_txt = detect_landmarks_for_frame(frame, m["obj"], m["ocr"])
            landmarks.append(lm); ocr_texts.append(ocr_txt)
            captions.append(generate_caption_for_frame(frame, *m["caption"], lm))
            scenes.append(classify_scene_for_frame(frame, *m["scene"]))
            if m["sign"]:
                for sb in m["sign"](frame, conf=0.15, verbose=False)[0].boxes:
                    x1,y1,x2,y2 = map(int, sb.xyxy[0])
                    txt = " ".join(t[1] for t in m["ocr"].readtext(frame[y1:y2,x1:x2]))
                    if not salient(txt): continue
                    k = kind_of(txt); ent = sign_stats[k].setdefault(txt,[0,0.0])
                    ent[0]+=1; ent[1]+=float(sb.conf[0])
        prev_gray = gray
    torch.cuda.empty_cache()
    return events, landmarks, captions, scenes, ocr_texts, sign_stats

def run_pipeline(video):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {video} (device: {device}) ===\n")
    models = load_models(device)
    ev,lm,cap,scn,ocr,signs = process_frames(video, models)
    for s in summarise_journey(ev,lm,cap,scn,ocr):
        print(f"[{s['step']:03}] {s['event']:<11} | Scene: {s['scene']:<20} | {s['description']}")
    torch.cuda.empty_cache()
    story = generate_long_summary(ev,lm,cap,scn,ocr,signs)
    print("\n―――――  Long‑form summary  ―――――\n"); print(story)
    print("\n―――――――――――――――――――――――――――――\n")

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    p = argparse.ArgumentParser(); p.add_argument("--video", required=True)
    run_pipeline(p.parse_args().video)
