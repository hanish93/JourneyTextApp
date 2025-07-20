import torch, cv2, logging
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
    (obj, sign), ocr = get_landmark_models(device)
    cap_proc, cap_mod = get_caption_models(device)
    scene_mod, scene_classes = get_scene_model(device)
    print("[Models] Done.")
    return {"obj":obj,"sign":sign,"ocr":ocr,
            "cap_proc":cap_proc,"cap_mod":cap_mod,
            "scene_mod":scene_mod,"scene_classes":scene_classes}

def process_frames(video, M):
    ev,lm,cap,scn,ocr_txt = [],[],[],[],[]
    sign_stats = {"shop":{}, "building":{}, "other":{}}
    prev = None
    for frm in extract_frames(video):
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        ev.append(detect_event_for_frame(prev, gray))
        with torch.no_grad():
            lmk, txt = detect_landmarks_for_frame(frm, M["obj"], M["ocr"])
            lm.append(lmk); ocr_txt.append(txt)
            cap.append(generate_caption_for_frame(frm, M["cap_proc"], M["cap_mod"], lmk))
            scn.append(classify_scene_for_frame(frm, M["scene_mod"], M["scene_classes"]))
            if M["sign"]:
                for b in M["sign"](frm, conf=0.15, verbose=False)[0].boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    s_txt = " ".join(t[1] for t in M["ocr"].readtext(frm[y1:y2,x1:x2]))
                    if not salient(s_txt): continue
                    k = kind_of(s_txt); e = sign_stats[k].setdefault(s_txt,[0,0.0])
                    e[0]+=1; e[1]+=float(b.conf[0])
        prev = gray
    torch.cuda.empty_cache()
    return ev,lm,cap,scn,ocr_txt,sign_stats

def run_pipeline(video):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {video} (device: {device}) ===\n")
    M = load_models(device)
    ev,lm,cap,scn,ocr,signs = process_frames(video, M)
    for row in summarise_journey(ev,lm,cap,scn,ocr):
        print(f"[{row['step']:03}] {row['event']:<11} | Scene: {row['scene']:<20} | {row['description']}")
    print("\n―――――  Long‑form summary  ―――――\n")
    print(generate_long_summary(ev,lm,cap,scn,ocr,signs))
    print("\n―――――――――――――――――――――――――――――\n")

if __name__ == "__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    p = argparse.ArgumentParser(); p.add_argument("--video", required=True)
    run_pipeline(p.parse_args().video)
