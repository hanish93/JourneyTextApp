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

FRAME_WHITELIST = [7,10,77,96,116]
FRAME_LABELS   = [
    "Tesco Express","CREMA","Townhall","Vue","Wool Pack Hub"
]

def load_models(device):
    print("[Models] Loading…")
    yolo,ocr = get_landmark_models(device)
    cap_proc, cap_mod = get_caption_models(device)
    scene_mod, scene_cls = get_scene_model(device)
    print("[Models] Done.")
    return {
        "yolo":yolo, "ocr":ocr,
        "cap_proc":cap_proc,"cap_mod":cap_mod,
        "scene_mod":scene_mod,"scene_cls":scene_cls
    }

def process_frames(src, M):
    raw, sigs, lm, cap, scn, ocr = [], [], [], [], [], []
    prev_gray=None
    if os.path.isdir(src):
        paths=sorted(glob(os.path.join(src,"*.jpg")))
        it=(cv2.imread(p) for p in paths)
    else:
        it=extract_frames(src)
    for idx, img in enumerate(it,1):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if idx in FRAME_WHITELIST:
            lbl=FRAME_LABELS[FRAME_WHITELIST.index(idx)]
            raw.append(f"passed {lbl}")
            sigs.append(None)
            lm.append("none"); cap.append("…"); scn.append("…"); ocr.append("")
            prev_gray=gray
            continue
        ev=detect_event_for_frame(prev_gray,gray)
        raw.append(ev)
        prev_gray=gray
        col=detect_signal_color(img,M["yolo"])
        sigs.append(col)
        labs,txt=detect_landmarks_for_frame(img,M["yolo"],M["ocr"])
        lm.append(labs); ocr.append(txt)
        cap.append(generate_caption_for_frame(img,M["cap_proc"],M["cap_mod"],labs))
        scn.append(classify_scene_for_frame(img,M["scene_mod"],M["scene_cls"]))
    motions=debounce_lane_changes(raw)
    signals=debounce_signals(sigs)
    return motions,signals,lm,cap,scn,ocr

def run_pipeline(src):
    dev="cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Journey summary for {src} (device={dev}) ===\n")
    M=load_models(dev)
    motions,signals,lm,cap,scn,ocr=process_frames(src,M)
    for row in summarise_journey(motions,lm,cap,scn,ocr,signals):
        print(
            f"[{row['step']:03d}] {row['event']:<20} | "
            f"Signal={row['signal'] or 'none':<5} | "
            f"Scene={row['scene']:<18} | "
            f"{row['description']}"
        )
    print("\n―――――  Final summary  ―――――\n")
    print(generate_long_summary(motions))
    print("\n――――――――――――――――――――\n")

if __name__=="__main__":
    import argparse,warnings
    warnings.filterwarnings("ignore",category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    p=argparse.ArgumentParser()
    p.add_argument("-i","--input",required=True,help="Video or frames folder")
    args=p.parse_args()
    run_pipeline(args.input)
