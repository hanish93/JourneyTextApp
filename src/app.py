import torch, cv2, logging, pathlib
from utils import (
    extract_frames, detect_event_for_frame,
    get_landmark_models, detect_landmarks_for_frame,
    get_caption_models, generate_caption_for_frame,
    get_scene_model, classify_scene_for_frame,
    generate_long_summary
)

def load_models(device):
    (obj,_), ocr = get_landmark_models(device)
    cap_proc, cap_mod = get_caption_models(device)
    scene_mod, scene_cls, scene_tf = get_scene_model(device)
    return dict(obj=obj, ocr=ocr,
                cap_proc=cap_proc, cap_mod=cap_mod,
                scene_mod=scene_mod, scene_cls=scene_cls, scene_tf=scene_tf)

def process(video, M):
    ev,lm,cap,scn,ocr = [],[],[],[],[]
    prev=None
    for f in extract_frames(video):
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        ev.append(detect_event_for_frame(prev,g))
        lmk,txt=detect_landmarks_for_frame(f,M["obj"],M["ocr"])
        lm.append(lmk); ocr.append(txt)
        cap.append(generate_caption_for_frame(f,M["cap_proc"],M["cap_mod"],lmk))
        scn.append(classify_scene_for_frame(f,M["scene_mod"],M["scene_cls"],M["scene_tf"]))
        prev=g
    return ev,lm,cap,scn,ocr,{"shop":{},"building":{},"other":{}}

def run(video_or_dir):
    device="cuda" if torch.cuda.is_available() else"cpu"
    M=load_models(device)
    p=pathlib.Path(video_or_dir)
    files=[p] if p.is_file() else sorted(p.glob("*.mp4"))
    for vid in files:
        ev,lm,cap,scn,ocr,stats=process(str(vid),M)
        print(f"\n——— {vid} ——")
        print(generate_long_summary(ev,lm,cap,scn,ocr,stats),"\n")

run_pipeline = run

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore",category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    arg=argparse.ArgumentParser(); arg.add_argument("--video",required=True)
    run(arg.parse_args().video)
