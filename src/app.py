import torch, cv2, logging, pathlib
from utils import (
    extract_frames, detect_event,              # renamed in utils
    get_landmark_models, detect_static,
    get_caption_models, caption,
    get_scene_model, classify_scene,
    generate_long_summary, table,
    salient, kind_of
)

def load_models(device):
    obj, ocr = get_landmark_models(device)
    cap_proc, cap_mod = get_caption_models(device)
    scene_mod, scene_cls, scene_tf = get_scene_model(device)
    return dict(obj=obj, ocr=ocr,
                cap_proc=cap_proc, cap_mod=cap_mod,
                scene_mod=scene_mod, scene_cls=scene_cls, scene_tf=scene_tf)

def run_single(path, M, device):
    ev,lm,cap,scn,ocr_txt=[],[],[],[],[]
    stats={"shop":{},"building":{},"other":{}}
    prev=None
    for f in extract_frames(path):
        gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        ev.append(detect_event(prev,gray))
        lmk,txt=detect_static(f,M["obj"],M["ocr"])
        lm.append(lmk); ocr_txt.append(txt)
        cap.append(caption(f,M["cap_proc"],M["cap_mod"],lmk))
        scn.append(classify_scene(f,M["scene_mod"],M["scene_cls"],M["scene_tf"]))
        prev=gray
    summary=generate_long_summary(ev,lm,cap,scn,ocr_txt,stats)
    print(f"\n——— {path} ——\n{summary}\n")

def run(target):
    device="cuda" if torch.cuda.is_available() else"cpu"
    M=load_models(device)
    p=pathlib.Path(target)
    files=[p] if p.is_file() else sorted(p.glob("*.mp4"))
    for vid in files:
        run_single(str(vid),M,device)

run_pipeline = run  # legacy

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore",category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    parser=argparse.ArgumentParser(); parser.add_argument("--video",required=True)
    run(parser.parse_args().video)
