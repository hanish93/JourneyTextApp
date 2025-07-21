import cv2, torch, pathlib, logging
from utils import frames, move, load_det, landmarks, load_cap, cap_img, diary, DYNAMIC

def run_clip(path, models, dev):
    cap_pipe=models["cap"]; yolo,ocr=models["det"]
    prev=None; sentences=[]; wl=set()

    for f in frames(path,fps=1):
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        verb=move(prev,g); prev=g
        names=landmarks(f,yolo,ocr); wl.update(names)
        cap=cap_img(f,cap_pipe," ".join(names) if names else "")
        cap=" ".join(w for w in cap.split() if w.lower() not in DYNAMIC)
        sentences.append(f"I {verb} and {cap.lower()}")

    return diary(sentences,wl)

def run(target):
    dev="cuda" if torch.cuda.is_available() else"cpu"
    yolo,ocr=load_det(dev); cap=load_cap(dev)
    models=dict(det=(yolo,ocr),cap=cap)
    p=pathlib.Path(target); vids=[p] if p.is_file() else sorted(p.glob("*.mp4"))
    for v in vids:
        print("\n———",v.name,"———")
        print(run_clip(str(v),models,dev))

run_pipeline=run   # cli entry

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore",category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    a=argparse.ArgumentParser(); a.add_argument("--video",required=True)
    run(a.parse_args().video)
