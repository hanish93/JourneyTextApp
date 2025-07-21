import cv2, torch, pathlib, logging
from utils import sample_frames, motion, load_detectors, static_landmarks, \
                  load_captioner, summarise

def process(video, models, dev, fps=1):
    caps=[]; whitelist=set(); prev=None
    for f in sample_frames(video,fps):
        gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        verb=motion(prev,gray); prev=gray
        names=static_landmarks(f,models["yolo"],models["ocr"])
        whitelist.update(names)
        cap=models["cap"](f,max_new_tokens=30)[0]["generated_text"]
        # strip dynamic words
        cap=" ".join(w for w in cap.split() if w.lower() not in {"a","an","the"}|DYNAMIC)
        caps.append(f"I {verb} and {cap.lower()}")
    return summarise(caps,whitelist)

def run(target):
    dev="cuda" if torch.cuda.is_available() else"cpu"
    yolo,ocr=load_detectors(dev); captioner=load_captioner(dev)
    models=dict(yolo=yolo,ocr=ocr,cap=captioner)
    p=pathlib.Path(target); files=[p] if p.is_file() else sorted(p.glob("*.mp4"))
    for v in files:
        print("\n———",v.name,"———")
        print(process(str(v),models,dev))

# legacy alias
run_pipeline=run

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore",category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    a=argparse.ArgumentParser(); a.add_argument("--video",required=True)
    run(a.parse_args().video)
