import torch, cv2, logging, time
from utils import (
    extract_frames, detect_event,
    get_models, detect_static,
    get_scene_model, classify_scene,
    get_caption_models, caption,
    summarise, journey_table,
    salient, kind_of
)

def load_all(device):
    (obj,sign), ocr = get_models(device)
    scene_mod, scene_cls, scene_tf = get_scene_model(device)
    cap_proc, cap_mod = get_caption_models(device)
    return dict(obj=obj,sign=sign,ocr=ocr,
                scene_mod=scene_mod,scene_cls=scene_cls,scene_tf=scene_tf,
                cap_proc=cap_proc,cap_mod=cap_mod)

def process(video, M):
    ev,lm,cap,scn,ocr_txt=[],[],[],[],[]
    sign_stats= {"shop":{}, "building":{}, "other":{}}
    conf_scores=[]; prev=None; start=time.time()
    frames=list(extract_frames(video))
    # first pass to get adaptive conf
    for f in frames[:min(len(frames),50)]:
        r=M["obj"](f,verbose=False,conf=0.1)[0]
        conf_scores.extend([float(c) for c in r.boxes.conf.cpu()])
    adaptive_conf = (np.median(conf_scores)*0.6) if conf_scores else 0.25

    for f in frames:
        gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        ev.append(detect_event(prev,gray))
        landmark,txt=detect_static(f,M["obj"],M["ocr"],adaptive_conf)
        lm.append(landmark); ocr_txt.append(txt)
        cap.append(caption(f,M["cap_proc"],M["cap_mod"],landmark))
        scn.append(classify_scene(f,M["scene_mod"],M["scene_cls"],M["scene_tf"]))

        if M["sign"]:
            for b in M["sign"](f,conf=0.15,verbose=False)[0].boxes:
                x1,y1,x2,y2=map(int,b.xyxy[0])
                s_txt=" ".join(M["ocr"].readtext(utils.preprocess_crop(f[y1:y2,x1:x2]),detail=0))
                if not salient(s_txt): continue
                k=kind_of(s_txt); e=sign_stats[k].setdefault(s_txt,[0,0.0])
                e[0]+=1; e[1]+=float(b.conf[0])
        prev=gray
    secs=time.time()-start
    return ev,lm,cap,scn,ocr_txt,sign_stats,secs

def run(video):
    device="cuda" if torch.cuda.is_available() else"cpu"
    M=load_all(device)
    ev,lm,cap,scn,ocr_txt,stats,secs=process(video,M)
    for row in journey_table(ev,lm,cap,scn,ocr_txt):
        print(f"[{row['step']:03}] {row['event']:<11} | Scene: {row['scene']:<20} | {row['description']}")
    print("\n―――――  Long‑form summary  ―――――\n")
    print(summarise(ev,lm,cap,scn,ocr_txt,stats,secs))
    print("\n―――――――――――――――――――――――――――――\n")

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore",category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    p=argparse.ArgumentParser(); p.add_argument("--video",required=True)
    run(p.parse_args().video)
