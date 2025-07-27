# src/app.py
import os, cv2, torch, numpy as np, logging
from glob import glob
from ultralytics import YOLO

def fetch(dst, url, fname):  # same helper as before
    import urllib.request, os
    os.makedirs(dst, exist_ok=True)
    path = os.path.join(dst, fname)
    if url and not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path

def load_detector(dev, path=None):
    """
    If `path` is your custom yolov8n.pt trained on
    your 6–7 shops + traffic_light, it will only
    ever detect those.
    """
    if path:
        model = YOLO(path).to(dev).half()
    else:
        pt = fetch("models",
                   "https://github.com/ultralytics/assets/releases/"
                   "download/v0.0.0/yolov8n.pt","yolov8n.pt")
        model = YOLO(pt).to(dev).half()
    return model

def detect_light_color(roi):
    """HSV‐mask quick red vs green on a cropped traffic‐light ROI."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # red
    r1,r2=np.array([0,70,50]),np.array([10,255,255])
    r3,r4=np.array([170,70,50]),np.array([180,255,255])
    red = int(cv2.countNonZero(cv2.inRange(hsv, r1, r2)) + \
              cv2.countNonZero(cv2.inRange(hsv, r3, r4)))
    # green
    g1,g2=np.array([40,40,40]),np.array([90,255,255])
    green = int(cv2.countNonZero(cv2.inRange(hsv, g1, g2)))
    if green>red and green>50: return "green"
    if red>green and red>50:   return "red"
    return None

def move(prev, cur):
    """Exactly your optical‐flow verb."""
    if prev is None: return "drive"
    f = cv2.calcOpticalFlowFarneback(prev, cur, None,
                                     0.5,3,15,3,5,1.2,0)
    dxm = f[...,0].mean(); mag = np.linalg.norm(f,axis=2).mean()
    if mag<0.2:       return "stop"
    if dxm>1.5:       return "turn_right"
    if dxm< -1.5:     return "turn_left"
    return "drive"

def run_clip(path, yolo_model):
    # build frame iterator
    if os.path.isdir(path):
        imgs = sorted(glob(os.path.join(path,"*.jpg")))
        if imgs:
            frames = (cv2.imread(f) for f in imgs)
        else:
            vids = sorted(glob(os.path.join(path,"*.mp4")))
            return "\n\n".join(run_clip(v,yolo_model) for v in vids)
    else:
        # 1fps sampler from cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, round(fps/1))
        idx,ok,frames=[],*cap.read()
        i=0
        while ok:
            if i%step==0: frames.append(frames)
            ok,frm=cap.read(); i+=1
        cap.release()
        frames = iter(frames)

    prev_gray=None
    seen=set()
    prev_light=None
    events=[]
    for i, img in enumerate(frames,1):
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        verb = move(prev_gray, gray); prev_gray=gray

        # run YOLO once
        r = yolo_model(img, conf=0.25, verbose=False)[0]
        for box in r.boxes:
            cls = yolo_model.model.names[int(box.cls[0])]
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            if cls=="traffic light":
                color = detect_light_color(crop)
                if color and color!=prev_light:
                    events.append(f"signal_{color}")
                prev_light=color

            else:
                # must be one of your 6–7 pre‐trained signs
                if cls not in seen:
                    seen.add(cls)
                    events.append(f"sign_{cls}")

        events.append(verb)
        print(f"[frame {i:03d}]  verb={verb:10s}  light={prev_light or '-':5s}"
              f"  new_signs={set(r.names[int(b.cls[0])] for b in r.boxes)-seen}")

    # map to English
    mapping = {
      "signal_green":"the signal turned green",
      "signal_red":"stopped at the red light",
      "drive":"drove straight",
      "stop":"came to a stop",
      "turn_right":"took a slight right",
      "turn_left":"took a slight left",
    }
    parts=[]
    for e in events:
        if e in mapping:
            parts.append(mapping[e])
        elif e.startswith("sign_"):
            parts.append(f"passed {e.split('_',1)[1]}")
    summary = "I " + " and ".join(parts) + "."
    print("\n=== JOURNEY SUMMARY ===\n" + summary)
    return summary

if __name__=="__main__":
    import argparse, warnings
    warnings.filterwarnings("ignore",category=UserWarning)
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    p=argparse.ArgumentParser()
    p.add_argument("--input","-i",required=True,
                   help="folder of frames or single mp4")
    p.add_argument("--yolo-model","-m",required=True,
                   help="your custom-trained yolov8 .pt")
    args=p.parse_args()

    dev="cuda" if torch.cuda.is_available() else "cpu"
    yolo,_ = load_detector(dev,args.yolo_model)
    run_clip(args.input,yolo)
