# utils.py  ─── only persistent‑landmark narrative ───────────────────────
import os, cv2, urllib.request, json, torch, easyocr, numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
)

# ───────────── static config / helpers ───────────────────────────────────
STATIC_YOLO_CLASSES = {
    "traffic light", "stop sign", "street sign", "traffic sign", "bus stop",
    "bench", "fire hydrant", "parking meter", "clock", "potted plant"
}
DYNAMIC_WORDS = {"car","person","truck","bus","motorcycle","bicycle","dog"}

def salient(t): 
    w=t.split(); return len(w)>=2 or (w and w[0][0].isupper())

def kind_of(t):
    l=t.lower()
    if any(x in l for x in ["shop","store","mart","market","express"]): return"shop"
    if any(x in l for x in ["registration","office","tower","hotel","plaza","building","center","carter"]): return"building"
    return"other"

# ───────────── frame extraction & motion ────────────────────────────────
def extract_frames(v,fps=1):
    cap=cv2.VideoCapture(v); nat=cap.get(cv2.CAP_PROP_FPS)or 30; step=max(1,round(nat/fps))
    idx,ok,img=0,*cap.read(); print("[Frames] Starting extraction …")
    while ok:
        if idx%step==0: yield img
        ok,img=cap.read(); idx+=1
    cap.release(); print("[Frames] Done.")

def detect_event_for_frame(prev,cur,dx=1.5,stop=0.2):
    if prev is None: return"drive"
    flow=cv2.calcOpticalFlowFarneback(prev,cur,None,.5,3,15,3,5,1.2,0)
    dxm=flow[...,0].mean(); mag=np.linalg.norm(flow,axis=2).mean()
    if mag<stop: return"stop"
    if dxm>dx:   return"turn_right"
    if dxm<-dx:  return"turn_left"
    return"drive"

# ───────────── download helper ──────────────────────────────────────────
def fetch(name,dir_,url,cfg):
    os.makedirs(dir_,exist_ok=True)
    path=os.path.join(dir_,cfg)
    if url and not os.path.exists(path):
        print(f"[Model] Downloading {name} …"); urllib.request.urlretrieve(url,path)
    return path

# ───────────── detectors & OCR ──────────────────────────────────────────
def get_landmark_models(device):
    obj=YOLO(fetch("yolov8n","models",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt","yolov8n.pt")).to(device).half()
    sign_path="models/yolov8_signs.pt"; sign=YOLO(sign_path).to(device).half() if os.path.exists(sign_path) else None
    return (obj,sign), easyocr.Reader(["en","it"],gpu=device.startswith("cuda"))

def detect_landmarks_for_frame(f,model,ocr,conf=0.25):
    r=model(f,verbose=False,conf=conf)[0]
    if not r.boxes: return"none",""
    lbls,txts=[],[]
    for b in r.boxes:
        cls=model.model.names[int(b.cls[0])]
        if cls not in STATIC_YOLO_CLASSES: continue
        x1,y1,x2,y2=map(int,b.xyxy[0])
        t=" ".join(s[1] for s in ocr.readtext(f[y1:y2,x1:x2]))
        if t: txts.append(t); lbls.append(f"{cls} [{t}]")
        else: lbls.append(cls)
    return(", ".join(lbls)if lbls else"none")," ".join(txts)

# ───────────── scene classifier ─────────────────────────────────────────
def get_scene_model(device):
    from torchvision import models
    ck=fetch("places365","models",
        "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
        "resnet18_places365.pth.tar")
    m=models.resnet18(num_classes=365); sd=torch.load(ck,map_location="cpu")["state_dict"]
    m.load_state_dict({k.replace("module.",""):v for k,v in sd.items()}); m.to(device).half().eval()
    cat="categories_places365.txt"
    if not os.access(cat,os.W_OK):
        os.system("wget -q https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt")
    cls=[l.strip().split(" ")[0][3:] for l in open(cat)]
    return m,cls

def classify_scene_for_frame(f,m,cls):
    from torchvision import transforms
    tf=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    inp=tf(img).unsqueeze(0).to(next(m.parameters()).device,dtype=next(m.parameters()).dtype)
    with torch.no_grad(): p=torch.nn.functional.softmax(m(inp),1)
    return cls[int(p.argmax())]

# ───────────── caption model ────────────────────────────────────────────
def get_caption_models(d):
    repo="Salesforce/blip-image-captioning-base"
    return BlipProcessor.from_pretrained(repo), BlipForConditionalGeneration.from_pretrained(repo).to(d)

def generate_caption_for_frame(f,proc,mod,lm=None):
    img=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    if max(img.size)>512:
        img.thumbnail((512,512),Image.Resampling.LANCZOS if hasattr(Image,"Resampling") else Image.LANCZOS)
    ins=proc(images=img,text=(f"Scene contains: {lm}." if lm else ""),return_tensors="pt").to(mod.device)
    with torch.no_grad(): ids=mod.generate(**ins,max_new_tokens=30)
    return proc.batch_decode(ids,skip_special_tokens=True)[0].strip()

# ───────────── summary generator ─────────────────────────────────────────
def generate_long_summary(events,landmarks,captions,scenes,ocr,stats):
    repo="mistralai/Mistral-7B-Instruct-v0.2"
    model=AutoModelForCausalLM.from_pretrained(repo,device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),torch_dtype=torch.float16)
    tok=AutoTokenizer.from_pretrained(repo)

    lines=[]
    for i in range(len(events)):
        cap_clean=" ".join(w for w in captions[i].split() if w.lower() not in DYNAMIC_WORDS)
        lines.append(f"Frame {i+1}: Scene={scenes[i]}, Event={events[i]}, "
                     f"Caption={cap_clean}, Landmark={landmarks[i]}, OCR='{ocr[i]}'")

    whitelist=[]
    for bag in stats.values():
        whitelist.extend(t for t,_ in sorted(bag.items(),key=lambda kv:(-kv[1][0],-kv[1][1]))[:2])

    if whitelist:
        guard=f"Use **only** these place names (and no others): {', '.join(whitelist)}.\n"
    else:
        guard="Do **not** mention any place names, cities, countries or regions.\n"

    prompt=("Summarize the journey factually in a diary tone. "
            "Ignore transient objects like people and vehicles.\n"+
            guard+"---\n"+ "\n".join(lines)+"\n---\nJourney Summary:\n")

    inp=tok(prompt,return_tensors="pt").to(model.device)
    with torch.no_grad():
        out=model.generate(**inp,max_new_tokens=160,do_sample=False,top_p=1.0)
    return tok.decode(out[0],skip_special_tokens=True).split("Journey Summary:")[-1].strip()

# ───────────── step table helper ─────────────────────────────────────────
def summarise_journey(ev,lm,cap,scn,ocr):
    return[{"step":i+1,"event":ev[i],"scene":scn[i],
            "description":f"{cap[i]}. Landmark: {lm[i]}. OCR: '{ocr[i]}'"} for i in range(len(ev))]
