import os, cv2, urllib.request, json, torch, easyocr, numpy as np, math
from PIL import Image
from collections import defaultdict
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
)

STATIC_YOLO_CLASSES = {
    "traffic light","stop sign","street sign","traffic sign","bus stop",
    "bench","fire hydrant","parking meter","clock","potted plant"
}
DYNAMIC_WORDS = {"car","person","truck","bus","motorcycle","bicycle","dog"}

# ───────────── helper rules ──────────────────────────────────────────────
def salient(txt:str)->bool:
    w=txt.split(); return len(w)>=2 or (w and w[0][0].isupper())

def kind_of(txt:str)->str:
    low=txt.lower()
    if any(x in low for x in ["shop","store","mart","market","express"]): return"shop"
    if any(x in low for x in ["registration","office","tower","hotel","plaza",
                              "building","center","carter"]): return"building"
    return"other"

def preprocess_crop(crop):
    h,w=crop.shape[:2]
    if h<32:
        scale=32/h; crop=cv2.resize(crop,(int(w*scale),32),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

# ───────────── frame extraction & motion ─────────────────────────────────
def extract_frames(video_path,fps=1):
    cap=cv2.VideoCapture(video_path)
    nat=cap.get(cv2.CAP_PROP_FPS) or 30
    step=max(1,round(nat/fps))
    i,ok,img=0,*cap.read()
    while ok:
        if i%step==0: yield img
        ok,img=cap.read(); i+=1
    cap.release()

def detect_event(prev,cur,dx_turn=1.5,flow_stop=0.20):
    if prev is None: return"drive"
    flow=cv2.calcOpticalFlowFarneback(prev,cur,None,.5,3,15,3,5,1.2,0)
    dx=flow[...,0].mean(); mag=np.linalg.norm(flow,axis=2).mean()
    if mag<flow_stop: return"stop"
    if dx>dx_turn:     return"turn_right"
    if dx<-dx_turn:    return"turn_left"
    return"drive"

def fetch(dir_,url,fname):
    os.makedirs(dir_,exist_ok=True); path=os.path.join(dir_,fname)
    if url and not os.path.exists(path):
        urllib.request.urlretrieve(url,path)
    return path

# ───────────── detectors / OCR ───────────────────────────────────────────
def get_models(device):
    obj=YOLO(fetch("models",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8n.pt")).to(device).half()
    sign_path="models/yolov8_signs.pt"
    sign=YOLO(sign_path).to(device).half() if os.path.exists(sign_path) else None
    ocr=easyocr.Reader(["en","latin"],gpu=device.startswith("cuda"))
    return(obj,sign),ocr

def detect_static(frame,det,ocr,conf):
    res=det(frame,verbose=False,conf=conf)[0]
    if not res.boxes: return"none",""
    labels,txts=[],[]
    for b in res.boxes:
        cls=det.model.names[int(b.cls[0])]
        if cls not in STATIC_YOLO_CLASSES: continue
        x1,y1,x2,y2=map(int,b.xyxy[0])
        crop=preprocess_crop(frame[y1:y2,x1:x2])
        t=" ".join(s[1] for s in ocr.readtext(crop,detail=0))
        if t: txts.append(t); labels.append(f"{cls} [{t}]")
        else: labels.append(cls)
    return(", ".join(labels)if labels else"none")," ".join(txts)

# ───────────── scene classifier & caption (unchanged API) ────────────────
def get_scene_model(device):
    from torchvision import models,transforms
    ck=fetch("models",
        "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
        "resnet18_places365.pth.tar")
    m=models.resnet18(num_classes=365); sd=torch.load(ck,map_location="cpu")["state_dict"]
    m.load_state_dict({k.replace("module.",""):v for k,v in sd.items()})
    m.to(device).half().eval()
    cat="categories_places365.txt"
    if not os.access(cat,os.W_OK):
        os.system("wget -q https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt")
    cls=[l.strip().split(" ")[0][3:] for l in open(cat)]
    tf=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return m,cls,tf

def classify_scene(frame,model,cls,tf):
    inp=tf(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))).unsqueeze(0).to(
        next(model.parameters()).device,dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        p=torch.nn.functional.softmax(model(inp),1)
    return cls[int(p.argmax())]

def get_caption_models(device):
    repo="Salesforce/blip-image-captioning-base"
    return BlipProcessor.from_pretrained(repo), BlipForConditionalGeneration.from_pretrained(repo).to(device)

def caption(frame,proc,mod,hint=""):
    img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    if max(img.size)>512: img.thumbnail((512,512),Image.Resampling.LANCZOS)
    ins=proc(images=img,text=hint,return_tensors="pt").to(mod.device)
    with torch.no_grad(): out=mod.generate(**ins,max_new_tokens=30)
    return proc.batch_decode(out,skip_special_tokens=True)[0].strip()

# ───────────── summary ───────────────────────────────────────────────────
USE_SAMPLING=False
def summarise(events,lm,cap,scn,ocr,stats,seconds):
    repo="mistralai/Mistral-7B-Instruct-v0.2"
    model=AutoModelForCausalLM.from_pretrained(
        repo,device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16)
    tok=AutoTokenizer.from_pretrained(repo)

    lines=[]
    for i in range(len(events)):
        clean=" ".join(w for w in cap[i].split() if w.lower() not in DYNAMIC_WORDS)
        lines.append(f"Frame {i+1}: Scene={scn[i]}, Event={events[i]}, "
                     f"Caption={clean}, Landmark={lm[i]}, OCR='{ocr[i]}'")

    minutes=seconds/60
    K=min(5,max(1,math.ceil(minutes/2)))
    whitelist=[]
    for bag in stats.values():
        whitelist.extend(t for t,_ in sorted(bag.items(),key=lambda kv:(-kv[1][0],-kv[1][1]))[:K])

    guard=("Use **only** these place names: "+", ".join(whitelist)+".\n") if whitelist else \
          "Do **not** mention any place names.\n"

    prompt=("Summarize the journey below in a factual diary tone. "
            "Ignore people and vehicles.\n"+guard+"---\n"+
            "\n".join(lines)+"\n---\nJourney Summary:\n")

    inp=tok(prompt,return_tensors="pt").to(model.device)
    gen_kwargs=dict(max_new_tokens=160,do_sample=USE_SAMPLING)
    if USE_SAMPLING: gen_kwargs.update(temperature=0.2,top_p=0.8)
    with torch.no_grad(): out=model.generate(**inp,**gen_kwargs)
    return tok.decode(out[0],skip_special_tokens=True).split("Journey Summary:")[-1].strip()

# ───────────── public helper for tables ──────────────────────────────────
def journey_table(ev,lm,cap,scn,ocr):
    return[{"step":i+1,"event":ev[i],"scene":scn[i],
            "description":f"{cap[i]}. Landmark: {lm[i]}. OCR: '{ocr[i]}'"}
            for i in range(len(ev))]
