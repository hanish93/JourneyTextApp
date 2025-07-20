import os, cv2, urllib.request, torch, easyocr, numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer,
)

STATIC_YOLO_CLASSES = {
    "traffic light", "stop sign", "street sign", "traffic sign", "bus stop",
    "bench", "fire hydrant", "parking meter", "clock", "potted plant",
}
DYNAMIC_WORDS = {"car","person","truck","bus","motorcycle","bicycle","dog"}

# ───── helper filters ────────────────────────────────────────────────
salient  = lambda t: (t.split() and (len(t.split()) >= 2 or t[0].isupper()))
kind_of  = lambda t: ("shop" if any(x in t.lower() for x in
                ["shop","store","mart","market","express"]) else
                "building" if any(x in t.lower() for x in
                ["registration","office","tower","hotel","plaza",
                 "building","center","carter"]) else "other")

# ───── frame utils ───────────────────────────────────────────────────
def extract_frames(path,fps=1):
    cap=cv2.VideoCapture(path); native=cap.get(cv2.CAP_PROP_FPS) or 30
    step=max(1,round(native/fps)); i,ok,img=0,*cap.read()
    while ok:
        if i%step==0: yield img
        ok,img=cap.read(); i+=1
    cap.release()

def detect_event(prev,cur,dx=1.5,stop=0.2):
    if prev is None: return"drive"
    flow=cv2.calcOpticalFlowFarneback(prev,cur,None,.5,3,15,3,5,1.2,0)
    dxm=flow[...,0].mean(); mag=np.linalg.norm(flow,axis=2).mean()
    if mag<stop: return"stop"
    if dxm>dx:   return"turn_right"
    if dxm<-dx:  return"turn_left"
    return"drive"

# ───── download helper ───────────────────────────────────────────────
def fetch(dir_,url,fname):
    os.makedirs(dir_,exist_ok=True); p=os.path.join(dir_,fname)
    if url and not os.path.exists(p): urllib.request.urlretrieve(url,p)
    return p

# ───── detectors & OCR ───────────────────────────────────────────────
def get_landmark_models(device):
    obj=YOLO(fetch("models",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8n.pt")).to(device).half()
    ocr=easyocr.Reader(["en"],gpu=device.startswith("cuda"))
    return obj, ocr

def detect_static(frame,model,ocr,conf=0.25):
    res=model(frame,verbose=False,conf=conf)[0]
    labels,txts=[],[]
    for b in res.boxes:
        cls=model.model.names[int(b.cls[0])]
        if cls not in STATIC_YOLO_CLASSES: continue
        x1,y1,x2,y2=map(int,b.xyxy[0])
        t=" ".join(ocr.readtext(frame[y1:y2,x1:x2],detail=0))
        if t: txts.append(t); labels.append(f"{cls} [{t}]")
        else: labels.append(cls)
    return (", ".join(labels) or "none"), " ".join(txts)

# ───── scene & caption models (unchanged API) ─────────────────────────
def get_scene_model(device):
    from torchvision import models,transforms
    ck=fetch("models",
        "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
        "resnet18_places365.pth.tar")
    m=models.resnet18(num_classes=365)
    m.load_state_dict({k.replace("module.",""):v for k,v in
        torch.load(ck,map_location="cpu")["state_dict"].items()})
    m.to(device).half().eval()
    tf=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    classes=[l.strip().split(" ")[0][3:] for l in open("categories_places365.txt")]
    return m,classes,tf

def classify_scene(frame,model,cls,tf):
    img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    inp=tf(img).unsqueeze(0).to(next(model.parameters()).device,
                                dtype=next(model.parameters()).dtype)
    with torch.no_grad(): p=torch.nn.functional.softmax(model(inp),1)
    return cls[int(p.argmax())]

def get_caption_models(device):
    repo="Salesforce/blip-image-captioning-base"
    return (BlipProcessor.from_pretrained(repo),
            BlipForConditionalGeneration.from_pretrained(repo).to(device))

def caption(frame,proc,mod,hint=""):
    img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    if max(img.size)>512: img.thumbnail((512,512),Image.Resampling.LANCZOS)
    ins=proc(images=img,text=hint,return_tensors="pt").to(mod.device)
    with torch.no_grad(): ids=mod.generate(**ins,max_new_tokens=30)
    return proc.batch_decode(ids,skip_special_tokens=True)[0].strip()

# ───── summary (deterministic, whitelist) ──────────────────────────────
def generate_long_summary(ev,lm,cap,scn,ocr_txt,stats):
    repo="mistralai/Mistral-7B-Instruct-v0.2"
    model=AutoModelForCausalLM.from_pretrained(
        repo,device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16)
    tok=AutoTokenizer.from_pretrained(repo)

    lines=[f"Frame {i+1}: Scene={scn[i]}, Event={ev[i]}, "
           f"Caption={' '.join(w for w in cap[i].split() if w.lower() not in DYNAMIC_WORDS)}, "
           f"Landmark={lm[i]}, OCR='{ocr_txt[i]}'" for i in range(len(ev))]

    wl=[t for bag in stats.values()
          for t,_ in sorted(bag.items(),key=lambda kv:(-kv[1][0],-kv[1][1]))[:2]]

    guard=("Use ONLY these place names: "+", ".join(wl)+".\n") if wl else \
          "Do NOT mention any place names.\nNo notable landmarks were detected.\n"

    prompt=("Summarize in 3‑4 simple sentences, first‑person. "
            "No headings like 'Day 1'. Ignore people/vehicles.\n"+
            guard+"---\n"+ "\n".join(lines)+"\n---\nSummary:\n")

    out=model.generate(**tok(prompt,return_tensors="pt").to(model.device),
                       max_new_tokens=120,do_sample=False)
    return tok.decode(out[0],skip_special_tokens=True).split("Summary:")[-1].strip()

def table(ev,lm,cap,scn,ocr):
    return[{"step":i+1,"event":ev[i],"scene":scn[i],
            "description":f"{cap[i]} | {lm[i]} | OCR '{ocr[i]}'"}
           for i in range(len(ev))]
