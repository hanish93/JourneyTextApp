# utils.py  ────────────────────────────────────────────────────────────────
import os, cv2, urllib.request, json, torch, easyocr, numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
)

# ───────────── static config / helpers ────────────────────────────────────
STATIC_YOLO_CLASSES = {
    "traffic light", "stop sign", "street sign", "traffic sign", "bus stop",
    "bench", "fire hydrant", "parking meter", "clock", "potted plant"
}
DYNAMIC_WORDS = {"car","person","truck","bus","motorcycle","bicycle","dog"}

def salient(txt:str)->bool:
    words=txt.split()
    return len(words)>=2 or (words and words[0][0].isupper())

def kind_of(txt:str)->str:
    low=txt.lower()
    if any(w in low for w in ["shop","store","mart","market","express"]):
        return "shop"
    if any(w in low for w in ["registration","office","tower","hotel","plaza",
                              "building","center","carter"]):
        return "building"
    return "other"

# ───────────── frame extraction & motion ──────────────────────────────────
def extract_frames(video_path,fps=1):
    cap=cv2.VideoCapture(video_path)
    native=cap.get(cv2.CAP_PROP_FPS)or 30
    step=max(1,round(native/fps))
    idx,ok,img=0,*cap.read()
    print("[Frames] Starting extraction …")
    while ok:
        if idx%step==0: yield img
        ok,img=cap.read(); idx+=1
    cap.release(); print("[Frames] Done.")

def detect_event_for_frame(prev,current,dx_turn=1.5,flow_stop=0.20):
    if prev is None: return"drive"
    flow=cv2.calcOpticalFlowFarneback(prev,current,None,.5,3,15,3,5,1.2,0)
    dx=flow[...,0].mean(); mag=np.linalg.norm(flow,axis=2).mean()
    if mag<flow_stop: return"stop"
    if dx>dx_turn: return"turn_right"
    if dx<-dx_turn:return"turn_left"
    return"drive"

# ───────────── model download helper ──────────────────────────────────────
def get_or_download_model(name,dir_,url,cfg):
    os.makedirs(dir_,exist_ok=True)
    if url and not os.path.exists(os.path.join(dir_,cfg)):
        print(f"[Model] Downloading {name} …")
        urllib.request.urlretrieve(url,os.path.join(dir_,os.path.basename(url)))
    return os.path.join(dir_,cfg)

# ───────────── detectors & OCR ────────────────────────────────────────────
def get_landmark_models(device):
    coco=get_or_download_model("yolov8n","models",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8n.pt")
    obj=YOLO(coco).to(device).half()
    sign_path="models/yolov8_signs.pt"
    sign=YOLO(sign_path).to(device).half() if os.path.exists(sign_path) else None
    ocr=easyocr.Reader(["en","it"],gpu=device.startswith("cuda"))
    return(obj,sign),ocr

def detect_landmarks_for_frame(frame,model,ocr,conf=0.25):
    res=model(frame,verbose=False,conf=conf)[0]
    if not res.boxes: return"none",""
    labels,ocr_txts=[],[]
    for b in res.boxes:
        cls=model.model.names[int(b.cls[0])]
        if cls not in STATIC_YOLO_CLASSES: continue
        x1,y1,x2,y2=map(int,b.xyxy[0])
        txt=" ".join(t[1] for t in ocr.readtext(frame[y1:y2,x1:x2]))
        if txt: ocr_txts.append(txt); labels.append(f"{cls} [{txt}]")
        else:   labels.append(cls)
    return(", ".join(labels)if labels else"none")," ".join(ocr_txts)

# ───────────── scene classifier ──────────────────────────────────────────
def get_scene_model(device):
    from torchvision import models
    ckpt=get_or_download_model("places365","models",
        "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
        "resnet18_places365.pth.tar")
    model=models.resnet18(num_classes=365)
    sd=torch.load(ckpt,map_location="cpu")["state_dict"]
    model.load_state_dict({k.replace("module.",""):v for k,v in sd.items()})
    model.to(device).half().eval()
    cat_file="categories_places365.txt"
    if not os.access(cat_file,os.W_OK):
        os.system("wget -q https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt")
    classes=[l.strip().split(" ")[0][3:] for l in open(cat_file)]
    return model,classes

def classify_scene_for_frame(frame,model,classes):
    from torchvision import transforms
    tf=transforms.Compose([
        transforms.Resize((256,256)),transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    inp=tf(img).unsqueeze(0).to(next(model.parameters()).device,
                                dtype=next(model.parameters()).dtype)
    with torch.no_grad(): prob=torch.nn.functional.softmax(model(inp),1)
    return classes[int(prob.argmax())]

# ───────────── caption model ──────────────────────────────────────────────
def get_caption_models(device):
    repo="Salesforce/blip-image-captioning-base"
    return(BlipProcessor.from_pretrained(repo),
           BlipForConditionalGeneration.from_pretrained(repo).to(device))

def generate_caption_for_frame(frame,proc,mod,landmark=None):
    img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    if max(img.size)>512:
        img.thumbnail((512,512),Image.Resampling.LANCZOS if hasattr(Image,"Resampling") else Image.LANCZOS)
    ins=proc(images=img,text=f"Scene contains: {landmark}." if landmark else"",return_tensors="pt").to(mod.device)
    with torch.no_grad(): ids=mod.generate(**ins,max_new_tokens=30)
    return proc.batch_decode(ids,skip_special_tokens=True)[0].strip()

# ───────────── long‑form summary ──────────────────────────────────────────
def generate_long_summary(events,landmarks,captions,scenes,ocr_txts,sign_stats):
    repo="mistralai/Mistral-7B-Instruct-v0.2"
    model=AutoModelForCausalLM.from_pretrained(
        repo,device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16)
    tok=AutoTokenizer.from_pretrained(repo)

    long_lines=[]
    for i in range(len(events)):
        clean=" ".join(w for w in captions[i].split() if w.lower() not in DYNAMIC_WORDS)
        long_lines.append(
            f"Frame {i+1}: Scene={scenes[i]}, Event={events[i]}, "
            f"Caption={clean}, Landmark={landmarks[i]}, OCR='{ocr_txts[i]}'")

    # build whitelist (max 2 per bucket)
    wlist=[]
    for bag in sign_stats.values():
        wlist.extend(t for t,_ in sorted(bag.items(),key=lambda kv:(-kv[1][0],-kv[1][1]))[:2])
    whitelist_clause=("Use **only** these place names: "+", ".join(wlist)+".\n") if wlist else ""

    prompt=("Summarize the journey below in a factual travel‑diary style. "
            "Ignore transient objects (people, vehicles).\n"+
            whitelist_clause+"---\n"+
            "\n".join(long_lines)+"\n---\nJourney Summary:\n")

    inp=tok(prompt,return_tensors="pt").to(model.device)
    with torch.no_grad():
        out=model.generate(**inp,max_new_tokens=160,do_sample=False,temperature=0.0,top_p=1.0)
    return tok.batch_decode(out,skip_special_tokens=True)[0].split("Journey Summary:")[-1].strip()

# ───────────── table helper ───────────────────────────────────────────────
def summarise_journey(events,landmarks,captions,scenes,ocr_txts):
    return[
        {"step":i+1,"event":events[i],"scene":scenes[i],
         "description":f"{captions[i]}. Landmark: {landmarks[i]}. OCR: '{ocr_txts[i]}'"}
        for i in range(len(events))]
