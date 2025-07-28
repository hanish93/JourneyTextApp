import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# 1) TEMPORAL SMOOTHER (CNN+LSTM) FOR EVENTS & SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
class TemporalSmoother(nn.Module):
    """
    Takes a sequence of per-frame detections (one-hot for events & signals)
    and outputs smoothed class predictions frame-by-frame.
    """
    def __init__(self, evt_classes=5, sig_classes=3, hidden_size=64, layers=1):
        super().__init__()
        # small CNN to embed each timestep (we just use a linear proj here)
        self.embed = nn.Linear(evt_classes + sig_classes, hidden_size)
        self.lstm  = nn.LSTM(hidden_size, hidden_size, layers, batch_first=True)
        self.evt_fc = nn.Linear(hidden_size, evt_classes)
        self.sig_fc = nn.Linear(hidden_size, sig_classes)

    def forward(self, x):
        # x: (B, T, evt+sig)
        h = self.embed(x)                # (B, T, H)
        y, _ = self.lstm(h)              # (B, T, H)
        evt_logits = self.evt_fc(y)      # (B, T, evt_classes)
        sig_logits = self.sig_fc(y)      # (B, T, sig_classes)
        return evt_logits.softmax(-1), sig_logits.softmax(-1)

_smoother = None

def load_temporal_smoother(device):
    global _smoother
    if _smoother is None:
        m = TemporalSmoother(evt_classes=5, sig_classes=3).to(device)
        path = "models/temporal.pt"
        if os.path.exists(path):
            state = torch.load(path, map_location=device)
            m.load_state_dict(state)
        else:
            print("[WARN] temporal.pt not found; falling back to debouncing")
        m.eval()
        _smoother = m
    return _smoother

def smooth_predictions(raw_events, raw_signals, device):
    """
    raw_events: list of strings in ["drive","stop","turn_left","turn_right","passed"]
    raw_signals: list of strings in [None,"red","green"]
    Returns lists of smoothed events & signals.
    """
    # maps
    evt_map = {"drive":0,"stop":1,"turn_left":2,"turn_right":3,"passed":4}
    sig_map = {None:0,"red":1,"green":2}

    T = len(raw_events)
    x = torch.zeros(1, T, len(evt_map)+len(sig_map), device=device)
    for i,(e,s) in enumerate(zip(raw_events,raw_signals)):
        xe = evt_map["passed"] if e.startswith("passed ") else evt_map[e]
        xs = sig_map[s]
        x[0,i,xe] = 1
        x[0,i,len(evt_map)+xs] = 1

    smoother = load_temporal_smoother(device)
    evt_sm, sig_sm = smoother(x)  # (1,T,classes)

    # pick argmax
    evt_idx = evt_sm.argmax(-1)[0].cpu().tolist()
    sig_idx = sig_sm.argmax(-1)[0].cpu().tolist()

    # reverse maps
    inv_evt = {v:k for k,v in evt_map.items()}
    inv_sig = {v:k for k,v in sig_map.items()}

    out_ev  = []
    for i,e in enumerate(evt_idx):
        if e==evt_map["passed"]:
            # preserve text
            text = raw_events[i].split(" ",1)[1]
            out_ev.append(f"passed {text}")
        else:
            out_ev.append(inv_evt[e])

    out_sig = [inv_sig[s] for s in sig_idx]
    return out_ev, out_sig

# ─────────────────────────────────────────────────────────────────────────────
# (old detection & debouncing code below)
# ─────────────────────────────────────────────────────────────────────────────
def extract_frames(path, fps=1):
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(".jpg"):
                img = cv2.imread(os.path.join(path, fn))
                if img is not None:
                    yield img
        return
    cap = cv2.VideoCapture(path)
    nat = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(nat/fps))
    idx, ok, frame = 0, *cap.read()
    while ok:
        if idx % step == 0:
            yield frame
        ok, frame = cap.read()
        idx += 1
    cap.release()

def detect_event_for_frame(prev_gray, cur_gray, dx_thresh=2.5, stop_thresh=0.3):
    if prev_gray is None:
        return "drive"
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None, .5,3,15,3,5,1.2,0
    )
    dx  = flow[...,0]
    dxm = float(np.median(dx))
    mag = float(np.linalg.norm(flow,axis=2).mean())
    if mag < stop_thresh:    return "stop"
    if dxm > dx_thresh:      return "turn_right"
    if dxm < -dx_thresh:     return "turn_left"
    return "drive"

def debounce_lane_changes(events, window=5):
    out = list(events); n=len(events)
    for i,e in enumerate(events):
        if e in ("turn_left","turn_right"):
            cnt = sum(1 for j in range(max(0,i-window),min(n,i+window+1)) if events[j]==e)
            if cnt<2: out[i]="drive"
    return out

def get_yolo_model(device):
    return YOLO("yolov8n").to(device).half()

def detect_signal_color(frame, yolo, conf=0.1):
    r = yolo(frame, conf=conf, verbose=False)[0]
    boxes=[]
    for b in r.boxes:
        cls=yolo.model.names[int(b.cls[0])]
        if cls=="traffic light":
            x1,y1,x2,y2=map(int,b.xyxy[0].cpu().numpy())
            boxes.append((x1,y1,x2,y2))
    if boxes:
        x1,y1,x2,y2=max(boxes,key=lambda bb:(bb[2]-bb[0])*(bb[3]-bb[1]))
        crop=frame[y1:y2,x1:x2]
    else:
        h,w=frame.shape[:2]
        top=int(0.2*h); left=int(0.3*w); right=int(0.7*w)
        crop=frame[0:top,left:right]
    if crop.size==0: return None
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    m1=cv2.inRange(hsv,(0,80,80),(10,255,255))
    m2=cv2.inRange(hsv,(160,80,80),(180,255,255))
    red=cv2.bitwise_or(m1,m2)
    green=cv2.inRange(hsv,(40,80,80),(85,255,255))
    rc, gc = int(cv2.countNonZero(red)), int(cv2.countNonZero(green))
    if max(rc,gc)<100: return None
    return "red" if rc>gc else "green"

def debounce_signals(states, window=3):
    out=[None]*len(states); n=len(states)
    for i,s in enumerate(states):
        if s in ("red","green"):
            cnt=sum(1 for j in range(max(0,i-window),min(n,i+window+1)) if states[j]==s)
            if cnt>=3: out[i]=s
    return out

def generate_long_summary(events, signals, *args, **kwargs):
    parts, last_sig = [], None
    for ev,sig in zip(events,signals):
        if sig=="red" and last_sig!="red":
            parts.append("stopped at the red light")
        if sig=="green" and last_sig=="red":
            parts.append("when it turned green, I drove on")
        last_sig = sig or last_sig
        if ev.startswith("passed "):
            parts.append(f"passed {ev.split(' ',1)[1]}")
        if ev=="turn_left":
            parts.append("turned left")
        if ev=="turn_right":
            parts.append("took a slight right")
    if not parts or not parts[0].startswith("stopped"):
        parts.insert(0,"drove straight")
    out=[parts[0]]
    for p in parts[1:]:
        if p!=out[-1]: out.append(p)
    sent=out[0].capitalize()
    for p in out[1:]:
        sent+=" and "+p
    if events and events[-1]=="drive":
        sent+=" continued straight."
    elif not sent.endswith("."):
        sent+="."
    return sent
