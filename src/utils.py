import os
import cv2
import numpy as np
from ultralytics import YOLO

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
    step = max(1, round(nat / fps))
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
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                        0.5,3,15,3,5,1.2,0)
    dxm = float(np.median(flow[...,0]))
    mag = float(np.linalg.norm(flow,axis=2).mean())
    if mag < stop_thresh:    return "stop"
    if dxm > dx_thresh:      return "turn_right"
    if dxm < -dx_thresh:     return "turn_left"
    return "drive"

def debounce_lane_changes(events, window=5):
    out = list(events)
    n = len(events)
    for i,e in enumerate(events):
        if e in ("turn_left","turn_right"):
            cnt = sum(1 for j in range(max(0,i-window),
                                       min(n,i+window+1))
                      if events[j]==e)
            if cnt < 2:
                out[i] = "drive"
    return out

def get_yolo_model(device):
    return YOLO("yolov8n").to(device).half()

def detect_signal_color(frame, yolo, conf=0.1):
    r = yolo(frame, conf=conf, verbose=False)[0]
    boxes = []
    for b in r.boxes:
        cls = yolo.model.names[int(b.cls[0])]
        if cls == "traffic light":
            x1,y1,x2,y2 = map(int,b.xyxy[0].cpu().numpy())
            boxes.append((x1,y1,x2,y2))
    if boxes:
        x1,y1,x2,y2 = max(boxes, key=lambda bb:(bb[2]-bb[0])*(bb[3]-bb[1]))
        crop = frame[y1:y2, x1:x2]
    else:
        h,w = frame.shape[:2]
        top = int(0.2*h)
        left, right = int(0.3*w), int(0.7*w)
        crop = frame[0:top, left:right]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv, (0,80,80), (10,255,255))
    r2 = cv2.inRange(hsv, (160,80,80), (180,255,255))
    red = cv2.bitwise_or(r1,r2)
    green = cv2.inRange(hsv, (40,80,80), (85,255,255))
    rc, gc = int(cv2.countNonZero(red)), int(cv2.countNonZero(green))
    if max(rc,gc) < 100:
        return None
    return "red" if rc>gc else "green"

def debounce_signals(states, window=3):
    out = [None]*len(states)
    n = len(states)
    for i,s in enumerate(states):
        if s in ("red","green"):
            cnt = sum(1 for j in range(max(0,i-window),
                                       min(n,i+window+1))
                      if states[j]==s)
            if cnt >= 3:
                out[i] = s
    return out

def generate_long_summary(events, signals):
    # 1) find all landmark passes
    passes = [(i, e.split(" ",1)[1])
              for i,e in enumerate(events)
              if e.startswith("passed ")]
    # 2) find all red-light spans
    spans = []
    in_red = False
    for i,s in enumerate(signals):
        if s=="red" and not in_red:
            start = i
            in_red = True
        if in_red and s!="red":
            spans.append((start, i-1))
            in_red = False
    if in_red:
        spans.append((start, len(signals)-1))

    parts = []

    # Segment 1: from 0 to first pass
    first_pass_i, _ = passes[0]
    # if red in [0, first_pass_i)
    if any(signals[j]=="red" for j in range(0, first_pass_i)):
        parts.append("I stopped at the red light")
    # all passes up to first_pass_i
    names = [name for i,name in passes if i <= first_pass_i]
    parts.append("then passed " + " and ".join(names))

    # Segment 2: after first pass to next red
    seg2_start = first_pass_i+1
    next_red = next((st for st,en in spans if st>=seg2_start), None)
    if next_red:
        parts.append("then continued straight for a while")
    else:
        next_red = seg2_start

    # Segment 3: at that red run
    span3 = next((sp for sp in spans if sp[0]==next_red), None)
    if span3:
        parts.append("then stopped at the red light again")
        # find the pass immediately after span3
        after3 = next((name for i,name in passes if i>span3[1]), None)
        parts.append(f"then once it turned green, crossed {after3}")

    # Segment 4: between that and next red
    if span3:
        seg4_start = span3[1]+1
    else:
        seg4_start = seg2_start
    # next red after span3
    span4 = next((sp for sp in spans if sp[0]>=seg4_start and sp!=span3), None)
    if span4:
        # find the pass before span4
        before4 = next((name for i,name in reversed(passes) if i<span4[0]), None)
        parts.append(f"then drove on past {before4}")
    else:
        # if no more red, take next pass
        remaining = [name for i,name in passes if i>first_pass_i]
        if remaining:
            parts.append(f"then drove on past {remaining[1]}")

    # Segment 5: final red & pass
    last_span = spans[-1] if spans else None
    if last_span and last_span[0] > (span3[1] if span3 else 0):
        parts.append("then stopped at the red light once more")
    last_name = passes[-1][1]
    parts.append(f"then passed {last_name}")
    parts.append("and continued straight")

    # glue into one sentence
    sent = parts[0].capitalize()
    for p in parts[1:]:
        sent += ", " + p
    return sent + "."
