# evaluate_journeys.py
import sys
import re
from sacrebleu import sentence_bleu
from tabulate import tabulate

def parse_file(path):
    """
    Expects sections like:
      Clip 2
      The vehicle turned right ...
    
    Returns dict: { '2': "The vehicle turned right ...", ... }
    """
    clips = {}
    lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    i = 0
    while i < len(lines):
        m = re.match(r'Clip\s+(\d+)', lines[i])
        if m:
            clip_id = m.group(1)
            # next non-blank line is the text
            if i+1 < len(lines):
                clips[clip_id] = lines[i+1]
            i += 2
        else:
            i += 1
    return clips

def main(gt_path, out_path):
    gt = parse_file(gt_path)
    out = parse_file(out_path)
    rows = []
    for clip_id in sorted(gt, key=lambda x: int(x)):
        ref = gt[clip_id]
        hyp = out.get(clip_id, "")
        # sacrebleu expects list of references, each being a list of sentences
        bleu = sentence_bleu(hyp, [ref], smooth_method="exp").score
        rows.append([clip_id, f"{bleu:.2f}", ref, hyp])
    print(tabulate(rows,
                   headers=["Clip", "BLEU", "Reference", "Prediction"],
                   tablefmt="github"))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_journeys.py Ground_Truth.txt Output.txt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
