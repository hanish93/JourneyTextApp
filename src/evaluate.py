import re, sys
from pathlib import Path
from tabulate import tabulate
import evaluate

CLIP_RE = re.compile(r"^Clip\s+(\d+)", re.IGNORECASE)

def load_journeys(path: Path):
    """
    Parse a file containing lines like:
      Clip 2
      Journey text...
      Clip 3
      Another journey...
    Returns a dict: { "clip_2": "Journey text...", ... }
    """
    lines = path.read_text().splitlines()
    out, current = {}, None
    for l in lines:
        m = CLIP_RE.match(l)
        if m:
            current = f"clip_{m.group(1)}"
            out[current] = ""
        elif current and l.strip():
            # first non-empty line after the header is the journey
            if not out[current]:
                out[current] = l.strip()
    return out

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.evaluate Ground_Truth.txt Output.txt")
        sys.exit(1)

    gt  = load_journeys(Path(sys.argv[1]))
    pred = load_journeys(Path(sys.argv[2]))
    clips = sorted(set(gt) & set(pred))
    if not clips:
        print("ℹ️  No overlapping clips found.")
        sys.exit(0)

    # load metrics
    bleu   = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    chrf   = evaluate.load("chrf")
    rouge  = evaluate.load("rouge")
    bert   = evaluate.load("bertscore")

    rows = []
    agg = {m: [] for m in ["BLEU","METEOR","chrF","ROUGE-1","ROUGE-L","BERT-F1"]}

    for clip in clips:
        ref = gt[clip]
        hyp = pred[clip]

        b = bleu.compute(predictions=[hyp], references=[[ref]])["bleu"] * 100
        m = meteor.compute(predictions=[hyp], references=[[ref]])["meteor"] * 100
        c = chrf.compute(predictions=[hyp], references=[[ref]])["score"] * 100

        r = rouge.compute(
            predictions=[hyp], references=[ref],
            rouge_types=["rouge1","rougeL"], use_aggregator="avg"
        )
        r1 = r["rouge1"].mid.fmeasure * 100
        rl = r["rougeL"].mid.fmeasure * 100

        br = bert.compute(predictions=[hyp], references=[ref], lang="en")
        bf = br["f1"][0] * 100

        rows.append([
            clip, f"{b:5.1f}", f"{m:5.1f}", f"{c:5.1f}",
            f"{r1:5.1f}", f"{rl:5.1f}", f"{bf:5.1f}"
        ])

        for k,v in zip(agg.keys(), [b,m,c,r1,rl,bf]):
            agg[k].append(v)

    # AVERAGE row
    avg = ["AVERAGE"] + [f"{sum(agg[k])/len(agg[k]):5.1f}" for k in agg]
    rows.append(avg)

    print(tabulate(
        rows,
        headers=["Clip","BLEU","METEOR","chrF","ROUGE-1","ROUGE-L","BERT-F1"],
        tablefmt="github",
        stralign="center"
    ))

if __name__ == "__main__":
    main()
