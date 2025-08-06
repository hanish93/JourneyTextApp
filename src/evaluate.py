# src/evaluate.py
import sys
from pathlib import Path
import evaluate
from bert_score import BERTScorer
from tabulate import tabulate

def load_file(path):
    """
    Expects lines like:
      Clip 2
      The ... journey...
      Clip 3
      Another journey...
    Returns dict: { "Clip 2": text, "Clip 3": text, ... }
    """
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    data = {}
    key = None
    for line in lines:
        if line.lower().startswith("clip"):
            key = line
            data[key] = []
        elif key:
            data[key].append(line)
    return {k: " ".join(v) for k, v in data.items()}

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 -m src.evaluate GROUND_TRUTH.txt OUTPUT.txt")
        sys.exit(1)

    gt_path, out_path = sys.argv[1], sys.argv[2]
    refs = load_file(gt_path)
    hyps = load_file(out_path)

    clips = sorted(refs.keys())
    assert set(clips) == set(hyps.keys()), "Clip names in GT and Output must match!"

    # load all the metrics
    bleu   = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    chrf   = evaluate.load("chrf")
    rouge  = evaluate.load("rouge")
    bert   = BERTScorer(lang="en", rescale_with_baseline=True)

    table = []
    agg = {"BLEU":0, "METEOR":0, "chrF":0, "ROUGE-1":0, "ROUGE-L":0, "BERT-F1":0}
    N = len(clips)

    for clip in clips:
        ref = refs[clip]
        hyp = hyps[clip]

        b = bleu.compute(predictions=[hyp], references=[[ref]])["bleu"] * 100
        m = meteor.compute(predictions=[hyp], references=[[ref]])["meteor"] * 100
        c = chrf.compute(predictions=[hyp], references=[[ref]])["f1"] * 100

        r = rouge.compute(
            predictions=[hyp],
            references=[ref],
            rouge_types=["rouge1","rougeL"],
            use_stemmer=False
        )
        r1 = r["rouge1"] * 100
        rl = r["rougeL"] * 100

        P, R, F = bert.score([hyp], [ref])
        bf = float(F[0]) * 100

        agg["BLEU"]   += b
        agg["METEOR"] += m
        agg["chrF"]   += c
        agg["ROUGE-1"]+= r1
        agg["ROUGE-L"]+= rl
        agg["BERT-F1"]+= bf

        table.append([
            clip,
            f"{b:5.1f}",
            f"{m:5.1f}",
            f"{c:5.1f}",
            f"{r1:5.1f}",
            f"{rl:5.1f}",
            f"{bf:5.1f}",
        ])

    # add averages row
    table.append([
        "AVERAGE",
        f"{(agg['BLEU']/N):5.1f}",
        f"{(agg['METEOR']/N):5.1f}",
        f"{(agg['chrF']/N):5.1f}",
        f"{(agg['ROUGE-1']/N):5.1f}",
        f"{(agg['ROUGE-L']/N):5.1f}",
        f"{(agg['BERT-F1']/N):5.1f}",
    ])

    headers = ["Clip", "BLEU", "METEOR", "chrF", "ROUGE-1", "ROUGE-L", "BERT-F1"]
    print(tabulate(table, headers, tablefmt="github"))

if __name__ == "__main__":
    main()
