#!/usr/bin/env python3
# evaluate_full.py

import sys
import argparse
from evaluate import load
from bert_score import BERTScorer
from tabulate import tabulate

def load_lines(path):
    with open(path, encoding="utf8") as f:
        return [l.strip() for l in f if l.strip()]

def main():
    p = argparse.ArgumentParser(
        description="Per‐clip & corpus evaluation: BLEU, METEOR, chrF, ROUGE-1, ROUGE-L, BERTScore"
    )
    p.add_argument("ref", help="Ground truth file (one line per clip)")
    p.add_argument("hyp", help="Output file (one line per clip)")
    args = p.parse_args()

    refs = load_lines(args.ref)
    hyps = load_lines(args.hyp)
    if len(refs) != len(hyps):
        sys.exit(f"❌ Line count mismatch: {len(refs)} refs vs {len(hyps)} hyps")

    # Load metrics once
    bleu    = load("bleu")
    meteor  = load("meteor")
    chrf    = load("chrf")
    rouge   = load("rouge")
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    # Containers
    rows = []
    accum = {
        "BLEU": [], "METEOR": [], "chrF": [],
        "ROUGE-1": [], "ROUGE-L": [], "BERT-F1": []
    }

    # Compute per‐clip
    P = len(refs)
    # BERTScore wants all at once, so accumulate pairs
    bs_p, bs_r, bs_f = bert_scorer.score(hyps, refs)

    for i,(r,h) in enumerate(zip(refs, hyps), start=1):
        # Sentence‐level BLEU (1‐4gram smoothing)
        bleu_res = bleu.compute(predictions=[h], references=[[r]])
        b = bleu_res["bleu"] * 100

        # METEOR
        m = meteor.compute(predictions=[h], references=[r])["meteor"] * 100

        # chrF
        c = chrf.compute(predictions=[h], references=[r])["chrf"] * 100

        # ROUGE‐1 & ROUGE‐L (all give recall, precision, f1)
        rg = rouge.compute(predictions=[h], references=[r])
        r1 = rg["rouge1"].mid.fmeasure * 100
        rL = rg["rougeL"].mid.fmeasure * 100

        # BERTScore‐F1
        f1 = bs_f[i-1].item() * 100

        # record
        rows.append([f"clip_{i}", f"{b:5.1f}", f"{m:6.1f}", f"{c:5.1f}", f"{r1:6.1f}", f"{rL:6.1f}", f"{f1:6.1f}"])
        accum["BLEU"].append(b)
        accum["METEOR"].append(m)
        accum["chrF"].append(c)
        accum["ROUGE-1"].append(r1)
        accum["ROUGE-L"].append(rL)
        accum["BERT-F1"].append(f1)

    # Append average row
    avg = ["AVERAGE"] + [f"{sum(accum[k])/P:6.1f}" for k in accum]
    rows.append(avg)

    # Print table
    headers = ["Clip", "BLEU", "METEOR", "chrF", "ROUGE-1", "ROUGE-L", "BERT-F1"]
    print(tabulate(rows, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    main()
