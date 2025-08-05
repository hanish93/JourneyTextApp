# src/evaluate_metrics.py

import sys
import argparse
from sacrebleu import corpus_bleu, corpus_chrf, corpus_meteor
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tabulate import tabulate

def load_lines(path):
    with open(path, encoding="utf8") as f:
        # assume one summary per line
        return [l.strip() for l in f if l.strip()]

def main():
    p = argparse.ArgumentParser(
        description="Compute BLEU, METEOR, chrF, ROUGE-1, ROUGE-L, and BERTScore-F1"
    )
    p.add_argument("reference", help="Ground truth file (one summary per line)")
    p.add_argument("hypothesis", help="System output file (one summary per line)")
    args = p.parse_args()

    refs = load_lines(args.reference)
    hyps = load_lines(args.hypothesis)
    if len(refs) != len(hyps):
        sys.exit(f"✖ line count mismatch: {len(refs)} refs vs. {len(hyps)} hyps")

    # BLEU
    bleu = corpus_bleu(hyps, [refs])
    # METEOR
    meteor = corpus_meteor(hyps, [refs])
    # chrF
    chrf = corpus_chrf(hyps, [refs])

    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)
    agg1 = {"precision":0,"recall":0,"fmeasure":0}
    aggL = {"precision":0,"recall":0,"fmeasure":0}
    for r,h in zip(refs, hyps):
        scores = scorer.score(r, h)
        for m in agg1:
            agg1[m] += getattr(scores["rouge1"], m)
            aggL[m] += getattr(scores["rougeL"], m)
    n = len(refs)
    for d in (agg1, aggL):
        for m in d:
            d[m] = 100 * d[m] / n

    # BERTScore
    P, R, F1 = bert_score(hyps, refs, lang="en", rescale_with_baseline=True)
    bert_f1 = 100 * F1.mean().item()

    # assemble table
    table = [
        ["BLEU",     f"{bleu.score:.1f}"],
        ["METEOR",   f"{meteor.score:.1f}"],
        ["chrF",     f"{chrf.score:.1f}"],
        ["ROUGE-1 R",f"{agg1['recall']:.1f}"],
        ["ROUGE-1 P",f"{agg1['precision']:.1f}"],
        ["ROUGE-1 F1",f"{agg1['fmeasure']:.1f}"],
        ["ROUGE-L R",f"{aggL['recall']:.1f}"],
        ["ROUGE-L P",f"{aggL['precision']:.1f}"],
        ["ROUGE-L F1",f"{aggL['fmeasure']:.1f}"],
        ["BERTScore-F1", f"{bert_f1:.1f}"],
    ]

    print("\n"+tabulate(table, headers=["Metric","Score"], tablefmt="github")+"\n")

if __name__ == "__main__":
    main()
