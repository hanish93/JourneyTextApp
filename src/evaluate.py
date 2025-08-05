# src/evaluate_metrics.py

import sys
import argparse
from sacrebleu.metrics import BLEU, CHRF, METEOR
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tabulate import tabulate

def load_lines(path):
    with open(path, encoding="utf8") as f:
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
        sys.exit(f"Line count mismatch: {len(refs)} vs. {len(hyps)}")

    # BLEU
    bleu = BLEU()
    bleu_score = bleu.corpus_score(hyps, [refs]).score

    # METEOR
    meteor = METEOR()
    meteor_score = meteor.corpus_score(hyps, [refs]).score

    # chrF
    chrf = CHRF()
    chrf_score = chrf.corpus_score(hyps, [refs]).score

    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)
    agg1 = {"p":0,"r":0,"f":0}
    aggL = {"p":0,"r":0,"f":0}
    for ref, hyp in zip(refs, hyps):
        sc = scorer.score(ref, hyp)
        agg1["p"] += sc["rouge1"].precision
        agg1["r"] += sc["rouge1"].recall
        agg1["f"] += sc["rouge1"].fmeasure
        aggL["p"] += sc["rougeL"].precision
        aggL["r"] += sc["rougeL"].recall
        aggL["f"] += sc["rougeL"].fmeasure
    n = len(refs)
    for m in agg1: agg1[m] = 100 * agg1[m] / n
    for m in aggL: aggL[m] = 100 * aggL[m] / n

    # BERTScore
    P, R, F1 = bert_score(hyps, refs, lang="en", rescale_with_baseline=True)
    bert_f1 = 100 * F1.mean().item()

    # display
    table = [
        ["BLEU",        f"{bleu_score:.1f}"],
        ["METEOR",      f"{meteor_score:.1f}"],
        ["chrF",        f"{chrf_score:.1f}"],
        ["ROUGE-1 P",   f"{agg1['p']:.1f}"],
        ["ROUGE-1 R",   f"{agg1['r']:.1f}"],
        ["ROUGE-1 F1",  f"{agg1['f']:.1f}"],
        ["ROUGE-L P",   f"{aggL['p']:.1f}"],
        ["ROUGE-L R",   f"{aggL['r']:.1f}"],
        ["ROUGE-L F1",  f"{aggL['f']:.1f}"],
        ["BERTScore-F1",f"{bert_f1:.1f}"],
    ]
    print("\n" + tabulate(table, headers=["Metric","Score"], tablefmt="github") + "\n")

if __name__ == "__main__":
    main()
