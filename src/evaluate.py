# src/evaluate_metrics.py

import sys
import argparse
from sacrebleu.metrics import BLEU, CHRF
from nltk.translate.meteor_score import meteor_score
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

    # chrF
    chrf = CHRF()
    chrf_score = chrf.corpus_score(hyps, [refs]).score

    # METEOR (average over sentences)
    meteor_scores = [meteor_score([r], h) for r, h in zip(refs, hyps)]
    meteor_score_avg = 100 * sum(meteor_scores) / len(meteor_scores)

    # ROUGE-1 & ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)
    agg = {"rouge1": {"p":0,"r":0,"f":0},
           "rougeL": {"p":0,"r":0,"f":0}}
    for ref, hyp in zip(refs, hyps):
        scores = scorer.score(ref, hyp)
        for key in agg:
            agg[key]["p"] += scores[key].precision
            agg[key]["r"] += scores[key].recall
            agg[key]["f"] += scores[key].fmeasure
    n = len(refs)
    for key in agg:
        for m in agg[key]:
            agg[key][m] = 100 * agg[key][m] / n

    # BERTScore-F1
    P, R, F1 = bert_score(hyps, refs, lang="en", rescale_with_baseline=True)
    bert_f1 = 100 * F1.mean().item()

    # Tabulate results
    table = [
        ["BLEU",         f"{bleu_score:.1f}"],
        ["METEOR",       f"{meteor_score_avg:.1f}"],
        ["chrF",         f"{chrf_score:.1f}"],
        ["ROUGE-1 P",    f"{agg['rouge1']['p']:.1f}"],
        ["ROUGE-1 R",    f"{agg['rouge1']['r']:.1f}"],
        ["ROUGE-1 F1",   f"{agg['rouge1']['f']:.1f}"],
        ["ROUGE-L P",    f"{agg['rougeL']['p']:.1f}"],
        ["ROUGE-L R",    f"{agg['rougeL']['r']:.1f}"],
        ["ROUGE-L F1",   f"{agg['rougeL']['f']:.1f}"],
        ["BERTScore-F1", f"{bert_f1:.1f}"],
    ]
    print("\n" + tabulate(table, headers=["Metric","Score"], tablefmt="github") + "\n")

if __name__ == "__main__":
    main()
