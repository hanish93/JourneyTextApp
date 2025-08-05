#!/usr/bin/env python3
import sys
import argparse
from tabulate import tabulate
import sacrebleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from bert_score import BERTScorer

def load_lines(path):
    with open(path, encoding='utf8') as f:
        return [l.strip() for l in f if l.strip()]

def main():
    p = argparse.ArgumentParser(
        description="Evaluate summarization: BLEU, METEOR, chrF, ROUGE-1/ROUGE-L, BERTScore"
    )
    p.add_argument("refs", help="Ground truth file (one line per clip)")
    p.add_argument("hyps", help="Output file    (one line per clip)")
    args = p.parse_args()

    refs = load_lines(args.refs)
    hyps = load_lines(args.hyps)
    if len(refs) != len(hyps):
        sys.exit(f"❌ Mismatch lines: {len(refs)} refs vs {len(hyps)} hyps")

    # Prepare metrics
    # 1) BLEU (sentence-level, sacrebleu)
    # 2) METEOR (NLTK)
    # 3) chrF (sacrebleu)
    # 4) ROUGE-1 & ROUGE-L (rouge-score)
    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    # 5) BERTScore
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    # Precompute BERTScore for all at once
    P = len(refs)
    P_scores = bert_scorer.score(hyps, refs)
    _, _, bert_f = P_scores

    rows = []
    sums = {k:0.0 for k in ["BLEU","METEOR","chrF","ROUGE-1","ROUGE-L","BERT-F1"]}

    for i,(r,h) in enumerate(zip(refs,hyps), start=1):
        # BLEU (1-4 gram, smoothing default)
        bleu = sacrebleu.sentence_bleu(h, [r]).score

        # METEOR
        meteor = single_meteor_score(r, h) * 100

        # chrF
        chrf = sacrebleu.CHRF().score(h, [r])

        # ROUGE-1 & ROUGE-L F1
        scores = scorer.score(r, h)
        r1 = scores['rouge1'].fmeasure * 100
        rL = scores['rougeL'].fmeasure * 100

        # BERTScore F1
        bf1 = bert_f[i-1].item() * 100

        rows.append([
            f"clip_{i}",
            f"{bleu:5.1f}",
            f"{meteor:6.1f}",
            f"{chrf:5.1f}",
            f"{r1:6.1f}",
            f"{rL:6.1f}",
            f"{bf1:6.1f}",
        ])
        for k,v in zip(sums.keys(), [bleu,meteor,chrf,r1,rL,bf1]):
            sums[k] += v

    # average row
    avg = ["AVERAGE"] + [f"{(sums[k]/P):6.1f}" for k in sums]
    rows.append(avg)

    print(tabulate(
        rows,
        headers=["Clip","BLEU","METEOR","chrF","ROUGE-1","ROUGE-L","BERT-F1"],
        tablefmt="github"
    ))


if __name__=="__main__":
    main()
