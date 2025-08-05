#!/usr/bin/env python3
import sys
import argparse
from tabulate import tabulate
import sacrebleu
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize
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
    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    # Precompute BERTScore
    _, _, bert_f = bert_scorer.score(hyps, refs)

    rows = []
    sums = {k:0.0 for k in ["BLEU","METEOR","chrF","ROUGE-1","ROUGE-L","BERT-F1"]}
    P = len(refs)

    for i,(r,h) in enumerate(zip(refs,hyps), start=1):
        # BLEU
        bleu = sacrebleu.sentence_bleu(h, [r]).score

        # METEOR (tokenized)
        r_tok = word_tokenize(r)
        h_tok = word_tokenize(h)
        meteor = single_meteor_score(r_tok, h_tok) * 100

        # chrF
        chrf = sacrebleu.CHRF().score(h, [r])

        # ROUGE-1 & ROUGE-L
        scores = scorer.score(r, h)
        r1 = scores['rouge1'].fmeasure * 100
        rL = scores['rougeL'].fmeasure * 100

        # BERTScore
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

    avg = ["AVERAGE"] + [f"{(sums[k]/P):6.1f}" for k in sums]
    rows.append(avg)

    print(tabulate(
        rows,
        headers=["Clip","BLEU","METEOR","chrF","ROUGE-1","ROUGE-L","BERT-F1"],
        tablefmt="github"
    ))

if __name__=="__main__":
    main()
