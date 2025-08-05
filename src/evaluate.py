#!/usr/bin/env python3
import sys, argparse
from tabulate import tabulate
import sacrebleu
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import BERTScorer

def load_nonempty_lines(path):
    with open(path, encoding="utf8") as f:
        return [l.rstrip() for l in f if l.strip()]

def main():
    p = argparse.ArgumentParser(
        description="Evaluate journeys vs. ground truth with multiple metrics"
    )
    p.add_argument("refs", help="Ground truth file (one journey per line)")
    p.add_argument("hyps", help="Output    file (one journey per line)")
    args = p.parse_args()

    refs = load_nonempty_lines(args.refs)
    hyps = load_nonempty_lines(args.hyps)

    if len(refs) != len(hyps):
        print(f"⚠️  Warning: {len(refs)} references vs {len(hyps)} outputs; pairing up to {min(len(refs), len(hyps))}")
    N = min(len(refs), len(hyps))

    # Initialize scorers once
    rouge = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    # We'll batch BERTScore on the trimmed lists
    batch_refs = refs[:N]
    batch_hyps = hyps[:N]
    _, _, bert_f = bert_scorer.score(batch_hyps, batch_refs)

    rows = []
    sums = {m:0.0 for m in ["BLEU","METEOR","chrF","ROUGE-1","ROUGE-L","BERT-F1"]}

    for i, (r, h) in enumerate(zip(batch_refs, batch_hyps), start=1):
        # BLEU (sentence-level)
        bleu = sacrebleu.sentence_bleu(h, [r]).score

        # METEOR (nltk expects token lists)
        r_tok = word_tokenize(r)
        h_tok = word_tokenize(h)
        meteor = single_meteor_score(r_tok, h_tok) * 100

        # chrF
        chrf = sacrebleu.sentence_chrf(h, [r]).score

        # ROUGE-1 & ROUGE-L (F1 * 100)
        sc = rouge.score(r, h)
        r1 = sc["rouge1"].fmeasure * 100
        rL = sc["rougeL"].fmeasure * 100

        # BERTScore-F1
        bf1 = bert_f[i-1].item() * 100

        # accumulate
        for k,v in zip(sums.keys(), [bleu, meteor, chrf, r1, rL, bf1]):
            sums[k] += v

        rows.append([
            f"clip_{i}",
            f"{bleu:6.1f}",
            f"{meteor:6.1f}",
            f"{chrf:6.1f}",
            f"{r1:6.1f}",
            f"{rL:6.1f}",
            f"{bf1:6.1f}",
        ])

    # add average row
    avg = ["AVERAGE"] + [f"{(sums[k]/N):6.1f}" for k in sums]
    rows.append(avg)

    print(tabulate(
        rows,
        headers=["Clip","BLEU","METEOR","chrF","ROUGE-1","ROUGE-L","BERT-F1"],
        tablefmt="github",
    ))

if __name__=="__main__":
    main()
