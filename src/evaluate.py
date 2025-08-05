#!/usr/bin/env python3
# evaluate_full.py

import sys
import argparse
from evaluate import load

def load_lines(path):
    with open(path, encoding="utf8") as f:
        return [l.strip() for l in f if l.strip()]

def main():
    p = argparse.ArgumentParser(
        description="Compute BLEU, METEOR, chrF, ROUGE-1/L & BERTScore on summaries"
    )
    p.add_argument("reference", help="Ground truth file (one line per clip)")
    p.add_argument("hypothesis", help="Output file    (one line per clip)")
    args = p.parse_args()

    refs = load_lines(args.reference)
    hyps = load_lines(args.hypothesis)
    if len(refs) != len(hyps):
        sys.exit(f"Line count mismatch: {len(refs)} refs vs {len(hyps)} hyps")

    # 1) BLEU (corpus-level, up to 4-gram)
    bleu = load("bleu")
    bleu_res = bleu.compute(predictions=hyps, references=[[r] for r in refs])

    # 2) METEOR (synonym-aware)
    meteor = load("meteor")
    meteor_res = meteor.compute(predictions=hyps, references=refs)

    # 3) chrF (character n-gram F-score)
    chrf = load("chrf")
    chrf_res = chrf.compute(predictions=hyps, references=refs)

    # 4) ROUGE-1 & ROUGE-L
    rouge = load("rouge")
    rouge_res = rouge.compute(predictions=hyps, references=refs)

    # 5) BERTScore-F1
    bert = load("bertscore")
    bert_res = bert.compute(predictions=hyps, references=refs,
                            lang="en", rescale_with_baseline=True)

    # Print nicely
    print("\n╭────────────────── Full Evaluation ──────────────────╮")
    print("│ Metric        │ Score                              │")
    print("├───────────────┼────────────────────────────────────┤")
    print(f"│ BLEU          │ {bleu_res['bleu']*100:6.2f}                           │")
    print(f"│ METEOR        │ {meteor_res['meteor']*100:6.2f}                           │")
    print(f"│ chrF          │ {chrf_res['chrF']*100:6.2f}                           │")
    print(f"│ ROUGE-1 F1    │ {rouge_res['rouge1']*100:6.2f}                           │")
    print(f"│ ROUGE-L F1    │ {rouge_res['rougeL']*100:6.2f}                           │")
    print(f"│ BERTScore F1  │ {sum(bert_res['f1'])/len(bert_res['f1'])*100:6.2f}                           │")
    print("╰────────────────────────────────────────────────────╯\n")

if __name__ == "__main__":
    main()
