#!/usr/bin/env python3
# evaluate_all.py

import sys
import argparse
from evaluate import load  # HuggingFace evaluate

def load_lines(path):
    with open(path, encoding="utf8") as f:
        return [l.strip() for l in f if l.strip()]

def main():
    p = argparse.ArgumentParser(
        description="Compute BLEU, ROUGE-1 & ROUGE-L against references"
    )
    p.add_argument("reference", help="Ground truth file (one summary per line)")
    p.add_argument("hypothesis", help="Output file    (one summary per line)")
    args = p.parse_args()

    refs = load_lines(args.reference)
    hyps = load_lines(args.hypothesis)
    if len(refs) != len(hyps):
        sys.exit(f"✗ line count mismatch: {len(refs)} refs vs {len(hyps)} hyps")

    # prepare for corpus metrics
    # BLEU wants list of hyps + list of list-of-refs
    bleu = load("bleu")
    bleu_res = bleu.compute(predictions=hyps, references=[[r] for r in refs])

    rouge = load("rouge")
    rouge_res = rouge.compute(predictions=hyps, references=refs)

    # print table
    print("\n╭──────────────────── Evaluator ────────────────────╮")
    print("│ Metric     │ Score                               │")
    print("├────────────┼─────────────────────────────────────┤")
    print(f"│ BLEU       │ {bleu_res['bleu']*100:6.2f}                              │")
    print(f"│ ROUGE-1 F1 │ {rouge_res['rouge1']*100:6.2f}                              │")
    print(f"│ ROUGE-L F1 │ {rouge_res['rougeL']*100:6.2f}                              │")
    print("╰──────────────────────────────────────────────────╯\n")

if __name__ == "__main__":
    main()
