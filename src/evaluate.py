# src/evaluate.py

import sys
import argparse
from sacrebleu import sentence_bleu
from rouge_score import rouge_scorer
from tabulate import tabulate

def load_lines(path):
    with open(path, encoding="utf8") as f:
        # strip out empty lines, keep order
        return [l.strip() for l in f.readlines() if l.strip()]

def main():
    p = argparse.ArgumentParser(
        description="Compare Ground_Truth.txt vs. Output.txt and report BLEU/ROUGE scores"
    )
    p.add_argument("ground_truth", help="Path to Ground_Truth.txt")
    p.add_argument("output",       help="Path to Output.txt")
    args = p.parse_args()

    refs = load_lines(args.ground_truth)
    hyps = load_lines(args.output)

    if len(refs) != len(hyps):
        print(f"[!] mismatch: {len(refs)} refs vs. {len(hyps)} outputs", file=sys.stderr)
        sys.exit(1)

    scorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)

    table = []
    totals = {"bleu":0.0, "rouge1":0.0, "rougeL":0.0}
    n = len(refs)

    for idx, (ref, hyp) in enumerate(zip(refs, hyps), start=2):
        # sentence-level BLEU
        bleu = sentence_bleu(hyp, [ref]).score
        # rouge
        scores = scorer.score(ref, hyp)
        r1 = scores["rouge1"].fmeasure * 100
        rL = scores["rougeL"].fmeasure * 100

        totals["bleu"]   += bleu
        totals["rouge1"] += r1
        totals["rougeL"] += rL

        table.append((
            f"clip_{idx}",
            f"{bleu:5.1f}",
            f"{r1:5.1f}",
            f"{rL:5.1f}",
        ))

    # add averages row
    table.append((
        "AVERAGE",
        f"{totals['bleu']/n:5.1f}",
        f"{totals['rouge1']/n:5.1f}",
        f"{totals['rougeL']/n:5.1f}"
    ))

    print(tabulate(
        table,
        headers=["Clip","BLEU","ROUGE-1","ROUGE-L"],
        tablefmt="github"
    ))

if __name__ == "__main__":
    main()
