import sys, re
from pathlib import Path

import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, ROUGE
from bert_score import BERTScorer
from tabulate import tabulate


def load_clips(path):
    """
    Parse:
      Clip 2
      text...
      Clip 3
      text...
    into { "Clip 2": "text...", ... }
    """
    lines = Path(path).read_text().splitlines()
    clips, cur, buf = {}, None, []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if re.match(r"^Clip\s+\d+", ln):
            if cur:
                clips[cur] = " ".join(buf).strip()
            cur, buf = ln, []
        elif cur:
            buf.append(ln)
    if cur:
        clips[cur] = " ".join(buf).strip()
    return clips


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.evaluate Ground_Truth.txt Output.txt")
        sys.exit(1)

    ref_clips = load_clips(sys.argv[1])
    hyp_clips = load_clips(sys.argv[2])

    # sort by clip number
    clips = sorted(ref_clips, key=lambda c: int(c.split()[1]))
    assert set(clips) == set(hyp_clips), "Clip mismatch!"

    refs = [ref_clips[c] for c in clips]
    hyps = [hyp_clips[c] for c in clips]

    # --- Corpus-level metrics ---
    # BLEU-4
    bleu_metric = BLEU(effective_order=True)
    bleu_score = bleu_metric.corpus_score(hyps, [refs]).score

    # chrF
    chrf_metric = CHRF()
    chrf_score = chrf_metric.corpus_score(hyps, [refs]).score

    # ROUGE-L
    rouge_metric = ROUGE()
    rouge_score = rouge_metric.corpus_score(hyps, [refs]).score

    # BERTScore-F1
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    P, R, F = bert_scorer.score(hyps, refs)
    bert_score = float(F.mean()) * 100

    # --- Per-clip sentence BLEU & ROUGE-L ---
    table = []
    for clip, r, h in zip(clips, refs, hyps):
        sb = sacrebleu.sentence_bleu(h, [r]).score
        sr = rouge_metric.sentence_score(h, [r]).score  # sentence ROUGE-L
        table.append([clip, f"{sb:5.1f}", f"{sr:5.1f}"])

    # append averages row
    table.append([
        "AVERAGE",
        f"{bleu_score:5.1f}",
        f"{rouge_score:5.1f}"
    ])

    # --- Print ---
    print("\nPer-clip sentence metrics")
    print(tabulate(
        table,
        headers=["Clip", "BLEU", "ROUGE-L"],
        tablefmt="github"
    ))

    print("\nCorpus-level metrics")
    print(f" BLEU-4    = {bleu_score:5.1f}")
    print(f" chrF      = {chrf_score:5.1f}")
    print(f" ROUGE-L   = {rouge_score:5.1f}")
    print(f" BERTScore = {bert_score:5.1f}\n")


if __name__ == "__main__":
    main()
