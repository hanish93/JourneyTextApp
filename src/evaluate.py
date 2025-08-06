import sys
import re
from pathlib import Path

import sacrebleu
from bert_score import BERTScorer
from tabulate import tabulate

def load_clips(path):
    """
    Read a file with lines like:
      Clip 2
      The … text
      Clip 3
      Another text
    Returns: dict { "Clip 2": "The … text", … }
    """
    text = Path(path).read_text().splitlines()
    clips = {}
    key, buf = None, []
    for ln in text:
        ln = ln.strip()
        if not ln:
            continue
        if re.match(r"^Clip\s+\d+", ln):
            if key:
                clips[key] = " ".join(buf).strip()
            key = ln
            buf = []
        elif key:
            buf.append(ln)
    if key:
        clips[key] = " ".join(buf).strip()
    return clips

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 -m src.evaluate Ground_Truth.txt Output.txt")
        sys.exit(1)

    refs = load_clips(sys.argv[1])
    hyps = load_clips(sys.argv[2])
    clips = sorted(refs.keys(), key=lambda x: int(x.split()[1]))
    assert set(clips) == set(hyps.keys()), "Mismatch in clip IDs!"

    # Prepare corpus lists for global scoring
    all_refs = [refs[c] for c in clips]
    all_hyps = [hyps[c] for c in clips]

    # 1) corpus BLEU
    bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs]).score

    # 2) corpus chrF
    chrf = sacrebleu.corpus_chrf(all_hyps, [all_refs]).score

    # 3) corpus ROUGE-L
    rouge = sacrebleu.corpus_rouge_l(all_hyps, [all_refs]).score

    # 4) BERTScore (F1)
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    P, R, F = scorer.score(all_hyps, all_refs)
    bert_f1 = float(F.mean()) * 100

    # Now per-clip BLEU and ROUGE-L (sentence‐level via sacrebleu)
    table = []
    for clip, ref, hyp in zip(clips, all_refs, all_hyps):
        sb = sacrebleu.sentence_bleu(hyp, [ref]).score
        sr = sacrebleu.sentence_rouge_l(hyp, [ref]).score
        table.append([clip, f"{sb:5.1f}", f"{sr:5.1f}"])

    # Add overall row
    table.append([
        "AVERAGE",
        f"{bleu:5.1f}",
        f"{rouge:5.1f}",
    ])

    print("\nPer-clip BLEU & ROUGE-L (sentence level)\n")
    print(tabulate(table, headers=["Clip","BLEU","ROUGE-L"], tablefmt="github"))

    print("\nCorpus-level metrics\n")
    print(f" BLEU-4    = {bleu:5.1f}")
    print(f" chrF      = {chrf:5.1f}")
    print(f" ROUGE-L   = {rouge:5.1f}")
    print(f" BERTScore = {bert_f1:5.1f}")

if __name__ == "__main__":
    main()
