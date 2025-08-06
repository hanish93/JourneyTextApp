# src/evaluate.py
import sys
import re
from pathlib import Path

import nltk
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import single_meteor_score
from tabulate import tabulate

# download once
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("omw-1.4", quiet=True)


def read_clips(path: Path):
    """
    Read blocks of the form:
      Clip N
      <some text>
    Returns dict mapping "Clip N" → text.
    """
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^(Clip\s+\d+)\s*$", text, flags=re.MULTILINE)
    clips = {}
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        val  = parts[i+1].strip().replace("\n", " ")
        clips[name] = val
    return clips


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.evaluate Ground_Truth.txt Output.txt")
        sys.exit(1)

    ref_file, out_file = Path(sys.argv[1]), Path(sys.argv[2])
    refs = read_clips(ref_file)
    hyps = read_clips(out_file)

    if set(refs) != set(hyps):
        missing = set(refs) ^ set(hyps)
        print("Mismatch in clips:", missing)
        sys.exit(1)

    # scorers
    bleu_scorer = sacrebleu.metrics.BLEU()
    chrf_scorer = sacrebleu.metrics.CHRF()
    rouge_s = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rows = []
    # sort by clip number
    for clip in sorted(refs.keys(), key=lambda c: int(re.search(r"\d+", c).group())):
        ref = refs[clip]
        hyp = hyps[clip]

        # BLEU
        bleu = bleu_scorer.corpus_score([hyp], [[ref]]).score

        # chrF
        chrf = chrf_scorer.corpus_score([hyp], [[ref]]).score

        # ROUGE
        sc = rouge_s.score(ref, hyp)
        r1 = sc["rouge1"].fmeasure * 100
        rl = sc["rougeL"].fmeasure * 100

        # METEOR (requires token lists)
        meteor = single_meteor_score(ref.split(), hyp.split()) * 100

        # BERTScore
        P, R, F1 = bert_score([hyp], [ref], lang="en", rescale_with_baseline=True)
        bert_f = F1[0].item() * 100

        rows.append([
            clip,
            f"{bleu:5.1f}",
            f"{meteor:5.1f}",
            f"{chrf:5.1f}",
            f"{r1:6.1f}",
            f"{rl:6.1f}",
            f"{bert_f:5.1f}"
        ])

    # compute averages
    cols = list(zip(*rows))
    avg = ["AVERAGE"] + [
        f"{sum(float(x) for x in col)/len(col):5.1f}"
        for col in cols[1:]
    ]
    rows.append(avg)

    print(tabulate(
        rows,
        headers=["Clip", "BLEU", "METEOR", "chrF", "ROUGE-1", "ROUGE-L", "BERT-F1"],
        tablefmt="github"
    ))


if __name__ == "__main__":
    main()
