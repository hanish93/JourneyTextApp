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

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("omw-1.4", quiet=True)


def read_clips(path: Path):
    """
    Reads a file with blocks:
      Clip N
      <some text>

      Clip M
      <some text>
    Returns dict[clip_name] = text
    """
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^(Clip\s+\d+)\s*$", text, flags=re.MULTILINE)
    # parts = ["", "Clip 2", "ref text", "Clip 3", "ref text", ...]
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

    assert set(refs) == set(hyps), "Mismatch in clip names between reference and output!"

    # initialize scorers
    bleu_scorer = sacrebleu.metrics.BLEU()
    chrf_scorer = sacrebleu.metrics.CHRF()
    rouge_s = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    table = []
    for clip in sorted(refs.keys(), key=lambda c: int(re.search(r"\d+", c).group())):
        ref = refs[clip]
        hyp = hyps[clip]

        # BLEU
        bleu = bleu_scorer.corpus_score([hyp], [[ref]]).score

        # chrF
        chrf = chrf_scorer.corpus_score([hyp], [[ref]]).score

        # ROUGE-1 & ROUGE-L F1
        scores = rouge_s.score(ref, hyp)
        r1_f = scores["rouge1"].fmeasure * 100
        rl_f = scores["rougeL"].fmeasure * 100

        # METEOR
        meteor = single_meteor_score(ref, hyp) * 100

        # BERTScore F1
        P, R, F1 = bert_score([hyp], [ref], lang="en", rescale_with_baseline=True)
        bert_f = F1[0].item() * 100

        table.append([
            clip,
            f"{bleu:5.1f}",
            f"{meteor:5.1f}",
            f"{chrf:5.1f}",
            f"{r1_f:6.1f}",
            f"{rl_f:6.1f}",
            f"{bert_f:5.1f}"
        ])

    # average row
    cols = list(zip(*table))
    avg = ["AVERAGE"] + [
        f"{sum(float(x) for x in col)/len(col):5.1f}"
        for col in cols[1:]
    ]
    table.append(avg)

    print(tabulate(
        table,
        headers=["Clip", "BLEU", "METEOR", "chrF", "ROUGE-1", "ROUGE-L", "BERT-F1"],
        tablefmt="github"
    ))


if __name__ == "__main__":
    main()
