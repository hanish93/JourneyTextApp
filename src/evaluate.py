import evaluate
import pandas as pd

# -----------------------------
# Load evaluation metrics
# -----------------------------
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
chrf = evaluate.load("chrf")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# -----------------------------
# Load your files
# -----------------------------
with open("ground_truth.txt", "r", encoding="utf-8") as f:
    references = [line.strip() for line in f.readlines()]

with open("output.txt", "r", encoding="utf-8") as f:
    outputs = [line.strip() for line in f.readlines()]

assert len(outputs) == len(references), "Mismatch in number of clips!"

# -----------------------------
# Evaluate each clip
# -----------------------------
results = []
for i, (pred, ref) in enumerate(zip(outputs, references), start=2):  # clips 2 → 14
    clip_name = f"clip_{i}"
    
    # Compute metrics
    bleu_score = bleu.compute(predictions=[pred], references=[[ref]])["bleu"] * 100
    meteor_score = meteor.compute(predictions=[pred], references=[ref])["meteor"] * 100
    chrf_score = chrf.compute(predictions=[pred], references=[ref])["score"]
    rouge_score = rouge.compute(predictions=[pred], references=[ref])
    bert_score = bertscore.compute(predictions=[pred], references=[ref], model_type="bert-base-uncased")

    results.append({
        "Clip": clip_name,
        "BLEU": round(bleu_score, 2),
        "METEOR": round(meteor_score, 2),
        "chrF": round(chrf_score, 2),
        "ROUGE-1": round(rouge_score["rouge1"] * 100, 2),
        "ROUGE-L": round(rouge_score["rougeL"] * 100, 2),
        "BERT-F1": round(sum(bert_score["f1"]) / len(bert_score["f1"]) * 100, 2)
    })

# -----------------------------
# Convert to DataFrame
# -----------------------------
df = pd.DataFrame(results)

# Compute averages
df.loc["AVERAGE"] = df.mean(numeric_only=True)
df.loc["AVERAGE", "Clip"] = "AVERAGE"

# Save / print table
print(df.to_string(index=False))

# Save to CSV for later use
df.to_csv("evaluation_results.csv", index=False)
