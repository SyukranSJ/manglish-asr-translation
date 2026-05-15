import pandas as pd
from jiwer import wer

# -------------------------------
# Load Data
# -------------------------------
# Main dataset containing ASR predictions and references
df = pd.read_csv("results/mt_results_full.csv")

# Category labels (e.g., English-dominant, Malay-dominant, code-switched)
categories = pd.read_csv("data/categories.csv")

# Merge category information into dataset
df = df.merge(categories, on="id")

# -------------------------------
# Define Models for Evaluation
# -------------------------------
models = {
    "Whisper": "whisper_pred",
    "wav2vec": "wav2vec_pred",
    "wav2vec_ft": "wav2vec_finetuned_pred"
}

results = []

# -------------------------------
# Compute WER by Speech Category
# -------------------------------
# This allows analysis of model performance across different speech types
for category in df["category"].unique():
    subset = df[df["category"] == category]

    # Ground truth references
    refs = subset["reference"].astype(str).tolist()

    row = {"Category": category}

    # Compute WER for each model
    for name, col in models.items():
        preds = subset[col].astype(str).tolist()
        row[name] = wer(refs, preds)

    results.append(row)

# -------------------------------
# Save and Display Results
# -------------------------------
result_df = pd.DataFrame(results)

print("\n=== WER by Category ===")
print(result_df)

# Save results for visualization (e.g., bar chart)
result_df.to_csv("results/wer_by_category.csv", index=False)

print("\nSaved to wer_by_category.csv")
