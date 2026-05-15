import pandas as pd
from jiwer import wer
from pathlib import Path
import re

# -------------------------------
# Setup Paths
# -------------------------------
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

# -------------------------------
# Text Normalization Function
# -------------------------------
# Ensures fair WER comparison by removing casing, punctuation, and extra spaces
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # normalize spacing
    return text

# -------------------------------
# Compute WER for a Given Model
# -------------------------------
def compute_wer(csv_file, pred_column, model_name):
    df = pd.read_csv(csv_file)

    # Normalize reference and predicted text
    references = [normalize(t) for t in df["reference"].astype(str)]
    predictions = [normalize(t) for t in df[pred_column].astype(str)]

    # Calculate Word Error Rate
    score = wer(references, predictions)

    print(f"{model_name} WER: {score:.3f}")
    return score

# -------------------------------
# Run WER Evaluation
# -------------------------------
print("=== WER Evaluation ===")

# Whisper baseline
whisper_wer = compute_wer(
    RESULTS_DIR / "whisper_baseline.csv",
    "whisper_pred",
    "Whisper"
)

# wav2vec baseline
wav2vec_wer = compute_wer(
    RESULTS_DIR / "wav2vec_baseline.csv",
    "wav2vec_pred",
    "wav2vec 2.0"
)

# wav2vec fine-tuned
wav2vec_finetuned_wer = compute_wer(
    RESULTS_DIR / "wav2vec_finetuned.csv",
    "wav2vec_finetuned_pred",
    "wav2vec (fine-tuned)"
)

# wav2vec with Language Model (optional)
try:
    wav2vec_lm_wer = compute_wer(
        RESULTS_DIR / "wav2vec_lm.csv",
        "wav2vec_lm_pred",
        "wav2vec + LM"
    )
except Exception as e:
    print("LM results not found or error:", e)
    wav2vec_lm_wer = None

# -------------------------------
# Save Summary Results
# -------------------------------
models = [
    "Whisper",
    "wav2vec 2.0",
    "wav2vec (fine-tuned)"
]

wers = [
    whisper_wer,
    wav2vec_wer,
    wav2vec_finetuned_wer
]

# Include LM results if available
if wav2vec_lm_wer is not None:
    models.append("wav2vec + LM")
    wers.append(wav2vec_lm_wer)

# Create summary dataframe
summary_df = pd.DataFrame({
    "Model": models,
    "WER": wers
})

# Save summary CSV
summary_path = RESULTS_DIR / "wer_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\nSummary saved to:", summary_path)
print(summary_df)