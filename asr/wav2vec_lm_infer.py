import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
from sacrebleu import corpus_bleu

# -------------------------------
# Setup Base Directory
# -------------------------------
BASE_DIR = Path(__file__).parent.parent

# -------------------------------
# Load NLLB Translation Model
# -------------------------------
# Used for English/Manglish → Malay translation
model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# -------------------------------
# Load Dataset
# -------------------------------
# Contains:
# - reference (source text)
# - reference_ms (ground truth Malay)
df = pd.read_csv(BASE_DIR / "data" / "transcripts_mt.csv")

# -------------------------------
# Load ASR Outputs
# -------------------------------
# Predictions from different ASR systems
whisper = pd.read_csv(BASE_DIR / "results" / "whisper_baseline.csv")
wav2vec = pd.read_csv(BASE_DIR / "results" / "wav2vec_baseline.csv")
wav2vec_ft = pd.read_csv(BASE_DIR / "results" / "wav2vec_finetuned.csv")

# Merge predictions into main dataframe
df["whisper_pred"] = whisper["whisper_pred"]
df["wav2vec_pred"] = wav2vec["wav2vec_pred"]
df["wav2vec_finetuned_pred"] = wav2vec_ft["wav2vec_finetuned_pred"]

# -------------------------------
# Translation Function
# -------------------------------
def translate(text):
    """
    Translates input text into Malay using NLLB.
    """
    inputs = tokenizer(text, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs,
        # Force output language to Malay (zsm_Latn)
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("zsm_Latn")
    )

    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# -------------------------------
# Ground Truth BLEU (Upper Bound)
# -------------------------------
# Measures best-case performance (clean input → translation)
print("\nTranslating Ground Truth (upper bound)...")

gt_translations = []

for text in df["reference"]:
    gt_translations.append(translate(str(text)))

# Reference Malay translations
refs = df["reference_ms"].astype(str).tolist()

# Compute upper-bound BLEU score
gt_bleu = corpus_bleu(gt_translations, [refs])

print(f"Ground Truth BLEU: {gt_bleu.score:.2f}")

# -------------------------------
# BLEU Evaluation per Model
# -------------------------------
models = [
    ("whisper_pred", "Whisper"),
    ("wav2vec_pred", "wav2vec"),
    ("wav2vec_finetuned_pred", "wav2vec_finetuned")
]

bleu_results = [("Ground Truth", gt_bleu.score)]

for col, name in models:
    print(f"\nTranslating {name}...")

    translations = []

    # Translate ASR outputs
    for text in df[col]:
        translations.append(translate(str(text)))

    # Store translated outputs
    df[f"{name}_mt"] = translations

    # Compute BLEU score
    bleu = corpus_bleu(translations, [refs])

    print(f"{name} BLEU: {bleu.score:.2f}")

    bleu_results.append((name, bleu.score))

# -------------------------------
# Save Results
# -------------------------------
output_path = BASE_DIR / "results" / "mt_results_full.csv"
df.to_csv(output_path, index=False)

print("\nMT completed. Saved to:", output_path)

# -------------------------------
# Final BLEU Summary
# -------------------------------
print("\n=== BLEU Summary ===")
for name, score in bleu_results:
    print(f"{name}: {score:.2f}")