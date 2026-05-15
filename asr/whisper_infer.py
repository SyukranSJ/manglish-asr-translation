import whisper
import pandas as pd
from pathlib import Path

# -------------------------------
# Load Whisper ASR Model
# -------------------------------
print("Loading Whisper model...")
model = whisper.load_model("small")  # using 'small' variant for balance of speed and accuracy

# -------------------------------
# Set Base Directory and Load Data
# -------------------------------
# Assumes script is inside a subfolder (e.g., src/)
BASE_DIR = Path(__file__).parent.parent

# Load dataset containing audio file paths
df = pd.read_csv(BASE_DIR / "data" / "transcripts.csv")

# Store predicted transcriptions
predictions = []

# -------------------------------
# Run Inference on Each Audio File
# -------------------------------
for _, row in df.iterrows():
    audio_path = BASE_DIR / row["audio_path"]
    print(f"Processing: {audio_path}")

    try:
        # Perform speech-to-text transcription
        result = model.transcribe(
            str(audio_path),
            task="transcribe",   # transcription (not translation)
            language="en"        # assume English / Manglish input
        )

        # Clean and normalize output text
        text = result["text"].strip().lower()

    except Exception as e:
        # Handle errors (e.g., missing/corrupted audio files)
        print(f"Error processing {audio_path}: {e}")
        text = ""

    predictions.append(text)

# -------------------------------
# Save Results
# -------------------------------
# Add predictions as new column
df["whisper_pred"] = predictions

# Save output to results folder
df.to_csv(BASE_DIR / "results" / "whisper_baseline.csv", index=False)

print("Whisper inference completed.")