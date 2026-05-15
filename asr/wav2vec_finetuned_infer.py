import torch
import pandas as pd
from pathlib import Path
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# -------------------------------
# Load Fine-Tuned Wav2Vec2 Model
# -------------------------------
print("Loading fine-tuned model...")

BASE_DIR = Path(__file__).parent.parent

# Load processor and model from local fine-tuned directory
processor = Wav2Vec2Processor.from_pretrained(
    BASE_DIR / "models" / "wav2vec_manglish_ft_v1"
)
model = Wav2Vec2ForCTC.from_pretrained(
    BASE_DIR / "models" / "wav2vec_manglish_ft_v1"
)

# -------------------------------
# Load Dataset
# -------------------------------
# CSV contains paths to audio files
df = pd.read_csv(BASE_DIR / "data" / "transcripts.csv")

# Store predicted transcriptions
predictions = []

# -------------------------------
# Run Inference on Each Audio File
# -------------------------------
for _, row in df.iterrows():
    audio_path = BASE_DIR / row["audio_path"]

    # Load audio file
    speech, sr = sf.read(audio_path)

    # Convert stereo to mono if needed
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)

    # Ensure correct data type for model input
    speech = speech.astype("float32")

    # Preprocess audio for Wav2Vec2 model
    inputs = processor(
        speech,
        sampling_rate=16000,   # model expects 16kHz audio
        return_tensors="pt"
    )

    # Perform inference (no gradient computation)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Get predicted token IDs and decode to text
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0].lower().strip()

    predictions.append(text)

# -------------------------------
# Save Results
# -------------------------------
# Add predictions to dataframe
df["wav2vec_finetuned_pred"] = predictions

# Save output CSV file
df.to_csv(BASE_DIR / "results" / "wav2vec_finetuned.csv", index=False)

print("Fine-tuned inference completed.")