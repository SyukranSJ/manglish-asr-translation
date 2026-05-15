import torch
import torchaudio
import pandas as pd
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pathlib import Path

# -------------------------------
# Load Pretrained Wav2Vec2 Model
# -------------------------------
print("Loading wav2vec 2.0 model...")

# Using pretrained English ASR model (baseline)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()  # set model to evaluation mode

# -------------------------------
# Load Dataset
# -------------------------------
BASE_DIR = Path(__file__).parent.parent

# CSV contains paths to audio files
df = pd.read_csv(BASE_DIR / "data" / "transcripts.csv")

# Store predicted transcriptions
predictions = []

# -------------------------------
# Run Inference on Each Audio File
# -------------------------------
for _, row in df.iterrows():
    audio_path = BASE_DIR / row["audio_path"]

    # Load audio using soundfile
    waveform, sr = sf.read(audio_path)
    waveform = torch.tensor(waveform, dtype=torch.float32)

    # Convert stereo to mono if necessary
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    # Resample audio to 16kHz (required by Wav2Vec2)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # Preprocess audio for model input
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    # Perform inference without gradient computation
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # Decode predicted token IDs into text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0]).lower().strip()

    predictions.append(transcription)

# -------------------------------
# Save Results
# -------------------------------
# Add predictions to dataframe
df["wav2vec_pred"] = predictions

# Save output CSV file
df.to_csv(BASE_DIR / "results" / "wav2vec_baseline.csv", index=False)

print("wav2vec 2.0 inference completed.")