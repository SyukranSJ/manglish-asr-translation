import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
import soundfile as sf
from pathlib import Path
import re

# -------------------------------
# Load Processor and Base Model
# -------------------------------
print("Loading processor and model...")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# -------------------------------
# Fine-Tuning Strategy
# -------------------------------
# Freeze most of the pretrained model to retain learned speech representations
for param in model.wav2vec2.parameters():
    param.requires_grad = False

# Unfreeze last encoder layers for domain adaptation (Manglish speech)
for param in model.wav2vec2.encoder.layers[-2:].parameters():
    param.requires_grad = True

# -------------------------------
# Load Dataset
# -------------------------------
BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / "data" / "transcripts.csv")

# -------------------------------
# Text Normalization
# -------------------------------
# Clean transcripts for stable training
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------------
# Audio + Label Preparation
# -------------------------------
def load_audio(example):
    audio_path = BASE_DIR / example["audio_path"]
    speech, sr = sf.read(audio_path)

    # Convert stereo to mono
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)

    # Ensure correct dtype
    speech = speech.astype("float32")

    # Clean target text
    text = clean_text(example["reference"])

    # Extract input features
    inputs = processor(
        speech,
        sampling_rate=16000
    )
    example["input_values"] = inputs.input_values[0]

    # Tokenize labels
    labels = processor.tokenizer(text).input_ids

    # Replace padding tokens with -100 (ignored in loss)
    labels = [
        l if l != processor.tokenizer.pad_token_id else -100
        for l in labels
    ]

    example["labels"] = labels

    return example

# Convert pandas DataFrame to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Apply preprocessing
dataset = dataset.map(
    load_audio,
    remove_columns=dataset.column_names
)

# -------------------------------
# Training Configuration
# -------------------------------
training_args = TrainingArguments(
    output_dir=str(BASE_DIR / "models" / "wav2vec_manglish_ft_v1"),
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=50,
    warmup_steps=10,
    save_total_limit=1,
    fp16=False  # set True if using GPU with mixed precision
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# -------------------------------
# Training
# -------------------------------
print("Starting training...")
trainer.train()

# -------------------------------
# Save Fine-Tuned Model
# -------------------------------
model.save_pretrained(BASE_DIR / "models" / "wav2vec_manglish_ft_v1")
processor.save_pretrained(BASE_DIR / "models" / "wav2vec_manglish_ft_v1")

print("Fine-tuning completed.")