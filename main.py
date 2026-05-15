import gradio as gr
import whisper
import torch
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)

# -------------------------------
# Setup Paths
# -------------------------------
BASE_DIR = Path(__file__).parent

# -------------------------------
# Load Models
# -------------------------------
print("Loading models...")

# Whisper ASR (FORCED ENGLISH)
whisper_model = whisper.load_model("small")

# wav2vec fine-tuned ASR
wav2vec_processor = Wav2Vec2Processor.from_pretrained(
    BASE_DIR / "models" / "wav2vec_manglish_ft_v1"
)
wav2vec_model = Wav2Vec2ForCTC.from_pretrained(
    BASE_DIR / "models" / "wav2vec_manglish_ft_v1"
)

# NLLB Translation Model (English → Malay)
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
mt_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Models loaded.")

# -------------------------------
# wav2vec Transcription Function
# -------------------------------
def wav2vec_transcribe(audio_path):
    speech, sr = sf.read(audio_path)

    # Stereo → Mono
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)

    speech = speech.astype("float32")

    inputs = wav2vec_processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = wav2vec_model(inputs.input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    text = wav2vec_processor.batch_decode(pred_ids)[0].lower().strip()

    return text

# -------------------------------
# Main Pipeline (ASR + MT)
# -------------------------------
def transcribe_and_translate(audio, model_choice):
    if audio is None:
        return "No audio uploaded", ""

    # ---- ASR ----
    if model_choice == "Whisper":
        # 🔥 FORCE ENGLISH (IMPORTANT FIX)
        result = whisper_model.transcribe(audio, language="en")
        transcription = result["text"]
    else:
        transcription = wav2vec_transcribe(audio)

    # ---- MT (Malay Translation) ----
    inputs = tokenizer(transcription, return_tensors="pt")

    translated_tokens = mt_model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("zsm_Latn")
    )

    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return transcription, translation

# -------------------------------
# Load Evaluation Tables
# -------------------------------
def load_wer_category():
    return pd.read_csv(BASE_DIR / "results" / "wer_by_category.csv")

def load_wer_summary():
    return pd.read_csv(BASE_DIR / "results" / "wer_summary.csv")

def load_bleu():
    return pd.read_csv(BASE_DIR / "results" / "bleu_summary.csv")

# -------------------------------
# WER Plot
# -------------------------------
def plot_wer():
    df = pd.read_csv(BASE_DIR / "results" / "wer_by_category.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    df.set_index("Category").plot(
        kind="bar",
        ax=ax,
        rot=0
    )

    ax.set_title("WER by Category")
    ax.set_ylabel("WER")
    ax.set_xlabel("")

    plt.style.use("ggplot")
    plt.tight_layout()

    return fig

# -------------------------------
# Gradio UI
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Manglish ASR + Translation System")
    gr.Markdown("Upload audio and select model to transcribe and translate.")

    audio_input = gr.Audio(type="filepath", label="Upload Audio")

    model_choice = gr.Dropdown(
        ["Whisper", "wav2vec (fine-tuned)"],
        value="Whisper",
        label="ASR Model"
    )

    with gr.Row():
        transcription_output = gr.Textbox(label="Transcription")
        translation_output = gr.Textbox(label="Translation (Malay)")

    run_btn = gr.Button("Run")

    run_btn.click(
        transcribe_and_translate,
        inputs=[audio_input, model_choice],
        outputs=[transcription_output, translation_output]
    )

    # -------------------------------
    # Evaluation Results
    # -------------------------------
    gr.Markdown("## Evaluation Results")

    with gr.Row():
        wer_cat = gr.Dataframe(load_wer_category(), label="WER by Category")
        wer_sum = gr.Dataframe(load_wer_summary(), label="WER Summary")

    with gr.Row():
        bleu_table = gr.Dataframe(load_bleu(), label="BLEU Summary")

    gr.Markdown("## WER Visualization")
    gr.Plot(plot_wer)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    demo.launch()