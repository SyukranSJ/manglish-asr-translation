# Manglish ASR + Translation System

A speech-to-text translation pipeline for English–Malay with informal and code-switched speech (Manglish). Built as a Final Year Project at the University of Nottingham Malaysia.

**Report:** [report.pdf](./report.pdf) — Full dissertation  

## Overview

This project evaluates four ASR configurations on a manually constructed Manglish dataset and pipes the transcriptions through a machine translation model to produce Malay output.

**Pipeline:** Audio → ASR → English/Manglish transcript → MT → Malay translation

### Models Used
| Component | Model |
|-----------|-------|
| ASR (baseline) | [openai/whisper-small](https://huggingface.co/openai/whisper-small) |
| ASR (baseline) | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) |
| ASR (fine-tuned) | wav2vec2-base-960h fine-tuned on Manglish dataset |
| ASR (LM) | wav2vec2 + KenLM n-gram language model |
| MT | [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) |

### Key Results
| Model | WER ↓ | BLEU ↑ |
|-------|--------|--------|
| Whisper (small) | 0.415 | 9.64 |
| wav2vec 2.0 baseline | 0.724 | 2.35 |
| wav2vec fine-tuned | 0.700 | 2.28 |
| wav2vec + KenLM | 0.916 | 2.05 |
| Upper bound (ground truth → MT) | — | 26.78 |

## Project Structure

```
fyp_asr/
├── main.py                        # Gradio web app
├── requirements.txt
├── asr/
│   ├── whisper_infer.py           # Whisper inference
│   ├── wav2vec_infer.py           # wav2vec baseline inference
│   ├── wav2vec_finetune.py        # wav2vec fine-tuning script
│   ├── wav2vec_finetuned_infer.py # Fine-tuned wav2vec inference
│   ├── wav2vec_lm_infer.py        # wav2vec + KenLM inference
│   ├── mt_infer.py                # MT evaluation (BLEU)
│   ├── wer_eval.py                # WER evaluation
│   └── compute_category_wer.py    # WER breakdown by speech category
├── data/                          # Not included — private voice recordings
│   ├── audio/                     # 61 WAV files (16 kHz)
│   ├── transcripts.csv            # Audio paths + reference transcriptions
│   ├── transcripts_mt.csv         # Subset for MT evaluation
│   ├── categories.csv             # Category labels per utterance
│   ├── lm_corpus.txt              # KenLM training corpus
│   ├── lm.arpa                    # ARPA language model
│   └── lm.binary                  # Compiled KenLM binary
├── results/
│   ├── wer_summary.csv
│   ├── wer_by_category.csv
│   ├── bleu_summary.csv
│   ├── whisper_baseline.csv
│   ├── wav2vec_baseline.csv
│   ├── wav2vec_finetuned.csv
│   ├── wav2vec_lm.csv
│   └── mt_results_full.csv
└── models/
    └── wav2vec_manglish_ft_v1/    # Fine-tuned model (download separately)
```

## Dataset

61 manually constructed Manglish utterances across four categories:

| Category | Description | Example |
|----------|-------------|---------|
| Pure informal English | Conversational English | *"you do finish already or still doing"* |
| Manglish + discourse particles | English with Malay particles | *"i think lah this one better we do tomorrow"* |
| Code-switched English–Malay | Interleaved languages | *"this part susah sikit i need more time"* |
| Predominantly Malay | Malay-dominant utterances | *"tadi lecturer explain laju sangat tak sempat nak faham"* |

Each sample has: a WAV audio file, a reference transcription, and a Malay reference translation.

> **Note:** The `data/` folder is not included in this repository as it contains private voice recordings. To reproduce the experiments, you will need to supply your own dataset in the same format — see `data/transcripts.csv` structure below.

**Expected `transcripts.csv` format:**
```
id,audio_path,reference
1,data/audio/test1.wav,you do finish already or still doing
2,data/audio/test2.wav,i think lah this one better we do tomorrow
...
```

> The same 61 samples were used for both fine-tuning and evaluation due to dataset size constraints. Results reflect domain adaptation behaviour, not generalisable performance.

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fyp_asr.git
cd fyp_asr
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install KenLM (required for wav2vec + LM only)
```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### 5. Download the fine-tuned model

The fine-tuned wav2vec model is hosted on Hugging Face. Download and place it at `models/wav2vec_manglish_ft_v1/`:

```bash
# Coming soon — Hugging Face link
```

Alternatively, you can fine-tune the model yourself:
```bash
python asr/wav2vec_finetune.py
```

## Running the Web App

```bash
python main.py
```

Then open `http://localhost:7860` in your browser. Upload a WAV file, select an ASR model, and click **Run** to get the transcription and Malay translation.

## Running Experiments

Run scripts in this order:

```bash
# 1. ASR inference
python asr/whisper_infer.py
python asr/wav2vec_infer.py
python asr/wav2vec_finetuned_infer.py
python asr/wav2vec_lm_infer.py

# 2. Evaluate ASR (WER)
python asr/wer_eval.py
python asr/compute_category_wer.py

# 3. Machine translation + BLEU
python asr/mt_infer.py
```

Results are saved to the `results/` folder.

## Requirements

- Python 3.9+
- ~8 GB disk space for models (downloaded automatically from Hugging Face on first run, except the fine-tuned model)
- GPU optional but recommended for MT inference

## Author

**Muhammad Syukran Shabaruddin**
BSc Computer Science + Artificial Intelligence (Hons)
University of Nottingham Malaysia
Supervisor: Dr. Kweh Yeah Lun
