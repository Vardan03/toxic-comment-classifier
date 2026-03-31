<div align="center">

# Toxic Comment Classifier

<p align="center">
  <b>Production-level multi-label toxic comment detection system</b><br/>
  Built with classical ML, deep learning, transformer, and ensemble approaches
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/NLP-Multi--Label-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Models-TFIDF%20%7C%20RNN%20%7C%20LSTM%20%7C%20BERT%20%7C%20Ensemble-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Advanced-brightgreen?style=for-the-badge"/>
</p>

</div>

---

## Overview

This project is a **multi-label toxic comment classification system** designed to detect harmful language in online text using a wide range of NLP and machine learning techniques.

Each comment can belong to multiple categories:
toxic, severe_toxic, obscene, threat, insult, identity_hate.

The project follows a full ML pipeline:
EDA → preprocessing → feature engineering → training → evaluation → inference.

---

## Why This Project Stands Out

- Multiple model families (not just one)
- Real ML project structure (not just notebooks)
- Full pipeline implemented
- Multi-label classification done properly
- Designed for experimentation and comparison

---

## Models Used

### Baseline (TF-IDF)
Located in `src/models/baseline/`

- TF-IDF + classical ML

Why:
- Fast
- Interpretable
- Strong baseline

---

### Deep Learning
Located in `src/models/deep_learning/`

- RNN
- LSTM
- NN models

Why:
- Capture sequence and context
- Learn semantic relationships

---

### Pretrained Models
Located in `src/models/pretrained/`

- BERT
- GPT

Why:
- Context-aware embeddings
- Strong NLP performance

---

### Ensemble
Located in `src/models/ensemble_model.py`

- Combines multiple models

Why:
- More robust
- Better performance

---

### Multi-Label Strategy

- Independent label prediction
- Threshold-based outputs
- Supports multiple labels per comment

---

## Architecture

Raw Text → Preprocessing → Features → Models → Evaluation → Saved Models → Prediction

---

## Project Structure

```bash
toxic-comment-classifier/
├── notebooks/
├── saved_models/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   │   ├── baseline/
│   │   ├── deep_learning/
│   │   ├── pretrained/
│   │   ├── ensemble_model.py
│   │   └── base_model.py
│   ├── utils/
│   ├── config.py
│   └── argparser.py
├── main.py
├── train.py
├── predict.py
├── requirements.txt
└── pyproject.toml
```

## Setup
```bash
git clone https://github.com/Vardan03/toxic-comment-classifier.git
cd toxic-comment-classifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python train.py
python predict.py
python main.py
```

## Example
```bash
text = "You are disgusting and horrible"
prediction = model.predict([text])
print(prediction)
```

## Workflow
EDA → Preprocessing → Feature Extraction → Models → Evaluation → Ensemble → Prediction

## Key Design Decisions
Modular architecture
Multiple model comparison
Saved artifacts
Separate training and inference
Real-world ML pipeline

## Future Improvements
Hyperparameter tuning
Better thresholds
Stronger transformer training
Deployment (API / UI)
Explainability tools

## Contributors
Vardan
Lina
Vahe
Davit


<div align="center">
⭐ Star the repo if you like it
</div> ```
