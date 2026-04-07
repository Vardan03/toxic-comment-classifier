<div align="center">

# Toxic Comment Classifier

<p align="center">
  <b>Production-level multi-label toxic comment detection system</b><br/>
  Built with classical ML, deep learning and pretrained transformers
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/NLP-Multi--Label-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Models-TFIDF%20%7C%20RNN%20%7C%20LSTM%20%7C%20BERT%20%7C%20GPT-purple?style=for-the-badge"/>
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

- RNN (Didn't work well)
- LSTM

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
├── data/
├── reports/
├── saved_models/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   │   ├── baseline/
│   │   ├── deep_learning/
│   │   ├── pretrained/
│   │   └── base_model.py
│   ├── utils/
│   ├── config.py
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
Step 1 — Data Preprocessing (done once):
  bashpython main.py --mode preprocess
Step 2 — Model Evaluation:
  All python models at once 
    main.py --evaluate mode
  The specific model
    python main.py --mode evaluate --models tfidf
    python main.py --mode evaluate --models lstm bert
  With a custom threshold
    python main.py --mode evaluate --models tfidf --threshold 0.4


An important point is that models must be trained before evaluating, otherwise the script will issue a warning:
⚠️ lstm model not found - Train it first with:
   python train.py --model lstm

read, first you need to run:
bashpython train.py --model lstm # or other model

The results after evaluation are saved in reports/result.txt .
```

## Example
```bash
text = "You are disgusting and horrible"
prediction = model.predict([text])
print(prediction)
```

## Results
```
=== tfidf_t0.3 | 2026-04-07 16:26:52 ===
Label                   ROC AUC         F1     Recall
=======================================================
toxic                    0.9494     0.4911     0.9392
severe_toxic             0.9801     0.1879     0.9210
obscene                  0.9632     0.4855     0.9119
threat                   0.9866     0.2385     0.8626
insult                   0.9595     0.4379     0.9151
identity_hate            0.9553     0.2282     0.8301
=======================================================
MACRO AVG                0.9657     0.3448     0.8966


=== rnn_t0.3 | 2026-04-07 16:27:44 ===
Label                   ROC AUC         F1     Recall
=======================================================
toxic                    0.5149     0.1738     1.0000
severe_toxic             0.5485     0.0114     1.0000
obscene                  0.5170     0.1091     1.0000
threat                   0.5253     0.0064     0.8910
insult                   0.5292     0.1017     1.0000
identity_hate            0.5162     0.0220     1.0000
=======================================================
MACRO AVG                0.5252     0.0707     0.9818


=== lstm_t0.3 | 2026-04-07 16:29:04 ===
Label                   ROC AUC         F1     Recall
=======================================================
toxic                    0.9613     0.5041     0.9631
severe_toxic             0.9842     0.1679     0.9319
obscene                  0.9725     0.4265     0.9615
threat                   0.9723     0.2009     0.8199
insult                   0.9644     0.4031     0.9448
identity_hate            0.9531     0.1857     0.8090
=======================================================
MACRO AVG                0.9680     0.3147     0.9050


=== gpt_t0.3 | 2026-04-07 16:50:54 ===
Label                   ROC AUC         F1     Recall
=======================================================
toxic                    0.9118     0.4051     0.9110
severe_toxic             0.9217     0.0865     0.8202
obscene                  0.9210     0.3477     0.8781
threat                   0.9135     0.0699     0.7299
insult                   0.9168     0.3249     0.8748
identity_hate            0.9159     0.1333     0.8230
=======================================================
MACRO AVG                0.9168     0.2279     0.8395


=== bert_t0.5 | 2026-04-07 17:56:49 ===
Label                   ROC AUC         F1     Recall
=======================================================
toxic                    0.9585     0.5474     0.9438
severe_toxic             0.9875     0.1471     0.9728
obscene                  0.9709     0.4971     0.9371
threat                   0.9861     0.1921     0.8341
insult                   0.9650     0.4503     0.9297
identity_hate            0.9724     0.2229     0.8792
=======================================================
MACRO AVG                0.9734     0.3428     0.9161

```


## Workflow
EDA → Preprocessing → Feature Extraction → Models → Evaluation → Prediction

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
Vardan Hovhannisyan,
Lina Baghunts,
Davit Tshaghryan,
Vahe Hovhannisyan


<div align="center">
⭐ Star the repo if you like it
</div> 
