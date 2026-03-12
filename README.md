# 24AI636 DL — Scaffolded Project
## MINI-PROJECT 2: Pretrained CNN + Temporal Modeling using LSTM
**Review 2 — 5th Mar 2026**

---

## Overview

This project combines Pretrained CNN models with an LSTM-based temporal model to classify sequences of MNIST digit images. Each sample is a sequence of 4 frames of the same digit, and the model predicts which digit it is.

---

## Project Structure

```
mini_project2_cnn_temporal.py   ← Main code file
sample_sequences.png            ← Sample MNIST sequences (generated)
training_curves.png             ← Accuracy and loss curves (generated)
confusion_matrix.png            ← Final test confusion matrix (generated)
data/                           ← MNIST dataset (auto downloaded)
```

---

## Rubric Coverage (20 Marks)

| Criterion | Marks | What Was Done |
|---|---|---|
| Temporal Data Preprocessing Pipeline | 2 | MNIST images grouped into sequences of 4 frames per digit |
| Feature Extraction using Pretrained CNNs | 3 | ResNet-18 and MobileNet-V2 used as frozen feature extractors |
| Fine-Tuning Pretrained CNN | 3 | ResNet-18 with layer4 unfrozen for transfer learning |
| Embedding Usage | 2 | CNN features projected + positional embeddings added |
| Attention-Based Model | 2 | Scaled dot-product attention over LSTM output frames |
| LSTM Implementation | 4 | Full pipeline: Frames → CNN → Embedding → LSTM → Attention → Label |
| Hyperparameter Experimentation | 1 | Grid search over learning rate and hidden size |
| Model Comparison & Evaluation Metrics | 2 | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Code Organisation | 1 | Modular sections, fixed seed, plots saved |

---

## Model Pipeline

```
Input Frames (4 MNIST images)
        ↓
CNN Feature Extractor (ResNet-18)
        ↓
Temporal Embedding (feature compression + position info)
        ↓
LSTM (captures dependencies across frames)
        ↓
Attention (focuses on important frames)
        ↓
Classifier → Digit Prediction (0–9)
```

---

## How to Run

**Step 1 — Install dependencies**
```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

**Step 2 — Run the project**
```bash
python mini_project2_cnn_temporal.py
```

MNIST dataset will be downloaded automatically on first run.

---

## Settings

| Parameter | Value |
|---|---|
| Sequence Length | 4 frames |
| Batch Size | 32 |
| Epochs | 5 |
| Embed Dim | 256 |
| Hidden Dim | 256 |
| LSTM Layers | 2 |
| Dropout | 0.3 |
| Learning Rate | 0.001 |
| Number of Classes | 10 (digits 0–9) |

---

## Hyperparameter Search

Tested the following combinations and picked the best by validation accuracy:

| Learning Rate | Hidden Dim |
|---|---|
| 0.001 | 128 |
| 0.001 | 256 |
| 0.0005 | 128 |
| 0.0005 | 256 |

---

## Outputs

- **sample_sequences.png** — Shows sample MNIST sequences used for training
- **training_curves.png** — Train vs Val accuracy and loss per epoch
- **confusion_matrix.png** — Predicted vs True labels on test set

---

## Why LSTM?

- Handles **long-term dependencies** across frames better than plain RNN
- Has **forget and input gates** giving better control over memory
- Most widely used sequence model for temporal classification tasks

---

## Requirements

```
torch
torchvision
scikit-learn
matplotlib
seaborn
```

---

## Author

**Course:** 24AI636 Deep Learning
**Project:** Mini-Project 2
**Review:** 2 — 5th March 2026
