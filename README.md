

---

## Overview

This project uses pretrained CNN models combined with sequence models (RNN, LSTM, GRU) to classify MNIST digit images treated as temporal sequences. Each sample consists of 4 frames of the same digit grouped into one sequence. All three sequence models are trained and compared against each other.

---

## Project Structure

```
mini_project2_cnn_temporal.py   ← Main code file
sample_sequences.png            ← Sample MNIST sequences (auto generated)
training_curves.png             ← Val accuracy and loss for RNN vs LSTM vs GRU
confusion_matrices.png          ← Confusion matrix for all 3 models
model_accuracy_bar.png          ← Bar chart comparing test accuracy
data/                           ← MNIST dataset (auto downloaded)
```

---

## Rubric Coverage (20 Marks)

| # | Criterion | Marks | What Was Done |
|---|---|---|---|
| 1 | Temporal Data Preprocessing Pipeline | 2 | MNIST images grouped into sequences of 4 frames per digit |
| 2 | Feature Extraction using Pretrained CNNs | 3 | ResNet-18 and MobileNet-V2 used as frozen feature extractors |
| 3 | Fine-Tuning Pretrained CNN | 3 | ResNet-18 with layer4 unfrozen for transfer learning on MNIST |
| 4 | Embedding Usage | 2 | CNN features projected to fixed size + positional embeddings added |
| 5 | Attention-Based Model | 2 | Scaled dot-product attention over sequence output frames |
| 6 | RNN / LSTM / GRU Implementation | 4 | All three implemented inside SequenceClassifier, trained separately |
| 7 | Hyperparameter Experimentation | 1 | Grid search over learning rate × hidden size (4 combinations) |
| 8 | Model Comparison & Evaluation Metrics | 2 | Accuracy, Precision, Recall, F1, Confusion Matrix for all 3 models |
| 9 | Code Organisation | 1 | Modular sections per criterion, fixed seed, all results saved as plots |

---

## Model Pipeline

```
Input Frames (4 MNIST images of same digit)
            ↓
  CNN Feature Extractor
  (ResNet-18 or MobileNet-V2 — frozen)
            ↓
  Temporal Embedding
  (feature compression + positional info)
            ↓
  RNN / LSTM / GRU
  (captures dependencies across frames)
            ↓
  Temporal Attention
  (focuses on most important frames)
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

MNIST will be downloaded automatically on the first run into the `./data` folder.

---

## Global Settings

| Parameter | Value |
|---|---|
| Sequence Length | 4 frames per sample |
| Batch Size | 32 |
| Epochs | 5 |
| Embed Dim | 256 |
| Hidden Dim | 256 |
| LSTM / RNN / GRU Layers | 2 |
| Dropout | 0.3 |
| Learning Rate | 0.001 |
| Number of Classes | 10 (digits 0–9) |
| Random Seed | 42 |

---

## Hyperparameter Search

Tested the following 4 combinations and selected the best by validation accuracy:

| Learning Rate | Hidden Dim |
|---|---|
| 0.001 | 128 |
| 0.001 | 256 |
| 0.0005 | 128 |
| 0.0005 | 256 |

---

## Model Comparison

All three models share the same pipeline. Only the sequence model changes:

| Model | Strengths |
|---|---|
| RNN | Simple, fast, struggles with long sequences |
| LSTM | Has forget and input gates, handles long-term memory well |
| GRU | Simpler than LSTM, similar performance, fewer parameters |

Final comparison is done using Accuracy, Precision, Recall, F1 Score and Confusion Matrix.

---

## Output Files

| File | Description |
|---|---|
| `sample_sequences.png` | Shows 3 sample MNIST sequences used for training |
| `training_curves.png` | Validation accuracy and loss per epoch for all 3 models |
| `confusion_matrices.png` | Predicted vs True labels for RNN, LSTM, GRU side by side |
| `model_accuracy_bar.png` | Bar chart of final test accuracy for all 3 models |

---

## Requirements

```
torch
torchvision
scikit-learn
matplotlib
seaborn
numpy
```

---


**Course:** 24AI636 Deep Learning
**Project:** Mini-Project 2
**Review:** 2 — 5th March 2026
