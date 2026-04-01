# CEG3004 DSP Mini-Project — Environmental Sound Classification

**Group ID:** `Pr_27`
**Course:** CEG3004 — Digital Signal Processing

---

## Project Overview

This project implements a robust audio classification pipeline for the **ESC-50** Environmental Sound Classification dataset. The model classifies 5-second mono audio clips into **50 sound categories** and is evaluated under three conditions:

- **Clean** — original unmodified audio
- **Noisy** — additive Gaussian noise applied
- **Bandlimited** — frequency content restricted (low-pass filtered)

The pipeline emphasises DSP-grounded feature engineering and training-time augmentation so that a single model generalises well across all three conditions.

---

## Repository Structure

```
├── CEG3004_Project_Colab.ipynb       # Main notebook (run end-to-end in Google Colab)
├── Pr_27_model.joblib                # Trained SVM model
├── Pr_27_predictions.csv             # Predictions on submission set (1200 rows)
├── README.md                         # This file
├── figures/
│   ├── confusion_matrix.png          # Validation confusion matrix
│   └── feature_visualisation.png     # Spectrogram comparison (clean / noisy / bandlimited)
└── github_link.txt                   # Link to this repository (for xsite submission)
```

---

## DSP Pipeline

### 1. Preprocessing (`preprocess_audio`)

| Step | What it does | Why it helps |
|---|---|---|
| Silence trimming (`top_db=20`) | Removes leading/trailing silence | Features describe the actual sound event, not dead air |
| Fixed-length pad/truncate (5 s) | Pads short clips with zeros, truncates long clips | Ensures consistent feature matrix dimensions across all clips |
| Pre-emphasis (α = 0.97) | First-order high-pass: `y[t] = y[t] − 0.97·y[t−1]` | Boosts high frequencies, compensating for natural spectral tilt; improves MFCC balance and robustness to bandlimited clips |
| Peak normalisation | Scales waveform so max |amplitude| = 1.0 | Removes loudness variability so the classifier focuses on spectral shape |

### 2. Feature Extraction (`extract_features`)

| Feature Group | Coefficients | Stats pooled | Dimensions | Rationale |
|---|---|---|---|---|
| MFCC + Δ + ΔΔ | 40 MFCCs | mean, std, median, p10, p90 | 600 | Timbral texture + temporal dynamics; 40 coefficients capture finer differences than the 20-coefficient baseline across 50 classes |
| Log-mel spectrogram | 64 mel bands | mean, std, median, p10, p90 | 320 | Perceptual frequency representation; complements MFCCs by preserving inter-band correlation |
| Spectral centroid | 1 | mean, std, median, p10, p90 | 5 | Brightness / perceived pitch of a sound |
| Spectral bandwidth | 1 | mean, std, median, p10, p90 | 5 | Width of energy distribution around the centroid |
| Spectral rolloff | 1 | mean, std, median, p10, p90 | 5 | Frequency below which 85 % of energy lies — separates tonal from noisy sounds |
| Zero-crossing rate | 1 | mean, std, median, p10, p90 | 5 | Percussiveness / noisiness proxy |
| RMS energy | 1 | mean, std, median, p10, p90 | 5 | Loudness envelope shape over time |
| Chroma (12 pitch classes) | 12 | mean, std, median, p10, p90 | 60 | Harmonic / pitch class content — helps separate musical and tonal environmental sounds |

**Total feature vector: 1 005 dimensions.**

The 10th and 90th percentile statistics provide robustness to outlier frames caused by noise bursts, ensuring stable feature values under the noisy test condition.

### 3. Training-Time Augmentation (`augment_audio`)

Each training clip is presented **twice** — once clean, once randomly augmented — doubling the effective training set from 2 000 to 4 000 samples. Augmentation randomly applies one of three strategies (equal probability):

| Strategy | Simulates | Parameters |
|---|---|---|
| No change | Clean test clips | — |
| Additive white Gaussian noise | `__noisy` test clips | σ = 0.005 (~20 dB SNR) |
| Low-pass Butterworth filter | `__bandlimited` test clips | 4th-order, 4 kHz cutoff |

By exposing the model to all three conditions during training, it learns features that are invariant to these distortions.

### 4. Classifier

**SVM with RBF kernel** wrapped in a `StandardScaler` pipeline.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `kernel` | `rbf` | Non-linear decision boundaries needed for 50 classes in 1 005-D space |
| `C` | 10 | Moderate regularisation — avoids overfitting while allowing some margin violations |
| `gamma` | `scale` | Adapts to feature variance: `1 / (n_features × X.var())` |
| `class_weight` | `balanced` | Compensates for any minor class imbalance in ESC-50 |
| `probability` | `True` | Enables soft probability estimates via Platt scaling |

**Why SVM over Logistic Regression?** Logistic Regression fits linear boundaries, which are insufficient for separating 50 classes of environmental sounds in a high-dimensional feature space. The RBF kernel implicitly maps features to an infinite-dimensional space, capturing complex non-linear class boundaries without manual feature interaction engineering.

---

## How to Reproduce

### Requirements

All dependencies are installed automatically by the first code cell in the notebook. Only a Google account and Google Colab are needed.

### Steps

1. Open `CEG3004_Project_Colab.ipynb` in **Google Colab**
2. Verify `GROUP_ID = "Pr_27"` in Cell 1
3. **Runtime → Run all**
4. The notebook will:
   - Download and extract the ESC-50 dataset
   - Visualise clean / noisy / bandlimited spectrograms
   - Extract features from all training clips (clean + augmented)
   - Train the SVM classifier
   - Print a classification report, confusion matrix, and per-class error analysis
   - Save and download `Pr_27_model.joblib`
   - Generate and download `Pr_27_predictions.csv`
5. Upload both files to the xsite Dropbox

> **Runtime estimate:** ~8–12 minutes on Colab free-tier CPU.

---

## Results

| Metric | Validation (80/20 stratified split) |
|---|---|
| Macro-F1 | *(fill in from notebook output)* |

*(Update with the actual Macro-F1 printed by the notebook after running.)*

---

## Experiments & Ablations

| # | Experiment | Change from previous | Macro-F1 |
|---|---|---|---|
| 1 | Baseline | 20 MFCCs + mean/std, Logistic Regression | *(run baseline to get this)* |
| 2 | More MFCCs | 40 MFCCs (finer timbral resolution) | — |
| 3 | Richer features | + log-mel, spectral shape, chroma, percentile pooling | — |
| 4 | Better model | SVM RBF (C=10) replaces Logistic Regression | — |
| 5 | Augmentation | + noise & bandlimit augmentation (2× training set) | — |

*(Fill in F1 scores incrementally if you ran each step separately. Otherwise note the final combined result.)*

---

## Error Analysis

The confusion matrix (`figures/confusion_matrix.png`) and per-class F1 printout in the notebook reveal which classes are hardest to separate. Common confusion patterns in ESC-50 include:

- **Engine-like sounds** (e.g., engine, helicopter, airplane) share similar broadband spectral profiles
- **Water sounds** (e.g., rain, water drops, sea waves) overlap in low-frequency energy
- **Human vocalisations** (e.g., laughing, crying baby, sneezing) can be confused when reduced to spectral statistics

The pre-emphasis filter and high-percentile pooling were added specifically to improve separation by emphasising high-frequency detail and suppressing transient noise artefacts.

The feature visualisation (`figures/feature_visualisation.png`) shows how additive noise raises the spectrogram floor uniformly while bandlimiting removes energy above the cutoff — confirming that percentile-based pooling and augmentation with both distortion types are sound design choices.

---

## Submission Checklist

- [x] `Pr_27_predictions.csv` — 1 200 rows (400 clips × 3 conditions)
- [x] `Pr_27_model.joblib` — trained SVM pipeline
- [x] GitHub repository with source code, README, and reproducible instructions
