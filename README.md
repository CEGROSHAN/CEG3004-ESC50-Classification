# CEG3004 DSP Mini-Project — Environmental Sound Classification

**Group ID:** Pr_27  
**Course:** CEG3004 — Digital Signal Processing

---

## Project Overview

This project implements a robust audio classification pipeline for the **ESC-50** dataset — 2000 labeled audio clips across 50 environmental sound classes (e.g. dog, rain, siren, chainsaw). The model is evaluated on three test conditions:

| Condition | Description |
|---|---|
| `__clean` | Original unmodified audio |
| `__noisy` | Additive Gaussian noise applied |
| `__bandlimited` | Frequency content restricted via low-pass filter |

---

## Repository Structure

```
CEG3004-Pr_27/
├── CEG3004_Project_Colab.ipynb   # Main notebook — all code
└── README.md                     # This file
```

> The model file (`Pr_27_model.joblib`) and predictions (`Pr_27_predictions.csv`) are submitted to xsite Dropbox — not stored in this repo due to file size.

---

## How to Reproduce

### Requirements
All dependencies are installed automatically by the notebook. A Google account and Google Colab are required.

### Steps

1. Open `CEG3004_Project_Colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set `GROUP_ID = "Pr_27"` in Cell 1
3. Run all cells in order: **Runtime → Run all**
4. The notebook will automatically:
   - Install all dependencies
   - Download and extract the ESC-50 dataset
   - Extract features from all 2000 training clips
   - Train the SVM classifier
   - Save and download `Pr_27_model.joblib`
   - Generate and download `Pr_27_predictions.csv`

> Feature extraction + training takes approximately **8–12 minutes** on Colab free tier.

### Dependencies

```
numpy, scipy, pandas, scikit-learn, librosa, soundfile, tqdm
```

---

## DSP Pipeline

### 1. Preprocessing (`preprocess_audio` — Cell 5)

| Step | Method | Rationale |
|---|---|---|
| Silence trimming | `librosa.effects.trim(top_db=20)` | Removes dead air so features focus on the actual sound event |
| Fixed-length padding | Pad/truncate to exactly 5s | Ensures consistent feature matrix dimensions across all clips |
| Pre-emphasis filter | `y[t] = y[t] − 0.97 × y[t−1]` | Boosts high frequencies; improves robustness to bandlimited clips |
| Peak normalisation | Scale to max amplitude = 1.0 | Removes loudness variability so classifier focuses on spectral shape |

### 2. Feature Extraction (`extract_features` — Cell 6)

| Feature Group | Dim | Rationale |
|---|---|---|
| MFCC (40 coeffs) + Δ + ΔΔ | 600 | Timbral texture + temporal dynamics. Deltas capture how spectrum evolves over time. |
| Log-mel spectrogram (64 bins) | 320 | Perceptual frequency representation aligned with human hearing |
| Spectral centroid | 5 | Brightness — distinguishes bright (glass breaking) vs dull (footsteps) sounds |
| Spectral bandwidth | 5 | Width of spectral energy — narrow for tonal, wide for noise |
| Spectral rolloff | 5 | Frequency below which 85% of energy lies — separates tonal from noisy sounds |
| Zero-crossing rate | 5 | Proxy for percussiveness and noisiness |
| RMS energy | 5 | Loudness envelope shape over time |
| Chroma features (12 bins) | 60 | Harmonic/pitch class content |

**Statistical pooling:** Each feature matrix is summarised with mean, std, median, 10th and 90th percentile across time frames. Percentiles provide robustness to outlier frames caused by noise.

**Total feature vector: ~1005 dimensions**

### 3. Augmentation (`augment_audio` — Cell 6)

Applied **only to the training split** to prevent data leakage into validation:

| Strategy | Implementation | Simulates |
|---|---|---|
| No change | — | `__clean` test clips |
| Additive Gaussian noise | `y += randn × 0.005` | `__noisy` test clips |
| Low-pass filter | Butterworth 4th order, 4kHz cutoff | `__bandlimited` test clips |

### 4. Classifier (Cell 8)

**SVM with RBF kernel** — `C=100`, `gamma='scale'`, `class_weight='balanced'`

- RBF kernel handles non-linear separation across 50 classes in high-dimensional feature space
- C=100 gives the decision boundary enough flexibility for ~1000-dimensional features
- `class_weight='balanced'` handles minor class imbalance in ESC-50

---

## Results

| Split | Macro-F1 |
|---|---|
| Validation (clean, 20% holdout) | 0.6471 |

---

## Experiments

| Experiment | Key Change | Macro-F1 |
|---|---|---|
| Baseline | 20 MFCCs, Logistic Regression | ~0.25 |
| + Richer features | 40 MFCCs + mel + spectral + chroma | ~0.35 |
| + Better model | SVM RBF C=10 | ~0.41 |
| + Fix data leakage + C=100 | Split before aug, higher C | **0.6471** |

---

## Error Analysis

Hardest classes (from confusion matrix):
- `airplane` / `helicopter` / `engine` — all sustained low-frequency mechanical sounds
- `washing_machine` / `vacuum_cleaner` — similar broadband noise profiles  
- `mouse_click` / `keyboard_typing` — very short transients, mostly silence

Easiest classes: `thunderstorm`, `sneezing`, `clapping` — highly distinctive spectral signatures stable across distortion conditions.
