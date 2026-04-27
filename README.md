# TH1-2025 — Meditation Level Research Workflow

This project predicts meditation level from raw EEG CSV files and compares three model families:
1) SVM (RBF), 2) Random Forest, 3) Sequence models (LSTM / Bi-LSTM track).

<img width="763" height="388" alt="image" src="https://github.com/user-attachments/assets/16ae08e7-fc7c-455e-9707-b583070af595" />

## 1) Raw data input

- Input files are CSVs loaded by:
  - `/home/runner/work/TH1-2025/TH1-2025/train_meditation_models.py`
  - `/home/runner/work/TH1-2025/TH1-2025/train_sequence_models.py`
- Expected data includes:
  - EEG band-power columns for Muse channels (TP9, AF7, AF8, TP10)
  - label column (default: `label`)
  - optional subject column (default: `subject_id`)
  - optional quality/motion columns (HSI, HeadBandOn, Accelerometer)

## 2) Shared preprocessing and feature extraction

Implemented in `/home/runner/work/TH1-2025/TH1-2025/meditation_pipeline.py` (`MeditationFeaturePipeline`).

Subprocess:
- Detect schema (band columns, IMU, timestamps, quality columns)
- Estimate sampling rate from timestamps
- Quality gating:
  - headband on check
  - HSI quality check
- Motion gating using accelerometer stability
- Sliding windows (default: 2 sec window, 1 sec stride)
- Extract 19 engineered features:
  - relative band powers
  - per-band spread
  - cross-band ratios
  - frontal asymmetry and regional means
- Align/clean final feature matrix via `align_features(...)`

## 3) Window-level label generation

- Labels are converted from row-level to window-level by majority vote (mode) over each window.
- Windows with invalid labels are removed.

## 4) Subject-wise split and leakage prevention

- Train/validation split uses `GroupShuffleSplit` (group = subject).
- Cross-validation uses `GroupKFold`.
- This prevents train/validation leakage across the same subject.

## 5) Model branch A: SVM (RBF)

In `train_meditation_models.py`:
- Pipeline per CV fold:
  - imputation (`SimpleImputer`)
  - scaling (`StandardScaler`)
  - imbalance handling (`SMOTE` inside CV)
  - classifier (`SVC`)
- Hyperparameter search: `GridSearchCV`
- Probability calibration: `CalibratedClassifierCV`
- Metrics: macro-F1, balanced accuracy, macro recall, confusion matrix
- Confidence-based reject threshold selection for uncertain predictions

## 6) Model branch B: Random Forest

In `train_meditation_models.py`:
- Same preprocessing/SMOTE pipeline, classifier = `RandomForestClassifier`
- Hyperparameter search with GridSearchCV
- Probability calibration
- Same evaluation protocol as SVM for apples-to-apples comparison

## 7) Model branch C: Sequence models (LSTM / Bi-LSTM track)

In `train_sequence_models.py`:
- Start from same 19 window features
- Build fixed-length sequences per subject (`--seq-len`, `--step`)
- Train:
  - LSTM
  - Bi-LSTM
- Use class weights, dropout, and early stopping
- Evaluate macro-F1 and balanced accuracy on subject-wise split

## 8) Model selection and comparison

- Classical bundle (SVM/RF): best model selected by validation macro-F1.
- For research reporting across 3 model families:
  - compare macro-F1 (primary)
  - compare balanced accuracy
  - inspect confusion matrix / class-wise recall
  - include uncertainty coverage (where reject threshold is applied)

## 9) Inference on new data

Use `/home/runner/work/TH1-2025/TH1-2025/test_mediation_classifier.py`:
- Load unified bundle (`models/meditation_model_bundle.pkl`) or legacy fallback artifacts
- Re-run same feature pipeline
- Predict per window with confidence
- Optionally mark low-confidence windows as `Uncertain`
- Save:
  - `<input>_predictions.csv`
  - `<input>_timeline.png`

## 10) Reproducible run sequence

From `/home/runner/work/TH1-2025/TH1-2025`:

```bash
# A) Train SVM + RF bundle
python train_meditation_models.py \
  --input path/to/S1.csv path/to/S2.csv path/to/S3.csv \
  --label-col label \
  --subject-col subject_id \
  --out-dir models

# B) Train sequence models
python train_sequence_models.py \
  --input path/to/S1.csv path/to/S2.csv path/to/S3.csv \
  --label-col label \
  --subject-col subject_id \
  --out-dir models

# C) Run inference
python test_mediation_classifier.py path/to/new_subject.csv
```

## 11) Advisor-ready summary

This project is an end-to-end and reproducible pipeline:
- raw EEG CSV -> gated window features -> subject-wise training/evaluation
- direct comparison of 3 model families (SVM, RF, sequence)
- confidence-aware inference outputs for practical deployment analysis.

