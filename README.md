# HMR-BiLSTM: A Trustworthy and Explainable Hybrid Memory Residual BiLSTM Framework for ECG Arrhythmia Classification

## Overview

HMR-BiLSTM is a deep learning framework for ECG arrhythmia classification on the MIT-BIH Arrhythmia Dataset. The architecture is designed to improve temporal representation learning, adversarial robustness, and explainability for safety-critical healthcare AI systems.

### Key Components

- **Residual Memory Control (RMC):** Hybrid memory path with softmax-based memory decomposition for adaptive long-term dependency modeling.
- **Bidirectional LSTM:** Captures both forward and backward temporal context in ECG signals.
- **CNN Feature Extractor:** 1D convolutional front-end for local morphological feature extraction.
- **Attention Pooling:** Learnable temporal attention mechanism for sequence aggregation.
- **Focal Loss + Class Weights:** Addresses severe class imbalance (N >> S, V, F, Q).
- **Adversarial Training (FGSM):** Improves robustness against adversarial perturbations.
- **Temporal Smoothness Regularization:** Encourages stable residual gate trajectories.

---

## Dataset

This project uses the [MIT-BIH Arrhythmia Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

After downloading, place the CSV files in `data/raw/`:

```
data/raw/
├── mitbih_train.csv
└── mitbih_test.csv
```

**5-class AAMI mapping:** N (Normal), S (Supraventricular), V (Ventricular), F (Fusion), Q (Unknown).

---

## Project Structure

```
HMR-BiLSTM/
│
├── data/
│   ├── raw/                          # Raw MIT-BIH CSV files
│   └── processed/                    # Preprocessed .npz splits + class weights
│
├── results/
│   ├── checkpoints/                  # Model checkpoints (.pt)
│   ├── figures/                      # Generated plots and visualizations
│   ├── logs/                         # Training logs (JSON)
│   ├── tables/                       # LaTeX and CSV result tables
│   └── ablation/                     # Ablation study results
│       ├── checkpoints/              # Ablation variant checkpoints
│       ├── ablation_results.json     # Full ablation metrics
│       ├── ablation_table.csv        # Summary table (CSV)
│       └── ablation_table.tex        # Summary table (LaTeX)
│
├── preprocess.py                     # Data preprocessing and train/val/test split
├── rlstm_model.py                    # HMR-BiLSTM model architecture + RLSTMLoss
├── rlstm_model_ablation.py           # Ablation variants (No-RMC, No-CNN, Mean-Pool)
├── train.py                          # Main training script (HMR-BiLSTM)
├── run_baselines.py                  # Train baseline models (LSTM, BiLSTM)
├── run_ablation.py                   # Ablation study (train + evaluate 4 variants)
├── report_results.py                 # Generate figures (confusion matrix, ROC, gates)
├── evaluate_fgsm.py                  # FGSM adversarial robustness evaluation
├── evaluate_pgd.py                   # PGD adversarial robustness evaluation
├── evaluate_calibration.py           # Calibration analysis (reliability diagram, ECE)
├── evaluate_robustness_all.py        # Gaussian noise robustness evaluation
├── compare_fgsm_baselines.py         # FGSM comparison across all models
├── generate_results_tables.py        # Generate LaTeX/CSV summary tables
├── run_all.bat                       # Windows batch script to run full pipeline
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Installation

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# For GPU support (NVIDIA CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### 1. Data Preprocessing

```bash
python preprocess.py
```

Splits the raw MIT-BIH dataset into train/val/test sets and computes class weights. Outputs saved to `data/processed/`.

### 2. Train HMR-BiLSTM (Proposed Model)

```bash
python train.py
```

Trains the main HMR-BiLSTM model with adversarial training (FGSM), focal loss, cosine annealing LR, and early stopping. Checkpoint saved to `results/checkpoints/best_rlstm.pt`.

### 3. Train Baseline Models

```bash
python run_baselines.py
```

Trains LSTM and BiLSTM baseline models for comparison. Checkpoints saved to `results/checkpoints/`.

### 4. Ablation Study

```bash
# Run all 4 variants (full, no_rmc, no_cnn, mean_pool)
python run_ablation.py

# Run specific variants
python run_ablation.py --variants no_rmc no_cnn

# Generate table from existing checkpoints only
python run_ablation.py --table-only
```

### 5. Evaluation & Visualization

```bash
# Generate confusion matrix, ROC curve, gate trajectories
python report_results.py

# FGSM adversarial robustness (all models)
python compare_fgsm_baselines.py

# PGD adversarial robustness
python evaluate_pgd.py

# Gaussian noise robustness
python evaluate_robustness_all.py

# Calibration analysis (reliability diagram, ECE)
python evaluate_calibration.py

# Generate LaTeX/CSV summary tables for paper
python generate_results_tables.py
```

### 6. Run Full Pipeline (Windows)

```bash
run_all.bat
```

---

## Results

### Clean Performance (Test Set)

| Model        | Accuracy | F1 (macro) | F1-S   | F1-V   | F1-F   | AUC    |
|-------------|----------|------------|--------|--------|--------|--------|
| LSTM         | 0.9700   | 0.8691     | 0.6881 | 0.9333 | 0.7288 | 0.9897 |
| BiLSTM       | 0.9693   | 0.8616     | 0.6575 | 0.9356 | 0.7179 | 0.9878 |
| **HMR-BiLSTM** | **0.9759** | **0.8921** | **0.7387** | **0.9492** | **0.7944** | **0.9917** |

### Adversarial Robustness (FGSM, ε=0.02)

| Model        | Clean F1 | FGSM F1 | F1 Drop | ASR    |
|-------------|----------|---------|---------|--------|
| LSTM         | 0.8691   | 0.8264  | 0.0427  | 0.0127 |
| BiLSTM       | 0.8616   | 0.7824  | 0.0792  | 0.0365 |
| **HMR-BiLSTM** | **0.8825** | **0.8555** | **0.0270** | **0.0079** |

---

## Research Contributions

- **Trustworthy healthcare AI:** Adversarial robustness via FGSM/PGD training and evaluation.
- **Explainable ECG classification:** Intrinsic interpretability through residual gate trajectory visualization.
- **Robust temporal modeling:** Hybrid memory decomposition with adaptive residual gating.
- **Clinical relevance:** Significant improvement on minority arrhythmia classes (S, V, F).

---

## Keywords

ECG Arrhythmia Classification, Explainable AI, Trustworthy AI, BiLSTM, Residual Memory Control, Adversarial Robustness, Biomedical Signal Processing, Deep Learning

---

## Citation

```bibtex
@article{hmr_bilstm_2026,
  title={HMR-BiLSTM: A Trustworthy and Explainable Hybrid Memory Residual BiLSTM Framework for ECG Arrhythmia Classification},
  author={Anonymous},
  year={2026}
}
```
