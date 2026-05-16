
## HMR-BiLSTM: A Trustworthy and Explainable Hybrid Memory Residual BiLSTM Framework for ECG Arrhythmia Classification

### Overview

HMR-BiLSTM is a deep learning framework for ECG arrhythmia classification designed to improve temporal representation learning, robustness, and explainability in healthcare AI systems.

The proposed architecture integrates:

* Residual Memory Control (RMC)
* Hybrid Memory Path mechanisms
* Bidirectional LSTM modeling
* Temporal smoothness regularization
* Intrinsic softmax-based memory decomposition

The framework aims to address:

* Long-term temporal dependency limitations
* Severe class imbalance
* Limited interpretability in safety-critical medical environments

---

## Features

* Hybrid Residual BiLSTM architecture
* Explainable temporal gating mechanism
* Robust ECG heartbeat classification
* Minority arrhythmia class enhancement
* Temporal gate trajectory visualization
* Noise robustness evaluation
* Baseline comparison experiments

---

## Dataset

This project uses the MIT-BIH Arrhythmia Dataset.

Dataset source:

https://www.kaggle.com/datasets/shayanfazeli/heartbeat

After downloading, place the dataset files inside:

```text
data/raw/
```

Required files:

```text
mitbih_train.csv
mitbih_test.csv
```

---

## Project Structure

```text
HMR-BiLSTM/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── checkpoints/
│   ├── figures/
│   └── logs/
│
├── preprocess.py
├── train.py
├── rlstm_model.py
├── run_baselines.py
├── report_results.py
├── evaluate_robustness_all.py
├── requirements.txt
└── README.md
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run training:

```bash
python train.py
```

---

## Baseline Evaluation

```bash
python run_baselines.py
```

---

## Robustness Evaluation

```bash
python evaluate_robustness_all.py
```

---

## Results

Experimental outputs are stored in:

```text
results/
```

Including:

* Confusion matrices
* ROC curves
* Robustness analysis
* Training logs
* Gate trajectory visualization

---

## Research Contributions

The proposed HMR-BiLSTM framework contributes toward:

* Trustworthy healthcare AI
* Explainable ECG classification
* Robust temporal representation learning
* Clinically interpretable deep learning systems

---

## Keywords

ECG Arrhythmia Classification, Explainable AI, Trustworthy AI, BiLSTM, Deep Learning, Biomedical Signal Processing, Residual Memory Control, AI Safety

---

## Citation

```bibtex
@article{hmr_bilstm_2026,
  title={HMR-BiLSTM: A Trustworthy and Explainable Hybrid Memory Residual BiLSTM Framework for ECG Arrhythmia Classification},
  author={Anonymous},
  year={2026}
}
```
