import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Custom imports
from rlstm_model import RLSTMClassifier
from report_results import load_rlstm_model, collect_predictions_and_gates
from evaluate_fgsm import load_baseline_model
from run_baselines import flatten_sequences, LSTMBaseline, NUM_CLASSES

def evaluate_sklearn_model_noise(model, X_test, y_test, noise_levels):
    f1_scores = []
    X_test_flat = flatten_sequences(X_test)
    for std in noise_levels:
        if std > 0:
            noise = np.random.normal(0, std, X_test_flat.shape)
            X_noisy = X_test_flat + noise
        else:
            X_noisy = X_test_flat
        preds = model.predict(X_noisy)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        f1_scores.append(f1)
    return f1_scores

def evaluate_torch_model_noise(model, X_test, y_test, device, noise_levels, is_rlstm=False):
    f1_scores = []
    model.eval()
    for std in noise_levels:
        if std > 0:
            noise = np.random.normal(0, std, X_test.shape)
            X_noisy = X_test + noise
        else:
            X_noisy = X_test
            
        if is_rlstm:
            preds, _, _ = collect_predictions_and_gates(model, X_noisy, y_test, device)
        else:
            test_ds = TensorDataset(torch.from_numpy(X_noisy).float(), torch.from_numpy(y_test).long())
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            all_preds = []
            with torch.no_grad():
                for X, _ in test_loader:
                    X = X.to(device)
                    preds = model(X).argmax(-1).cpu().numpy()
                    all_preds.append(preds)
            preds = np.concatenate(all_preds)
            
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        f1_scores.append(f1)
    return f1_scores

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("[Loading data]")
    train = np.load("data/processed/train.npz")
    val   = np.load("data/processed/val.npz")
    test  = np.load("data/processed/test.npz")
    X_tr, y_tr = train["X"], train["y"]
    X_va, y_va = val["X"], val["y"]
    X_te, y_te = test["X"], test["y"]
    
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {"Noise Std": noise_levels}
    
    # 1. Logistic Regression
    print("\n[Evaluating Logistic Regression]")
    lr = LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced", n_jobs=-1, solver="lbfgs")
    lr.fit(flatten_sequences(X_tr), y_tr)
    results["LR"] = evaluate_sklearn_model_noise(lr, X_te, y_te, noise_levels)
    
    # 2. Decision Tree
    print("\n[Evaluating Decision Tree]")
    dt = DecisionTreeClassifier(max_depth=15, min_samples_leaf=10, class_weight="balanced", random_state=42)
    dt.fit(flatten_sequences(X_tr), y_tr)
    results["DT"] = evaluate_sklearn_model_noise(dt, X_te, y_te, noise_levels)
    
    # 3. LSTM
    print("\n[Evaluating LSTM Baseline]")
    lstm_ckpt = Path("results/checkpoints/best_lstm.pt")
    if lstm_ckpt.exists():
        lstm_model, _ = load_baseline_model(lstm_ckpt, device)
        results["LSTM"] = evaluate_torch_model_noise(lstm_model, X_te, y_te, device, noise_levels, is_rlstm=False)
    else:
        print("[WARNING] best_lstm.pt not found. Skipping LSTM.")
        results["LSTM"] = [0.0] * len(noise_levels)

    # 4. BiLSTM
    print("\n[Evaluating BiLSTM Baseline]")
    bilstm_ckpt = Path("results/checkpoints/best_bilstm.pt")
    if bilstm_ckpt.exists():
        bilstm_model, _ = load_baseline_model(bilstm_ckpt, device)
        results["BiLSTM"] = evaluate_torch_model_noise(bilstm_model, X_te, y_te, device, noise_levels, is_rlstm=False)
    else:
        print("[WARNING] best_bilstm.pt not found. Skipping BiLSTM.")
        results["BiLSTM"] = [0.0] * len(noise_levels)
    
    # 5. HMR-BiLSTM
    print("\n[Evaluating HMR-BiLSTM]")
    # Guard against missing checkpoints
    rlstm_ckpt = Path("results/checkpoints/best_rlstm.pt")
    if rlstm_ckpt.exists():
        input_size = X_te.shape[-1] if len(X_te.shape) > 2 else 1
        rlstm_model, _ = load_rlstm_model(str(rlstm_ckpt), device, input_size)
        results["HMR-BiLSTM"] = evaluate_torch_model_noise(rlstm_model, X_te, y_te, device, noise_levels, is_rlstm=True)
    else:
        print("[WARNING] best_rlstm.pt not found. Skipping HMR-BiLSTM.")
        results["HMR-BiLSTM"] = [0.0] * len(noise_levels)
    
    # Print results table
    print("\n" + "="*80)
    print(f"{'Noise Std':<10} | {'LR':<10} | {'DT':<10} | {'LSTM':<10} | {'BiLSTM':<10} | {'HMR-BiLSTM':<10}")
    print("-" * 80)
    for i, std in enumerate(noise_levels):
        print(f"{std:<10.2f} | {results['LR'][i]:<10.4f} | {results['DT'][i]:<10.4f} | {results['LSTM'][i]:<10.4f} | {results['BiLSTM'][i]:<10.4f} | {results['HMR-BiLSTM'][i]:<10.4f}")
        
    # Plotting
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_path = fig_dir / "robustness_noise_all.png"
    
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"LR": "#9E9E9E", "DT": "#FFA726", "LSTM": "#42A5F5", "BiLSTM": "#5C6BC0", "HMR-BiLSTM": "#43A047"}
    markers = {"LR": "v", "DT": "^", "LSTM": "x", "BiLSTM": "d", "HMR-BiLSTM": "o"}
    
    for model_name in ["LR", "DT", "LSTM", "BiLSTM", "HMR-BiLSTM"]:
        ax.plot(noise_levels, results[model_name], marker=markers[model_name], 
                linewidth=2, color=colors[model_name], label=model_name)
    
    ax.set_xlabel("Gaussian Noise Std", fontsize=12)
    ax.set_ylabel("Macro F1-Score", fontsize=12)
    ax.set_title("Model Robustness to Gaussian Noise (Macro F1)", fontsize=13)
    ax.set_ylim([0, 1.0])
    ax.set_xlim([0, max(noise_levels) + 0.02])
    ax.grid(alpha=0.4, linestyle="--")
    ax.legend(loc="lower left", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Saved comprehensive robustness chart to {save_path}")

if __name__ == "__main__":
    main()
