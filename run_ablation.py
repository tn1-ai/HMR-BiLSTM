"""
run_ablation.py
===============
Chạy ablation study cho HMR-BiLSTM trên MIT-BIH ECG.

Các variant:
    full        — HMR-BiLSTM đầy đủ (dùng làm reference)
    no_rmc      — bỏ RMC path, c_t = c_lstm  ← QUAN TRỌNG NHẤT
    no_cnn      — bỏ CNN, raw ECG đi thẳng vào BiRLSTM
    mean_pool   — thay AttentionPooling bằng mean pooling

Cách chạy:
    # Chạy tất cả 4 variants (mất nhiều thời gian):
    python run_ablation.py

    # Chỉ chạy variant quan trọng nhất:
    python run_ablation.py --variants no_rmc

    # Chạy 2 variants:
    python run_ablation.py --variants no_rmc no_cnn

    # Bỏ qua training, chỉ generate bảng từ checkpoint có sẵn:
    python run_ablation.py --table-only

Kết quả lưu tại:
    results/ablation/checkpoints/best_rlstm_{variant}.pt
    results/ablation/ablation_results.json
    results/ablation/ablation_table.csv
    results/ablation/ablation_table.tex
"""

import argparse
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)

from rlstm_model_ablation import RLSTMClassifier, RLSTMLoss


# =============================================================================
#  CONFIG — giữ nguyên giống train.py để so sánh fair
# =============================================================================
BASE_CONFIG = {
    "data_dir":                "data/processed",
    "seed":                    42,
    "batch_size":              128,
    "hidden_size":             96,
    "dropout":                 0.25,
    "learning_rate":           1e-3,
    "min_lr":                  1e-5,
    "weight_decay":            1e-4,
    "lambda_smooth":           0.003,
    "epochs":                  45,
    "early_stopping_patience": 8,
    "grad_clip":               1.0,
    "num_classes":             5,
    "input_size":              1,
    "cnn_out_channels":        64,
    "num_layers":              2,
    "use_focal_loss":          True,
    "focal_gamma":             1.5,
    "adversarial_training":    True,
    "adv_epsilon":             0.02,
    "adv_ratio":               0.3,
    "use_class_weights":       True,
}

# Định nghĩa 4 variants
VARIANTS = {
    "full": {
        "use_rmc": True, "use_cnn": True, "use_attention": True,
        "label": "HMR-BiLSTM (full)",
    },
    "no_rmc": {
        "use_rmc": False, "use_cnn": True, "use_attention": True,
        "label": "No-RMC (c_t = c_lstm)",
    },
    "no_cnn": {
        "use_rmc": True, "use_cnn": False, "use_attention": True,
        "label": "No-CNN (raw input)",
    },
    "mean_pool": {
        "use_rmc": True, "use_cnn": True, "use_attention": False,
        "label": "Mean-Pool (no attention)",
    },
}


# =============================================================================
#  UTILITIES
# =============================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def cosine_lr(epoch, total_epochs, base_lr, min_lr):
    progress = epoch / max(1, total_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def load_data(data_dir, batch_size):
    def make_loader(split, shuffle=False):
        d = np.load(f"{data_dir}/{split}.npz")
        ds = TensorDataset(
            torch.from_numpy(d["X"]).float(),
            torch.from_numpy(d["y"]).long(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return (
        make_loader("train", shuffle=True),
        make_loader("val"),
        make_loader("test"),
    )


@torch.no_grad()
def evaluate(model, loader, device, num_classes=5):
    model.eval()
    all_logits, all_y = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        all_logits.append(logits.cpu())
        all_y.append(y)

    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    probs  = torch.softmax(logits, dim=-1).numpy()
    preds  = logits.argmax(dim=-1).numpy()

    metrics = {
        "accuracy":        float(accuracy_score(y_true, preds)),
        "precision_macro": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "f1_macro":        float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "f1_weighted":     float(f1_score(y_true, preds, average="weighted", zero_division=0)),
    }
    try:
        metrics["auc_ovr"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["auc_ovr"] = 0.0

    # Per-class F1 cho S, V, F (clinically important)
    per_class = f1_score(y_true, preds, average=None, zero_division=0)
    for i, cls in enumerate(["N", "S", "V", "F", "Q"]):
        metrics[f"f1_{cls}"] = float(per_class[i]) if i < len(per_class) else 0.0

    metrics["_preds"]  = preds
    metrics["_y_true"] = y_true
    return metrics


def fgsm_attack_train(model, x, y, epsilon, criterion):
    """Generate FGSM adversarial examples during training."""
    x_adv = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        logits = model(x_adv)
        loss, _ = criterion(logits, y, r_fwd=None, r_bwd=None)
        model.zero_grad()
        loss.backward()
    return (x + epsilon * x_adv.grad.sign()).detach()


def train_one_epoch(model, loader, optimizer, criterion, device, cfg):
    model.train()
    total_loss, n = 0.0, 0

    adv_training = cfg["adversarial_training"]
    adv_epsilon  = cfg["adv_epsilon"]
    adv_ratio    = cfg["adv_ratio"]

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        if adv_training and adv_epsilon > 0:
            split  = int(len(X) * (1 - adv_ratio))
            X_adv  = fgsm_attack_train(
                model, X[split:], y[split:], adv_epsilon, criterion
            )
            X = torch.cat([X[:split], X_adv], dim=0)
            y = torch.cat([y[:split], y[split:]], dim=0)

        optimizer.zero_grad()

        logits, internals = model(X, return_internals=True)
        loss, _ = criterion(
            logits, y,
            r_fwd=internals["r_fwd"],
            r_bwd=internals["r_bwd"],
        )

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        n          += X.size(0)

    return total_loss / max(1, n)


# =============================================================================
#  TRAIN ONE VARIANT
# =============================================================================
def train_variant(variant_name, variant_flags, cfg, device,
                  train_loader, val_loader, test_loader,
                  class_weights, out_dir):
    label = variant_flags["label"]
    print(f"\n{'='*60}")
    print(f"  Variant: {label}")
    print(f"  use_rmc={variant_flags['use_rmc']}  "
          f"use_cnn={variant_flags['use_cnn']}  "
          f"use_attention={variant_flags['use_attention']}")
    print(f"{'='*60}")

    set_seed(cfg["seed"])

    model = RLSTMClassifier(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        dropout=cfg["dropout"],
        num_classes=cfg["num_classes"],
        cnn_out_channels=cfg["cnn_out_channels"],
        num_layers=cfg["num_layers"],
        use_rmc=variant_flags["use_rmc"],
        use_cnn=variant_flags["use_cnn"],
        use_attention=variant_flags["use_attention"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = RLSTMLoss(
        lambda_smooth=cfg["lambda_smooth"],
        class_weights=class_weights,
        use_focal=cfg["use_focal_loss"],
        focal_gamma=cfg["focal_gamma"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    ckpt_path     = out_dir / f"best_rlstm_{variant_name}.pt"
    best_f1       = 0.0
    best_epoch    = 0
    patience_cnt  = 0
    history       = []

    for epoch in range(1, cfg["epochs"] + 1):
        lr = cosine_lr(epoch - 1, cfg["epochs"],
                       cfg["learning_rate"], cfg["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0         = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg
        )
        val_m = evaluate(model, val_loader, device, cfg["num_classes"])
        elapsed = time.time() - t0

        marker = ""
        if val_m["f1_macro"] > best_f1:
            best_f1, best_epoch, patience_cnt = val_m["f1_macro"], epoch, 0
            torch.save({
                "model_state":   model.state_dict(),
                "config":        cfg,
                "variant_flags": variant_flags,
                "variant_name":  variant_name,
                "epoch":         epoch,
                "val_f1_macro":  best_f1,
                "n_params":      n_params,
            }, ckpt_path)
            marker = " <-- best"
        else:
            patience_cnt += 1

        history.append({
            "epoch": epoch, "lr": lr,
            "train_loss": train_loss,
            "val_f1_macro": val_m["f1_macro"],
            "val_accuracy": val_m["accuracy"],
        })

        print(f"  ep{epoch:3d} | lr={lr:.5f} | loss={train_loss:.4f} | "
              f"val_f1={val_m['f1_macro']:.4f} "
              f"acc={val_m['accuracy']:.4f} | {elapsed:.0f}s{marker}")

        if patience_cnt >= cfg["early_stopping_patience"]:
            print(f"\n  [Early stop] ep={epoch} best={best_epoch} F1={best_f1:.4f}")
            break

    # Load best và evaluate test
    print(f"\n  [Loading best checkpoint — epoch {best_epoch}]")
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    test_m = evaluate(model, test_loader, device, cfg["num_classes"])

    report = classification_report(
        test_m["_y_true"], test_m["_preds"],
        target_names=["N", "S", "V", "F", "Q"],
        zero_division=0, digits=4,
    )

    print(f"\n  Test F1 macro: {test_m['f1_macro']:.4f}")
    print(f"  Test Accuracy: {test_m['accuracy']:.4f}")
    print(f"  Test AUC:      {test_m['auc_ovr']:.4f}")
    print(f"\n  Per-class:\n{report}")

    # Xóa keys nội bộ trước khi lưu
    clean_test = {k: v for k, v in test_m.items()
                  if not k.startswith("_")}

    return {
        "variant":      variant_name,
        "label":        label,
        "n_params":     n_params,
        "best_epoch":   best_epoch,
        "best_val_f1":  best_f1,
        "test_metrics": clean_test,
        "report":       report,
        "history":      history,
        "flags": {
            "use_rmc":       variant_flags["use_rmc"],
            "use_cnn":       variant_flags["use_cnn"],
            "use_attention": variant_flags["use_attention"],
        },
    }


# =============================================================================
#  GENERATE TABLE
# =============================================================================
def generate_table(results, out_dir):
    """
    Tạo bảng so sánh ablation, format cho paper.

    Cột: Variant | Params | Acc | F1-macro | F1-S | F1-V | F1-F | AUC
    """
    rows = []
    for r in results:
        tm = r["test_metrics"]
        rows.append({
            "Variant":  r["label"],
            "Params":   f"{r['n_params']:,}",
            "Acc":      f"{tm['accuracy']:.4f}",
            "F1-macro": f"{tm['f1_macro']:.4f}",
            "F1-S":     f"{tm.get('f1_S', 0.0):.4f}",
            "F1-V":     f"{tm.get('f1_V', 0.0):.4f}",
            "F1-F":     f"{tm.get('f1_F', 0.0):.4f}",
            "AUC":      f"{tm['auc_ovr']:.4f}",
            "Best Ep":  str(r["best_epoch"]),
        })

    cols = ["Variant", "Params", "Acc", "F1-macro", "F1-S", "F1-V", "F1-F", "AUC", "Best Ep"]

    # CSV
    csv_path = out_dir / "ablation_table.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(row[c] for c in cols) + "\n")
    print(f"\n[OK] ablation_table.csv -> {csv_path}")

    # LaTeX
    tex_path = out_dir / "ablation_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        col_fmt = "l" + "r" * (len(cols) - 1)
        f.write("% Requires \\usepackage{booktabs} in LaTeX preamble\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation Study --- MIT-BIH ECG Test Set}\n")
        f.write("\\label{tab:ablation}\n")
        f.write(f"\\begin{{tabular}}{{{col_fmt}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(cols) + " \\\\\n")
        f.write("\\midrule\n")
        for i, row in enumerate(rows):
            # Bold dòng full model (index 0 nếu "full" chạy trước)
            if row["Variant"].startswith("HMR-BiLSTM (full)"):
                line = " & ".join(
                    f"\\textbf{{{row[c]}}}" for c in cols
                ) + " \\\\\n"
            else:
                line = " & ".join(row[c] for c in cols) + " \\\\\n"
            f.write(line)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[OK] ablation_table.tex -> {tex_path}")

    # Console print
    print("\n" + "=" * 80)
    print("  ABLATION RESULTS")
    print("=" * 80)
    header = f"{'Variant':<30} {'Params':>10} {'F1-mac':>8} {'F1-S':>7} {'F1-V':>7} {'F1-F':>7} {'AUC':>8}"
    print(header)
    print("-" * 80)
    for row in rows:
        print(
            f"{row['Variant']:<30} {row['Params']:>10} "
            f"{row['F1-macro']:>8} {row['F1-S']:>7} "
            f"{row['F1-V']:>7} {row['F1-F']:>7} {row['AUC']:>8}"
        )
    print("=" * 80)
    print("\nNote: F1-S / F1-V / F1-F = per-class F1 for Supraventricular, "
          "Ventricular, Fusion (clinically most important)")


# =============================================================================
#  MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="HMR-BiLSTM Ablation Study")
    parser.add_argument(
        "--variants", nargs="*",
        choices=["full", "no_rmc", "no_cnn", "mean_pool"],
        default=["full", "no_rmc", "no_cnn", "mean_pool"],
        help="Variants to train. Default: all 4.",
    )
    parser.add_argument(
        "--table-only", action="store_true",
        help="Skip training, load existing checkpoints and generate table.",
    )
    parser.add_argument(
        "--data-dir", default="data/processed",
        help="Path to processed data directory.",
    )
    parser.add_argument(
        "--out-dir", default="results/ablation",
        help="Output directory for checkpoints and tables.",
    )
    args = parser.parse_args()

    cfg = {**BASE_CONFIG, "data_dir": args.data_dir}
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Variants to run: {args.variants}")

    # Load data một lần dùng chung
    print("\n[Loading data]")
    train_loader, val_loader, test_loader = load_data(cfg["data_dir"], cfg["batch_size"])

    # Class weights
    class_weights = None
    if cfg["use_class_weights"]:
        cw_path = Path(cfg["data_dir"]) / "class_weights.npy"
        if cw_path.exists():
            class_weights = torch.from_numpy(
                np.load(cw_path)
            ).float().to(device)
            print(f"Class weights: {class_weights.cpu().numpy()}")

    json_path = out_dir / "ablation_results.json"
    all_results = []
    
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"[OK] Loaded {len(all_results)} existing results from {json_path}")
        except Exception as e:
            print(f"[WARNING] Could not load existing results: {e}")

    if args.table_only:
        if not all_results:
            print(f"[ERROR] {json_path} not found or empty. Run without --table-only first.")
            return
    else:
        set_seed(cfg["seed"])

        for variant_name in args.variants:
            if variant_name not in VARIANTS:
                print(f"[SKIP] Unknown variant: {variant_name}")
                continue

            variant_flags = VARIANTS[variant_name]
            result = train_variant(
                variant_name=variant_name,
                variant_flags=variant_flags,
                cfg=cfg,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_weights=class_weights,
                out_dir=ckpt_dir,
            )
            all_results.append(result)

            # Lưu sau mỗi variant phòng crash
            json_path = out_dir / "ablation_results.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n[OK] Saved intermediate results -> {json_path}")

    if all_results:
        generate_table(all_results, out_dir)


if __name__ == "__main__":
    main()
