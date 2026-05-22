import csv
import json
from pathlib import Path
from train import RLSTMLoss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from report_results import load_rlstm_model
from run_baselines import LSTMBaseline

from evaluate_fgsm import evaluate_fgsm, build_test_loader

MODEL_SPECS = {
    "LSTM": Path("results/checkpoints/best_lstm.pt"),
    "BiLSTM": Path("results/checkpoints/best_bilstm.pt"),
    "HMR-BiLSTM": Path("results/checkpoints/best_rlstm.pt"),
}

DEFAULT_EPSILONS = [0.0, 0.02, 0.05]


def load_baseline_model(name, checkpoint_path: Path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if name == "HMR-BiLSTM":
        model, _ = load_rlstm_model(str(checkpoint_path), device)
        return model

    # run_baselines.py saves raw state_dict via torch.save(best_state, path)
    # so checkpoint is never wrapped in a dict — handle both formats defensively.
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    # Read input_size dynamically from checkpoint state_dict (conv1 weight)
    input_size = state_dict["cnn.0.weight"].shape[1] if "cnn.0.weight" in state_dict else 1

    model = LSTMBaseline(
        input_size=input_size,
        hidden_size=96,
        bidirectional=(name == "BiLSTM"),
        dropout=0.25,
        num_classes=5,
    ).to(device)
    # strict=True: any key mismatch raises RuntimeError immediately,
    # preventing silent loading of mismatched / random weights.
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def save_baseline_comparison(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path("results/logs/fgsm_baseline_comparison.json"); json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fgsm_baseline_comparison.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    header = [
        "model", "epsilon", "accuracy", "macro_f1", "macro_recall",
        "attack_success_rate", "avg_confidence", "orig_avg_confidence",
        "confidence_drop",
        # per-class recall clean and under attack for S, V, F
        "recall_clean_S", "recall_adv_S",
        "recall_clean_V", "recall_adv_V",
        "recall_clean_F", "recall_adv_F",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in results:
            f.write(",".join(str(row.get(h, "")) for h in header) + "\n")

    print(f"Saved baseline comparison JSON: {json_path}")
    print(f"Saved baseline comparison CSV: {csv_path}")
    return json_path, csv_path


def plot_baseline_f1_vs_epsilon(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in sorted({row["model"] for row in results}):
        eps = [row["epsilon"] for row in results if row["model"] == model_name]
        f1 = [row["macro_f1"] for row in results if row["model"] == model_name]
        ax.plot(eps, f1, marker="o", label=model_name, linewidth=2)

    ax.set_xlabel("Epsilon", fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("FGSM Macro F1 Comparison", fontsize=13)
    ax.set_xticks(DEFAULT_EPSILONS)
    ax.set_ylim([0, 1.0])
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    path = output_dir / "fgsm_baseline_f1_vs_epsilon.png"
    plt.savefig(path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved baseline F1 comparison figure: {path}")
    return path


def plot_per_class_recall_bars(results, output_dir: Path, attack_epsilon=0.02):
    """Grouped bar chart: per-class recall (S, V, F) clean vs adversarial
    at *attack_epsilon* for every model evaluated.

    Produces two sub-plots (one per epsilon context) in a single 1×3 layout
    — one bar-group per model, pairs of bars (clean / adversarial) per class.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = sorted({row["model"] for row in results})
    classes = [("S", "Supraventricular"), ("V", "Ventricular"), ("F", "Fusion")]
    # Colours: clean = solid, adv = hatched lighter shade
    PALETTE = {
        "LSTM":       ("#1f77b4", "#aec7e8"),
        "BiLSTM":     ("#ff7f0e", "#ffbb78"),
        "HMR-BiLSTM": ("#2ca02c", "#98df8a"),
    }

    n_classes = len(classes)
    n_models  = len(model_names)
    bar_width = 0.18
    # Each class gets a group; within a group: 2 bars per model (clean, adv)
    group_width = n_models * 2 * bar_width + 0.15   # gap between class groups

    fig, ax = plt.subplots(figsize=(11, 5))

    for m_idx, model_name in enumerate(model_names):
        # Find the row at attack_epsilon for this model
        row = next(
            (r for r in results
             if r["model"] == model_name and abs(r["epsilon"] - attack_epsilon) < 1e-6),
            None,
        )
        if row is None:
            continue

        solid_color, light_color = PALETTE.get(model_name, ("#888888", "#cccccc"))

        for c_idx, (cls, _) in enumerate(classes):
            # centre of this class group
            group_centre = c_idx * group_width
            # offset within the group for this model
            offset = (m_idx - (n_models - 1) / 2) * 2 * bar_width

            clean_val = row.get(f"recall_clean_{cls}", 0.0)
            adv_val   = row.get(f"recall_adv_{cls}",   0.0)

            x_clean = group_centre + offset - bar_width / 2
            x_adv   = group_centre + offset + bar_width / 2

            label_clean = f"{model_name} (clean)" if c_idx == 0 else None
            label_adv   = f"{model_name} (adv \u03b5={attack_epsilon})" if c_idx == 0 else None

            ax.bar(x_clean, clean_val, width=bar_width,
                   color=solid_color, label=label_clean, zorder=3)
            ax.bar(x_adv,   adv_val,   width=bar_width,
                   color=light_color, hatch="//", edgecolor=solid_color,
                   linewidth=0.8, label=label_adv, zorder=3)

            # Annotate delta
            delta = adv_val - clean_val
            ax.annotate(
                f"{delta:+.2f}",
                xy=(x_adv, adv_val),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5,
                color=solid_color,
            )

    # x-tick labels centred on each class group
    group_centres = [i * group_width for i in range(n_classes)]
    ax.set_xticks(group_centres)
    ax.set_xticklabels(
        [f"Class {cls}\n({full})" for cls, full in classes],
        fontsize=11,
    )
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title(
        f"Per-Class Recall Under FGSM Attack (\u03b5={attack_epsilon})\n"
        "Solid = Clean  |  Hatched = Adversarial",
        fontsize=12,
    )
    ax.set_ylim([0, 1.08])
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    path = output_dir / f"fgsm_per_class_recall_eps{attack_epsilon:.2f}.png"
    plt.savefig(path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved per-class recall bar chart: {path}")
    return path


def build_summary_table(results, summary_epsilon=0.02):
    table = []
    for model_name in sorted({row["model"] for row in results}):
        clean_row = next((r for r in results if r["model"] == model_name and r["epsilon"] == 0.0), None)
        attack_row = next((r for r in results if r["model"] == model_name and r["epsilon"] == summary_epsilon), None)
        if clean_row is None or attack_row is None:
            continue

        entry = {
            "model": model_name,
            "clean_f1": float(clean_row["macro_f1"]),
            "fgsm_f1": float(attack_row["macro_f1"]),
            "asr": float(attack_row["attack_success_rate"]),
        }
        # Per-class recall delta (clean → adv) for S, V, F
        for cls in ("S", "V", "F"):
            clean_key = f"recall_clean_{cls}"
            adv_key = f"recall_adv_{cls}"
            entry[f"recall_clean_{cls}"] = float(attack_row.get(clean_key, 0.0))
            entry[f"recall_adv_{cls}"] = float(attack_row.get(adv_key, 0.0))
        table.append(entry)
    return table


def save_summary_table(table, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fgsm_baseline_summary.csv"
    header = [
        "model", "clean_f1", "fgsm_f1", "asr",
        "recall_clean_S", "recall_adv_S",
        "recall_clean_V", "recall_adv_V",
        "recall_clean_F", "recall_adv_F",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in table:
            f.write(",".join(str(row.get(h, "")) for h in header) + "\n")
    print(f"Saved baseline summary table: {csv_path}")
    return csv_path


def main():
    # Fix seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build criterion matching train config (FocalLoss + class weights)
    cw_path = Path("data/processed/class_weights.npy")
    if cw_path.exists():
        class_weights = torch.from_numpy(np.load(cw_path)).float().to(device)
    else:
        class_weights = None
    criterion = RLSTMLoss(
        lambda_smooth=0.003,
        class_weights=class_weights,
        use_focal=True,
        focal_gamma=1.5,
    )
    print(f"Criterion: FocalLoss(gamma=1.5), class_weights={'loaded' if class_weights is not None else 'None'}")

    test_loader = build_test_loader(batch_size=128)
    output_dir = Path("results/figures")
    comparison_results = []

    for model_name, checkpoint_path in MODEL_SPECS.items():
        print(f"\nLoading model {model_name} from {checkpoint_path}")
        if not checkpoint_path.exists():
            print(f"  Skipping {model_name}: checkpoint not found.")
            continue

        try:
            model = load_baseline_model(model_name, checkpoint_path, device)
        except Exception as exc:
            print(f"  Failed to load {model_name}: {exc}")
            continue

        for epsilon in DEFAULT_EPSILONS:
            print(f"  Evaluating {model_name} @ epsilon={epsilon}")
            result = evaluate_fgsm(model, test_loader, device, criterion, epsilon=epsilon)
            comparison_results.append({
                "model": model_name,
                **result,
            })

    if not comparison_results:
        raise SystemExit("No models were evaluated. Check that the checkpoint files exist.")

    save_baseline_comparison(comparison_results, output_dir)
    plot_baseline_f1_vs_epsilon(comparison_results, output_dir)
    # Per-class recall bar charts at the two main attack epsilons
    for eps in (0.02, 0.05):
        if any(abs(r["epsilon"] - eps) < 1e-6 for r in comparison_results):
            plot_per_class_recall_bars(comparison_results, output_dir, attack_epsilon=eps)

    summary_table = build_summary_table(comparison_results, summary_epsilon=0.02)
    if summary_table:
        save_summary_table(summary_table, output_dir)
        print("\nBaseline comparison table (epsilon=0.02):")
        header_line = f"{'Model':<12} {'Clean F1':>8} {'FGSM F1':>8} {'ASR':>7} | {'Rec_S clean':>11} {'Rec_S adv':>9} | {'Rec_V clean':>11} {'Rec_V adv':>9} | {'Rec_F clean':>11} {'Rec_F adv':>9}"
        print(header_line)
        print("-" * len(header_line))
        for row in summary_table:
            print(
                f"{row['model']:<12} {row['clean_f1']:>8.4f} {row['fgsm_f1']:>8.4f} {row['asr']:>7.4f}"
                f" | {row.get('recall_clean_S', 0):>11.4f} {row.get('recall_adv_S', 0):>9.4f}"
                f" | {row.get('recall_clean_V', 0):>11.4f} {row.get('recall_adv_V', 0):>9.4f}"
                f" | {row.get('recall_clean_F', 0):>11.4f} {row.get('recall_adv_F', 0):>9.4f}"
            )
    else:
        print("No summary table could be built because required epsilons were missing.")

    # ── Clean up stale old-format output files ────────────────────────────────
    stale = [
        output_dir / "fgsm_comparison_results.json",   # old format with linf_norm_mean
        output_dir / "fgsm_comparison_table.csv",       # generated by old compare path
        output_dir / "fgsm_comparison_macro_f1.png",    # generated by old compare path
        output_dir / "fgsm_results.json",               # single-model old run
        output_dir / "fgsm_results.csv",                # single-model old run
    ]
    for p in stale:
        try:
            if p.exists():
                p.unlink()
                print(f"Removed stale file: {p.name}")
        except Exception as exc:
            print(f"Could not remove {p.name}: {exc}")


if __name__ == "__main__":
    main()
