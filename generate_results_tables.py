"""
generate_results_tables.py
--------------------------
PRIORITY 1: Auto-generate final CSV + LaTeX tables
PRIORITY 2: Combined trustworthy_ai_summary.png
PRIORITY 3: Organised output under results/tables/ and results/figures/
"""

import json
import csv
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path("results")
TABLES_DIR  = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load source data dynamically ────────────────────────────────────────────

CLEAN = {}
if (ROOT / "final_results.csv").exists():
    df_clean = pd.read_csv(ROOT / "final_results.csv")
    for _, row in df_clean.iterrows():
        # Columns: Model,Accuracy,Precision_macro,Recall_macro,F1_macro,F1_weighted,AUC_OvR
        CLEAN[row["Model"]] = dict(
            acc=row["Accuracy"], prec=row["Precision_macro"], rec=row["Recall_macro"],
            f1=row["F1_macro"], f1w=row["F1_weighted"], auc=row["AUC_OvR"]
        )

FGSM = {}
if (FIGURES_DIR / "fgsm_baseline_summary.csv").exists():
    df_fgsm = pd.read_csv(FIGURES_DIR / "fgsm_baseline_summary.csv")
    for _, row in df_fgsm.iterrows():
        # Columns: model,clean_f1,fgsm_f1,asr,recall_clean_S,recall_adv_S,recall_clean_V,recall_adv_V,recall_clean_F,recall_adv_F
        FGSM[row["model"]] = dict(
            clean_f1=row["clean_f1"], adv_f1=row["fgsm_f1"], asr=row["asr"],
            rec_s_c=row["recall_clean_S"], rec_s_a=row["recall_adv_S"],
            rec_v_c=row["recall_clean_V"], rec_v_a=row["recall_adv_V"],
            rec_f_c=row["recall_clean_F"], rec_f_a=row["recall_adv_F"]
        )

CALIB = {}
if (ROOT / "calibration_results.csv").exists():
    df_calib = pd.read_csv(ROOT / "calibration_results.csv")
    for _, row in df_calib.iterrows():
        CALIB[row["Model"]] = dict(ece=row["ECE (lower=better)"], brier=row["Brier Score (lower=better)"])

PGD = {}
if (FIGURES_DIR / "pgd_baseline_comparison.csv").exists():
    df_pgd = pd.read_csv(FIGURES_DIR / "pgd_baseline_comparison.csv")
    for _, row in df_pgd.iterrows():
        # Columns: model,clean_f1,pgd_f1_002,pgd_f1_005,f1_drop_002,asr_002,asr_005
        PGD[row["model"]] = dict(
            clean_f1=row["clean_f1"], adv_f1=row.get("pgd_f1_002", 0),
            asr=row.get("asr_002", 0), adv_f1_005=row.get("pgd_f1_005", 0), 
            asr_005=row.get("asr_005", 0)
        )

# ── helpers ────────────────────────────────────────────────────────────────────
BOLD = lambda x: f"\\textbf{{{x}}}"

def fmt(v, decimals=4): return f"{v:.{decimals}f}"

def best_idx(vals, lower_is_better=False):
    return vals.index(min(vals)) if lower_is_better else vals.index(max(vals))


# ==============================================================================
# PRIORITY 1 – Tables
# ==============================================================================

def write_csv_and_latex(rows, header, stem, caption, label):
    """Write both CSV and LaTeX tabular for a given table."""
    csv_path = TABLES_DIR / f"{stem}.csv"
    tex_path = TABLES_DIR / f"{stem}.tex"

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"[CSV] {csv_path}")

    # LaTeX
    ncols = len(header)
    col_fmt = "l" + "r" * (ncols - 1)
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[TEX] {tex_path}")


# ── Table 1: Clean Performance ─────────────────────────────────────────────────
def table_clean():
    header = ["Model", "Accuracy", "Precision", "Recall", "F1-macro", "F1-weighted", "AUC-OvR"]
    models = list(CLEAN.keys())
    cols = {
        "acc": [CLEAN[m]["acc"] for m in models],
        "prec":[CLEAN[m]["prec"] for m in models],
        "rec": [CLEAN[m]["rec"] for m in models],
        "f1":  [CLEAN[m]["f1"]  for m in models],
        "f1w": [CLEAN[m]["f1w"] for m in models],
        "auc": [CLEAN[m]["auc"] for m in models],
    }
    rows = []
    for i, m in enumerate(models):
        def cell(key, v): return BOLD(fmt(v)) if i == best_idx(cols[key]) else fmt(v)
        rows.append([
            m,
            cell("acc",  CLEAN[m]["acc"]),
            cell("prec", CLEAN[m]["prec"]),
            cell("rec",  CLEAN[m]["rec"]),
            cell("f1",   CLEAN[m]["f1"]),
            cell("f1w",  CLEAN[m]["f1w"]),
            cell("auc",  CLEAN[m]["auc"]),
        ])
    write_csv_and_latex(rows, header, "table1_clean_performance",
        caption="Clean test-set performance on MIT-BIH Arrhythmia dataset.",
        label="clean_perf")


# ── Table 2: FGSM Robustness ───────────────────────────────────────────────────
def table_fgsm():
    header = ["Model", "F1 (clean)", "F1 (adv)", "F1 drop", "ASR",
              "Rec-S clean", "Rec-S adv", "Rec-V clean", "Rec-V adv"]
    models = list(FGSM.keys())
    adv_f1s = [FGSM[m]["adv_f1"] for m in models]
    rows = []
    for i, m in enumerate(models):
        d = FGSM[m]
        drop = d["clean_f1"] - d["adv_f1"]
        best_adv = i == best_idx(adv_f1s)
        rows.append([
            m,
            fmt(d["clean_f1"]),
            BOLD(fmt(d["adv_f1"])) if best_adv else fmt(d["adv_f1"]),
            fmt(drop),
            fmt(d["asr"], 4),
            fmt(d["rec_s_c"]), fmt(d["rec_s_a"]),
            fmt(d["rec_v_c"]), fmt(d["rec_v_a"]),
        ])
    write_csv_and_latex(rows, header, "table2_fgsm_robustness",
        caption="FGSM adversarial robustness (epsilon=0.02). ASR = Attack Success Rate.",
        label="fgsm_robust")


# ── Table 3: Calibration ───────────────────────────────────────────────────────
def table_calibration():
    header = ["Model", "ECE (↓)", "Brier Score (↓)", "ECE Grade", "Brier Grade"]
    models = list(CALIB.keys())

    def grade_ece(v):
        if v < 0.02: return "Excellent"
        if v < 0.05: return "Good"
        if v < 0.10: return "Fair"
        return "Poor"

    def grade_brier(v):
        if v < 0.05: return "Excellent"
        if v < 0.10: return "Good"
        if v < 0.20: return "Fair"
        return "Poor"

    eces   = [CALIB[m]["ece"]   for m in models]
    briers = [CALIB[m]["brier"] for m in models]
    rows = []
    for i, m in enumerate(models):
        d = CALIB[m]
        best_e = i == best_idx(eces, lower_is_better=True)
        best_b = i == best_idx(briers, lower_is_better=True)
        rows.append([
            m,
            BOLD(fmt(d["ece"])) if best_e else fmt(d["ece"]),
            BOLD(fmt(d["brier"])) if best_b else fmt(d["brier"]),
            grade_ece(d["ece"]),
            grade_brier(d["brier"]),
        ])
    write_csv_and_latex(rows, header, "table3_calibration",
        caption="Calibration quality on MIT-BIH test set. Bold = best per column.",
        label="calibration")


# ── Table 4: PGD Robustness ────────────────────────────────────────────────────
def table_pgd():
    header = ["Model", "F1 (clean)", "F1-PGD (0.02)", "F1-PGD (0.05)",
              "F1 drop (0.02)", "ASR (0.02)", "ASR (0.05)"]
    models = list(PGD.keys())
    adv_f1s = [PGD[m]["adv_f1"] for m in models]
    rows = []
    for i, m in enumerate(models):
        d = PGD[m]
        drop = d["clean_f1"] - d["adv_f1"]
        best_adv = i == best_idx(adv_f1s)
        rows.append([
            m,
            fmt(d["clean_f1"]),
            BOLD(fmt(d["adv_f1"])) if best_adv else fmt(d["adv_f1"]),
            fmt(d["adv_f1_005"]),
            fmt(drop),
            fmt(d["asr"], 4),
            fmt(d["asr_005"], 4),
        ])
    write_csv_and_latex(rows, header, "table4_pgd_robustness",
        caption="PGD adversarial robustness (20 steps, alpha=0.005). Bold = best adversarial F1.",
        label="pgd_robust")


# ── Table 5: Consolidated Summary (LSTM/BiLSTM/HMR only) ───────────────────────
def table_consolidated():
    header = ["Model", "Accuracy", "F1-macro", "AUC",
              "FGSM-F1", "FGSM-drop", "PGD-F1", "PGD-drop",
              "ECE", "Brier"]
    deep_models = ["LSTM", "BiLSTM", "HMR-BiLSTM"]

    # Ensure all data sources are populated before accessing dict
    # Nếu bất kỳ CSV nào chưa generate → KeyError crash. Thay bằng graceful skip.
    missing = [
        m for m in deep_models
        if m not in CLEAN or m not in FGSM or m not in PGD or m not in CALIB
    ]
    if missing:
        missing_src = []
        if not CLEAN: missing_src.append("final_results.csv")
        if not FGSM:  missing_src.append("fgsm_baseline_summary.csv")
        if not PGD:   missing_src.append("pgd_baseline_comparison.csv")
        if not CALIB: missing_src.append("calibration_results.csv")
        print(f"[SKIP] table_consolidated: missing data for models {missing}. "
              f"Run these scripts first: {missing_src or ['check model name mismatch']}")
        return

    rows = []
    for m in deep_models:
        c = CLEAN[m]
        f = FGSM[m]
        p = PGD[m]
        k = CALIB[m]
        rows.append([
            m,
            fmt(c["acc"]),
            fmt(c["f1"]),
            fmt(c["auc"]),
            fmt(f["adv_f1"]),
            fmt(f["clean_f1"] - f["adv_f1"]),
            fmt(p["adv_f1"]),
            fmt(p["clean_f1"] - p["adv_f1"]),
            fmt(k["ece"]),
            fmt(k["brier"]),
        ])
    write_csv_and_latex(rows, header, "table5_consolidated",
        caption="Consolidated trustworthy AI evaluation: clean performance, FGSM/PGD adversarial robustness, and calibration.",
        label="consolidated")


# ==============================================================================
# PRIORITY 2 – Combined Summary Figure
# ==============================================================================

COLORS = {"LSTM": "#1f77b4", "BiLSTM": "#ff7f0e", "HMR-BiLSTM": "#2ca02c"}
MODELS = ["LSTM", "BiLSTM", "HMR-BiLSTM"]
X = np.arange(len(MODELS))
W = 0.55


def make_summary_figure():
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("#0f1117")

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                           left=0.06, right=0.97, top=0.84, bottom=0.16)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── common style ───────────────────────────────────────────────────────────
    def style(ax, title, ylabel):
        ax.set_facecolor("#1a1d27")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, color="#cccccc", fontsize=10)
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        for sp in ax.spines.values(): sp.set_color("#333344")
        ax.yaxis.grid(True, color="#2a2d3a", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(X)
        ax.set_xticklabels(MODELS, color="#cccccc", fontsize=9)

    # ── Panel A: Clean Performance ─────────────────────────────────────────────
    f1_clean  = [CLEAN[m]["f1"]  for m in MODELS]
    auc_clean = [CLEAN[m]["auc"] for m in MODELS]
    bars = ax1.bar(X - W/4, f1_clean,  width=W/2, label="F1-macro", zorder=3,
                   color=[COLORS[m] for m in MODELS], alpha=0.9)
    bars2= ax1.bar(X + W/4, auc_clean, width=W/2, label="AUC-OvR", zorder=3,
                   color=[COLORS[m] for m in MODELS], alpha=0.55, hatch="//")
    for bar, v in zip(bars,  f1_clean):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.003, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7.5, color="white", fontweight="bold")
    for bar, v in zip(bars2, auc_clean):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.003, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7.5, color="#dddddd")
    ax1.set_ylim(0.70, 1.00)
    style(ax1, "A  Clean Performance", "Score")
    ax1.legend(fontsize=8, framealpha=0.2, labelcolor="white")

    # ── Panel B: FGSM vs PGD Robustness ────────────────────────────────────────
    from matplotlib.patches import Patch
    fgsm_f1 = [FGSM[m]["adv_f1"] for m in MODELS]
    pgd_f1  = [PGD[m]["adv_f1"]  for m in MODELS]
    bw = W / 2.2
    bars_fgsm = ax2.bar(X - bw/2, fgsm_f1, width=bw, zorder=3,
                        color=[COLORS[m] for m in MODELS], alpha=0.9)
    bars_pgd  = ax2.bar(X + bw/2, pgd_f1,  width=bw, zorder=3,
                        color=[COLORS[m] for m in MODELS], alpha=0.55, hatch="//")
    for bar, v in zip(bars_fgsm, fgsm_f1):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.003, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7, color="white", fontweight="bold")
    for bar, v in zip(bars_pgd, pgd_f1):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.003, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7, color="#dddddd")
    ax2.set_ylim(0.65, 1.00)
    style(ax2, "B  Adversarial Robustness (eps=0.02)", "F1-macro (adversarial)")
    ax2.legend(handles=[
        Patch(color="gray", alpha=0.9, label="FGSM"),
        Patch(color="gray", alpha=0.55, hatch="//", label="PGD-20"),
    ], fontsize=8, framealpha=0.2, labelcolor="white")

    # ── Panel C: Calibration ───────────────────────────────────────────────────
    eces   = [CALIB[m]["ece"]   for m in MODELS]
    briers = [CALIB[m]["brier"] for m in MODELS]
    bars4 = ax3.bar(X - W/4, eces,   width=W/2, label="ECE",         zorder=3,
                    color=[COLORS[m] for m in MODELS], alpha=0.9)
    bars5 = ax3.bar(X + W/4, briers, width=W/2, label="Brier Score", zorder=3,
                    color=[COLORS[m] for m in MODELS], alpha=0.55, hatch="//")
    for bar, v in zip(bars4, eces):
        ax3.text(bar.get_x()+bar.get_width()/2, v+0.001, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=7.5, color="white", fontweight="bold")
    for bar, v in zip(bars5, briers):
        ax3.text(bar.get_x()+bar.get_width()/2, v+0.001, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=7.5, color="#dddddd")
    ax3.set_ylim(0.0, 0.12)
    style(ax3, "C  Calibration (lower = better)", "Score")
    ax3.legend(fontsize=8, framealpha=0.2, labelcolor="white")

    # ── Super-title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "Trustworthy AI Evaluation: Clean Performance | Adversarial Robustness | Calibration",
        color="white", fontsize=14, fontweight="bold", y=0.97
    )

    out = FIGURES_DIR / "trustworthy_ai_summary.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[FIG] {out}")


# ==============================================================================
# PRIORITY 3 – Polish output folder & print index
# ==============================================================================

def print_output_index():
    print("\n" + "=" * 60)
    print(" OUTPUT INDEX")
    print("=" * 60)
    for p in sorted((ROOT).rglob("*")):
        if p.is_file():
            rel = p.relative_to(ROOT)
            size_kb = p.stat().st_size / 1024
            print(f"  {str(rel):<55} {size_kb:>6.1f} KB")
    print("=" * 60)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n[PRIORITY 1] Generating tables...")
    table_clean()
    table_fgsm()
    table_pgd()
    table_calibration()
    table_consolidated()

    print("\n[PRIORITY 2] Generating summary figure...")
    make_summary_figure()

    print("\n[PRIORITY 3] Output folder index:")
    print_output_index()

    print("\nDone. Next step: paper narrative / threat model / discussion.")
