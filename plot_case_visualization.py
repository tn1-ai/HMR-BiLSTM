"""
plot_case_visualization.py
===========================
Generate case-study figures for explainability.

Each figure shows:
  - Top panel:    Raw ECG waveform with P / QRS / T region shading
  - Middle panel: Residual gate r_t trajectory (forward + backward averaged)
  - Bottom panel: Attention pooling weights

Produces:
  results/figures/case_normal.png
  results/figures/case_ventricular.png
  results/figures/case_fusion.png
  results/figures/case_supraventricular.png
  results/figures/case_comparison_N_vs_V.png  (side-by-side)

Usage:
    python plot_case_visualization.py

Runtime: < 10 seconds (inference only, no training)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.interpolate import interp1d

from report_results import load_hmr_bilstm as load_rlstm_model
# ─── Config ──────────────────────────────────────────────────────────────
CHECKPOINT = "results/checkpoints/best_rlstm.pt"
TEST_DATA  = "data/processed/test.npz"
OUTPUT_DIR = Path("results/figures")
SAMPLING_RATE = 360  # Hz, MIT-BIH

CLASS_NAMES = {0: "N (Normal)", 1: "S (Supraventricular)",
               2: "V (Ventricular)", 3: "F (Fusion)", 4: "Q (Unknown)"}
CLASS_SHORT = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

# Colors for cardiac regions
COLORS = {
    "P":   ("#3498db", 0.15),   # blue
    "QRS": ("#e74c3c", 0.20),   # red
    "T":   ("#2ecc71", 0.15),   # green
}


# ─── Utilities ───────────────────────────────────────────────────────────
def detect_cardiac_regions(ecg_signal):
    """
    Estimate P / QRS / T regions for a single MIT-BIH heartbeat segment.
    MIT-BIH segments are 187 samples, pre-segmented and roughly centered
    on the R-peak. We detect R-peak as the max absolute value, then
    estimate regions based on typical cardiac timing at 360 Hz.
    """
    T = len(ecg_signal)
    signal = ecg_signal.flatten()

    # R-peak: largest absolute deflection
    r_peak = int(np.argmax(np.abs(signal)))

    # Timing at 360 Hz (approximate):
    #   P-wave:  60-120 ms before R  → 22-43 samples before R
    #   QRS:     40 ms before R to 60 ms after R → 14 before to 22 after
    #   T-wave:  80-200 ms after R   → 29-72 samples after R

    qrs_start = max(0, r_peak - 14)
    qrs_end   = min(T - 1, r_peak + 22)
    p_start   = max(0, r_peak - 50)
    p_end     = max(0, r_peak - 16)
    t_start   = min(T - 1, r_peak + 28)
    t_end     = min(T - 1, r_peak + 75)

    return {
        "P":     (p_start, p_end),
        "QRS":   (qrs_start, qrs_end),
        "T":     (t_start, t_end),
        "R_peak": r_peak,
    }


def upsample_gate(gate_seq, target_len=187):
    """
    Upsample gate trajectory from CNN-compressed length (46) to
    original ECG length (187) using linear interpolation.
    """
    current_len = len(gate_seq)
    if current_len == target_len:
        return gate_seq
    x_old = np.linspace(0, 1, current_len)
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, gate_seq, kind="linear")
    return f(x_new)


@torch.no_grad()
def get_single_sample_internals(model, x_single, device):
    """
    Run one ECG sample through model and return all internals.
    x_single: numpy array (187, 1)
    Returns: prediction, probability, r_t (46 steps), attention_weights (46 steps)
    """
    x_tensor = torch.from_numpy(x_single).float().unsqueeze(0).to(device)
    logits, internals = model(x_tensor, return_internals=True)

    pred = int(logits.argmax(dim=-1).cpu().item())
    prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # Gate: average forward + backward, then average across hidden dimensions
    r_fwd = internals["r_fwd"].cpu().numpy()[0]   # (T', H)
    r_bwd = internals["r_bwd"].cpu().numpy()[0]   # (T', H)
    r_combined = ((r_fwd + r_bwd) / 2).mean(axis=-1)  # (T',)

    # Attention weights
    attn = internals["attention_weights"].cpu().numpy()[0]  # (T',)

    return pred, prob, r_combined, attn


def find_good_samples(X_test, y_test, model, device, target_classes=[0, 1, 2, 3]):
    """
    Find correctly-classified samples with high confidence for each class.
    Returns dict: class_id -> (sample_index, x, pred, prob, r_t, attn)
    """
    samples = {}
    for cls in target_classes:
        # Get all indices for this class
        indices = np.where(y_test == cls)[0]
        if len(indices) == 0:
            print(f"  [WARN] No samples for class {cls}")
            continue

        # Find correctly classified sample with highest confidence
        best_idx, best_conf = None, 0.0
        for idx in indices[:200]:  # check first 200 to save time
            x = X_test[idx]
            pred, prob, r_t, attn = get_single_sample_internals(model, x, device)
            if pred == cls and prob[cls] > best_conf:
                best_conf = prob[cls]
                best_idx = idx
                best_data = (x, pred, prob, r_t, attn)

        if best_idx is not None:
            samples[cls] = (best_idx, *best_data)
            print(f"  Class {CLASS_SHORT[cls]}: sample #{best_idx}, "
                  f"conf={best_conf:.4f}")
        else:
            print(f"  [WARN] No correctly classified sample for class {cls}")

    return samples


# ─── Plotting ────────────────────────────────────────────────────────────
def plot_single_case(ecg_signal, r_t, attn, true_class, pred_class, prob,
                     save_path, title_suffix=""):
    """
    3-panel figure for one ECG sample:
      Panel A: ECG waveform + P/QRS/T shading
      Panel B: Residual gate r_t trajectory
      Panel C: Attention pooling weights
    """
    ecg = ecg_signal.flatten()
    T_ecg = len(ecg)
    regions = detect_cardiac_regions(ecg)

    # Upsample gate and attention to match ECG length
    r_up = upsample_gate(r_t, T_ecg)
    attn_up = upsample_gate(attn, T_ecg)

    # Time axis in milliseconds
    t_ms = np.arange(T_ecg) / SAMPLING_RATE * 1000

    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                              gridspec_kw={"height_ratios": [3, 2, 1.2]},
                              sharex=True)
    fig.subplots_adjust(hspace=0.08)

    class_label = CLASS_NAMES[true_class]
    pred_label = CLASS_SHORT[pred_class]
    conf = prob[pred_class]

    # ── Panel A: ECG Waveform ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_ms, ecg, color="#1a1a2e", linewidth=1.2, zorder=3)

    # Shade cardiac regions
    for region_name, val in regions.items():
        if region_name == "R_peak":
            continue
        start, end = val
        color, alpha = COLORS[region_name]
        ax.axvspan(t_ms[start], t_ms[end], alpha=alpha, color=color,
                   label=f"{region_name}-wave", zorder=1)

    # Mark R-peak
    r_peak = regions["R_peak"]
    ax.axvline(t_ms[r_peak], color="#e74c3c", linestyle="--", alpha=0.5,
               linewidth=1, zorder=2)
    ax.annotate("R", xy=(t_ms[r_peak], ecg[r_peak]),
                xytext=(t_ms[r_peak] + 8, ecg[r_peak] + 0.3),
                fontsize=10, color="#e74c3c", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))

    ax.set_ylabel("Amplitude (normalized)", fontsize=11)
    ax.set_title(f"Case Study: {class_label} beat  |  "
                 f"Predicted: {pred_label} (conf={conf:.3f}){title_suffix}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(alpha=0.2, linestyle="--")

    # ── Panel B: Gate Trajectory ─────────────────────────────────────────
    ax = axes[1]
    ax.plot(t_ms, r_up, color="#8e44ad", linewidth=1.8, zorder=3)
    ax.fill_between(t_ms, 0, r_up, alpha=0.15, color="#8e44ad", zorder=2)

    # Shade same cardiac regions for alignment
    for region_name, val in regions.items():
        if region_name == "R_peak":
            continue
        start, end = val
        color, alpha = COLORS[region_name]
        ax.axvspan(t_ms[start], t_ms[end], alpha=alpha * 0.7, color=color,
                   zorder=1)

    ax.set_ylabel("Residual Gate $r_t$", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2, linestyle="--")

    # Annotate peak gate region
    peak_idx = np.argmax(r_up)
    ax.annotate(f"peak={r_up[peak_idx]:.3f}",
                xy=(t_ms[peak_idx], r_up[peak_idx]),
                xytext=(t_ms[peak_idx] + 20, r_up[peak_idx] + 0.02),
                fontsize=9, color="#8e44ad",
                arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1))

    # ── Panel C: Attention Weights ───────────────────────────────────────
    ax = axes[2]
    ax.bar(t_ms, attn_up, width=t_ms[1] - t_ms[0], color="#e67e22",
           alpha=0.7, zorder=3)

    # Shade cardiac regions
    for region_name, val in regions.items():
        if region_name == "R_peak":
            continue
        start, end = val
        color, alpha = COLORS[region_name]
        ax.axvspan(t_ms[start], t_ms[end], alpha=alpha * 0.5, color=color,
                   zorder=1)

    ax.set_ylabel("Attention $\\gamma_t$", fontsize=11)
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2, linestyle="--")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_comparison(sample_a, sample_b, class_a, class_b, save_path):
    """
    Side-by-side comparison of two beats (e.g., Normal vs Ventricular).
    Each side has 2 panels: ECG waveform + Gate trajectory.
    """
    _, x_a, pred_a, prob_a, r_a, attn_a = sample_a
    _, x_b, pred_b, prob_b, r_b, attn_b = sample_b

    ecg_a = x_a.flatten()
    ecg_b = x_b.flatten()
    T_ecg = len(ecg_a)
    t_ms = np.arange(T_ecg) / SAMPLING_RATE * 1000

    r_a_up = upsample_gate(r_a, T_ecg)
    r_b_up = upsample_gate(r_b, T_ecg)

    regions_a = detect_cardiac_regions(ecg_a)
    regions_b = detect_cardiac_regions(ecg_b)

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.25,
                           height_ratios=[2, 1.5])

    titles = [
        f"{CLASS_NAMES[class_a]}  |  pred={CLASS_SHORT[pred_a]} "
        f"(conf={prob_a[pred_a]:.3f})",
        f"{CLASS_NAMES[class_b]}  |  pred={CLASS_SHORT[pred_b]} "
        f"(conf={prob_b[pred_b]:.3f})",
    ]
    ecgs = [ecg_a, ecg_b]
    gates = [r_a_up, r_b_up]
    regions_list = [regions_a, regions_b]

    for col, (ecg, gate, regions, title) in enumerate(
            zip(ecgs, gates, regions_list, titles)):

        # ── ECG ──────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t_ms, ecg, color="#1a1a2e", linewidth=1.2, zorder=3)
        for region_name, val in regions.items():
            if region_name == "R_peak":
                continue
            start, end = val
            color, alpha = COLORS[region_name]
            ax.axvspan(t_ms[start], t_ms[end], alpha=alpha, color=color,
                       label=f"{region_name}" if col == 0 else None, zorder=1)
        r_peak = regions["R_peak"]
        ax.axvline(t_ms[r_peak], color="#e74c3c", linestyle="--",
                   alpha=0.4, linewidth=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Amplitude" if col == 0 else "")
        ax.grid(alpha=0.2, linestyle="--")
        if col == 0:
            ax.legend(fontsize=8, loc="upper right")

        # ── Gate ─────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, col])
        ax.plot(t_ms, gate, color="#8e44ad", linewidth=1.8, zorder=3)
        ax.fill_between(t_ms, 0, gate, alpha=0.15, color="#8e44ad", zorder=2)
        for region_name, val in regions.items():
            if region_name == "R_peak":
                continue
            start, end = val
            color, alpha = COLORS[region_name]
            ax.axvspan(t_ms[start], t_ms[end], alpha=alpha * 0.6,
                       color=color, zorder=1)
        ax.set_ylabel("Gate $r_t$" if col == 0 else "")
        ax.set_xlabel("Time (ms)", fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2, linestyle="--")

        # Annotate gate peak
        peak_idx = np.argmax(gate)
        ax.annotate(f"peak={gate[peak_idx]:.3f}",
                    xy=(t_ms[peak_idx], gate[peak_idx]),
                    xytext=(t_ms[peak_idx] + 15, gate[peak_idx] + 0.015),
                    fontsize=8, color="#8e44ad",
                    arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1))

    fig.suptitle(
        f"Case Comparison: {CLASS_SHORT[class_a]} vs {CLASS_SHORT[class_b]} "
        f"— ECG Waveform and Residual Gate Trajectory",
        fontsize=14, fontweight="bold", y=1.01)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n[Loading model]")
    model, ckpt = load_rlstm_model(CHECKPOINT, device)
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Best epoch: {ckpt.get('epoch', '?')}")

    # Load test data
    print("\n[Loading test data]")
    test = np.load(TEST_DATA)
    X_test, y_test = test["X"], test["y"]
    print(f"  Test samples: {len(y_test)}")

    # Find good samples for each class
    print("\n[Finding representative samples]")
    samples = find_good_samples(
        X_test, y_test, model, device,
        target_classes=[0, 1, 2, 3]  # N, S, V, F
    )

    if not samples:
        print("[ERROR] No samples found. Check checkpoint and data.")
        return

    # ── Individual case figures ──────────────────────────────────────────
    print("\n[Generating individual case figures]")
    case_names = {0: "normal", 1: "supraventricular", 2: "ventricular", 3: "fusion"}

    for cls, data in samples.items():
        idx, x, pred, prob, r_t, attn = data
        save_path = OUTPUT_DIR / f"case_{case_names[cls]}.png"
        plot_single_case(x, r_t, attn, cls, pred, prob, save_path)

    # ── Comparison: Normal vs Ventricular ────────────────────────────────
    print("\n[Generating comparison figures]")
    if 0 in samples and 2 in samples:
        plot_comparison(
            samples[0], samples[2], 0, 2,
            OUTPUT_DIR / "case_comparison_N_vs_V.png"
        )

    # ── Comparison: Normal vs Supraventricular ───────────────────────────
    if 0 in samples and 1 in samples:
        plot_comparison(
            samples[0], samples[1], 0, 1,
            OUTPUT_DIR / "case_comparison_N_vs_S.png"
        )

    # ── Comparison: Normal vs Fusion ─────────────────────────────────────
    if 0 in samples and 3 in samples:
        plot_comparison(
            samples[0], samples[3], 0, 3,
            OUTPUT_DIR / "case_comparison_N_vs_F.png"
        )

    print("\n" + "=" * 60)
    print("  CASE VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Files:")
    for f in sorted(OUTPUT_DIR.glob("case_*.png")):
        print(f"    {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()