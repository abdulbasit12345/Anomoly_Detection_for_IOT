import os
import json
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

logger = logging.getLogger(__name__)

BG   = "#0F0F1A"
SURF = "#1A1A2E"
TEXT = "#E0E0FF"
PURP = "#6C63FF"
PINK = "#FF6584"
TEAL = "#43AA8B"
GOLD = "#F9C74F"
RED  = "#F94144"
COLS = [PURP, PINK, TEAL, GOLD, RED, "#A8DADC", "#FFB347"]

def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    SURF,
        "axes.edgecolor":    TEXT,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "text.color":        TEXT,
        "grid.color":        "#2A2A4A",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "legend.facecolor":  SURF,
        "legend.edgecolor":  PURP,
        "font.family":       "DejaVu Sans",
        "figure.dpi":        150,
    })

def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("    Saved: %s", path)

def plot_gan_losses(g_losses: list, d_losses: list, save_dir: str):
    """
    Plot GAN training loss curves with:
    - y=0 reference line (helps visualise convergence)
    - Rolling-average smoothing for long runs (100+ epochs)
    - WGAN-GP annotation box explaining negative loss semantics
    """
    _apply_dark_style()

    # Detect GAN type from loss sign
    has_neg = any(x < 0 for x in g_losses) or any(x < 0 for x in d_losses)
    is_wgan = has_neg

    # Smooth curves with a rolling average when there are many epochs
    def _smooth(vals, window=5):
        if len(vals) < window * 2:
            return vals
        import pandas as _pd
        return _pd.Series(vals).rolling(window, min_periods=1).mean().tolist()

    g_smooth = _smooth(g_losses)
    d_smooth = _smooth(d_losses)

    epochs = list(range(1, len(g_losses) + 1))

    fig, ax = plt.subplots(figsize=(11, 5))

    # Raw (faint) + smoothed (solid)
    ax.plot(epochs, g_losses, color=PURP, lw=0.8, alpha=0.3)
    ax.plot(epochs, d_losses, color=PINK, lw=0.8, alpha=0.3)
    ax.plot(epochs, g_smooth, color=PURP, lw=2.2, label="Generator Loss (smoothed)")
    ax.plot(epochs, d_smooth, color=PINK, lw=2.2, label="Discriminator / Critic Loss (smoothed)")

    ax.fill_between(epochs, g_smooth, alpha=0.12, color=PURP)
    ax.fill_between(epochs, d_smooth, alpha=0.12, color=PINK)

    # y = 0 reference line — crucial visual anchor
    ax.axhline(0, color=GOLD, lw=1.2, linestyle="--", alpha=0.7, label="y = 0 reference")

    title = "CTGAN (WGAN-GP) Critic Loss Curves" if is_wgan else "Simple GAN (BCE) Loss Curves"
    ax.set_title(title, fontsize=15, fontweight="bold", color=TEXT, pad=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Wasserstein Critic Score" if is_wgan else "BCE Loss", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)

    if is_wgan:
        # Annotate WGAN-GP semantics — this explains the "negative loss" concern
        note = (
            "WGAN-GP Explanation:\n"
            "  • Negative values are MATHEMATICALLY CORRECT for Wasserstein GANs.\n"
            "  • The critic outputs a real-valued score (not a probability 0-1).\n"
            "  • Discriminator (critic) loss = -(real_score - fake_score)  →  converges negative.\n"
            "  • Generator loss = -fake_score  →  decreases as the generator improves.\n"
            "  • Convergence is indicated by both curves becoming less volatile over epochs."
        )
        ax.text(
            0.01, 0.03, note,
            transform=ax.transAxes,
            fontsize=7.5,
            color=GOLD,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=SURF, edgecolor=GOLD, alpha=0.85),
        )
    else:
        note = (
            "Simple GAN (BCE) Explanation:\n"
            "  • Loss values should stay positive (binary cross-entropy is always ≥ 0).\n"
            "  • Healthy convergence: Generator loss ~0.7, Discriminator loss ~0.7 (Nash equilibrium).\n"
            "  • Generator improving = its loss decreasing toward 0."
        )
        ax.text(
            0.01, 0.03, note,
            transform=ax.transAxes,
            fontsize=7.5,
            color=TEAL,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=SURF, edgecolor=TEAL, alpha=0.85),
        )

    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "gan_loss_curves.png"))


def plot_training_history(history: dict, save_dir: str):
    _apply_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Classifier Training History", fontsize=16, fontweight="bold", color=TEXT, y=1.01)
    metrics_pairs = [
        ("train_loss",  "val_loss",  "Loss",     PURP, PINK),
        ("train_acc",   "val_acc",   "Accuracy", TEAL, GOLD),
        ("train_f1",    "val_f1",    "F1-Score", PURP, RED),
    ]
    for ax, (tr_key, vl_key, title, c1, c2) in zip(axes, metrics_pairs):
        epochs = range(1, len(history[tr_key]) + 1)
        ax.plot(epochs, history[tr_key], color=c1, lw=2, label="Train")
        ax.plot(epochs, history[vl_key], color=c2, lw=2, linestyle="--", label="Validation")
        ax.fill_between(epochs, history[tr_key], history[vl_key], alpha=0.1, color=c1)
        ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT)
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "classifier_training_history.png"))

def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_dir: str,
                           title: str = "Confusion Matrix"):
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    cm_pct = cm.astype(float) / row_sums_safe * 100
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", [SURF, PURP, PINK])
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label="% of True Class")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT, pad=12)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt_color = TEXT if cm_pct[i, j] < 60 else BG
            ax.text(j, i, f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=txt_color)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "confusion_matrix.png"))

def plot_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray, save_dir: str):
    _apply_dark_style()
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    
    # Check if y_proba has 3 columns (probabilities for Benign, Botnet, Malware)
    if len(y_proba.shape) == 1 or y_proba.shape[1] == 1:
        # Binary case fallback
        if len(y_proba.shape) == 2:
            y_proba = y_proba.squeeze()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("ROC & Precision-Recall Curves", fontsize=15, fontweight="bold", color=TEXT, y=1.01)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc      = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=PURP, lw=2.5, label=f"Classifier (AUC = {roc_auc:.4f})")
        ax1.plot([0,1],[0,1], color=TEXT, lw=1, linestyle=":", alpha=0.5, label="Random")
        ax1.fill_between(fpr, tpr, alpha=0.15, color=PURP)
        ax1.set_title("ROC Curve", fontsize=13, fontweight="bold", color=TEXT)
        ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
        ax1.legend(loc="lower right"); ax1.grid(True)
        ax1.set_xlim([-0.01, 1.01]); ax1.set_ylim([-0.01, 1.05])
        
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        pr_auc        = auc(rec, prec)
        ax2.plot(rec, prec, color=TEAL, lw=2.5, label=f"Classifier (AUC = {pr_auc:.4f})")
        ax2.fill_between(rec, prec, alpha=0.15, color=TEAL)
        ax2.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", color=TEXT)
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.legend(loc="upper right"); ax2.grid(True)
        ax2.set_xlim([-0.01, 1.01]); ax2.set_ylim([-0.01, 1.05])
        fig.tight_layout()
        _save(fig, os.path.join(save_dir, "roc_pr_curves.png"))
        return

    # Multi-class case
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Multiclass ROC & Precision-Recall Curves (One-vs-Rest)", fontsize=15, fontweight="bold", color=TEXT, y=1.01)
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = min(3, y_proba.shape[1])
    class_names = ["Benign", "Botnet", "Malware"]
    colors = [TEAL, PURP, PINK]
    
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, color=colors[i], lw=2.2, label=f"{class_names[i]} (AUC = {roc_auc:.4f})")
        except Exception:
            pass
            
    ax1.plot([0,1],[0,1], color=TEXT, lw=1, linestyle=":", alpha=0.5, label="Random")
    ax1.set_title("ROC Curve", fontsize=13, fontweight="bold", color=TEXT)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right"); ax1.grid(True)
    ax1.set_xlim([-0.01, 1.01]); ax1.set_ylim([-0.01, 1.05])
    
    for i in range(n_classes):
        try:
            prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            pr_auc = auc(rec, prec)
            ax2.plot(rec, prec, color=colors[i], lw=2.2, label=f"{class_names[i]} (AUC = {pr_auc:.4f})")
        except Exception:
            pass
            
    ax2.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", color=TEXT)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left"); ax2.grid(True)
    ax2.set_xlim([-0.01, 1.01]); ax2.set_ylim([-0.01, 1.05])
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "roc_pr_curves.png"))

def plot_metric_comparison(comparison_df: pd.DataFrame, save_dir: str):
    """
    Grouped bar chart comparing model performance.

    Fixes vs previous version:
    - Y-axis dynamically zoomed to [min_val - 0.03, 1.005] so differences are visible
    - Bar labels drawn INSIDE bars (rotated 90°) so they never overlap
    - Reference lines at 0.95 and 0.99 for quick benchmarking
    - Short model name labels to prevent legend overflow
    """
    _apply_dark_style()
    metrics_to_plot = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score"]
    avail   = [m for m in metrics_to_plot if m in comparison_df.columns]
    df_plot = comparison_df[avail].astype(float)
    models  = df_plot.index.tolist()

    # Shorten model names for legend readability
    short_names = {
        "Tabular Baseline (MLP)":           "Tabular Base",
        "Tabular Simple GAN + MLP":         "GAN + MLP",
        "Tabular CTGAN + MLP":              "CTGAN + MLP",
        "Hybrid Baseline":                  "Hybrid Base",
        "Hybrid Simple GAN + DistilBERT":   "GAN + BERT",
        "Hybrid CTGAN + DistilBERT":        "CTGAN + BERT",
    }
    display_names = [short_names.get(m, m) for m in models]

    x     = np.arange(len(avail))
    width = 0.80 / len(models)
    fig, ax = plt.subplots(figsize=(15, 7))

    all_vals = df_plot.values.flatten()
    y_min = max(0.0, float(all_vals.min()) - 0.04)
    y_max = 1.010

    for i, (model, dname) in enumerate(zip(models, display_names)):
        vals   = df_plot.loc[model].values
        offset = (i - len(models) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals - y_min, width,
                        label=dname,
                        color=COLS[i % len(COLS)], alpha=0.88,
                        edgecolor=BG, linewidth=0.8,
                        bottom=y_min)
        # Labels INSIDE bars (vertical) — no overlap
        for bar, val in zip(bars, vals):
            bar_h = bar.get_height()
            if bar_h > 0.005:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_min + bar_h / 2,
                    f"{val:.4f}",
                    ha="center", va="center",
                    fontsize=7, color="white",
                    fontweight="bold",
                    rotation=90,
                )

    # Reference lines
    for ref, style, lbl in [(0.99, "--", "0.99 ref"), (0.95, ":", "0.95 ref")]:
        if ref > y_min:
            ax.axhline(ref, color=GOLD, lw=1.0, linestyle=style, alpha=0.6, label=lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(avail, fontsize=11, fontweight="bold")
    ax.set_ylim(y_min, y_max)

    # Y-axis: show real values not the offset ones
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=15,
                 fontweight="bold", color=TEXT, pad=15)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.7)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "metric_comparison.png"))



def plot_per_class_heatmap(metrics: dict, save_dir: str):
    _apply_dark_style()
    data = {
        "Precision": [metrics.get("Benign_precision", metrics.get("base_Benign_precision", 0)), 
                      metrics.get("Botnet_precision", metrics.get("base_Botnet_precision", 0)), 
                      metrics.get("Malware_precision", metrics.get("base_Malware_precision", 0))],
        "Recall":    [metrics.get("Benign_recall", metrics.get("base_Benign_recall", 0)), 
                      metrics.get("Botnet_recall", metrics.get("base_Botnet_recall", 0)), 
                      metrics.get("Malware_recall", metrics.get("base_Malware_recall", 0))],
        "F1-Score":  [metrics.get("Benign_f1", metrics.get("base_Benign_f1", 0)), 
                      metrics.get("Botnet_f1", metrics.get("base_Botnet_f1", 0)), 
                      metrics.get("Malware_f1", metrics.get("base_Malware_f1", 0))],
    }
    df = pd.DataFrame(data, index=["Benign", "Botnet", "Malware"])
    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cm", [SURF, TEAL, GOLD])
    sns.heatmap(df, annot=True, fmt=".4f", cmap=cmap, linewidths=1, linecolor=BG, ax=ax,
                annot_kws={"fontsize": 13, "fontweight": "bold"}, vmin=0, vmax=1)
    ax.set_title("Per-Class Metrics Heatmap", fontsize=14, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("Metric", fontsize=11); ax.set_ylabel("Class", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "per_class_heatmap.png"))


def plot_class_distribution(before_counts: list, after_counts: list, class_names: list, save_dir: str):
    """
    Plot bar chart of class frequencies before vs after GAN/CTGAN augmentation.
    """
    _apply_dark_style()
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, before_counts, width, label="Before GAN (Real)", color=PINK, alpha=0.85)
    rects2 = ax.bar(x + width/2, after_counts, width, label="After GAN (Balanced)", color=TEAL, alpha=0.85)
    
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution Before vs After GAN Augmentation", fontsize=14, fontweight="bold", color=TEXT, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, axis="y")
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{int(h):,}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=TEXT)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{int(h):,}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=TEXT)
                    
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "class_distribution_comparison.png"))


def plot_synthetic_distribution_pca(real_X: np.ndarray, synth_X: np.ndarray, save_dir: str, filename: str = "pca_synthetic.png"):
    """
    Plot 2D PCA projection of real vs synthetic data to visually verify coverage.
    """
    if len(synth_X) == 0:
        logger.warning(" No synthetic samples to plot for PCA.")
        return
        
    _apply_dark_style()
    from sklearn.decomposition import PCA
    
    # Subsample for speed
    n_samples = min(1000, len(real_X), len(synth_X))
    rng = np.random.default_rng(42)
    
    real_idx = rng.choice(len(real_X), n_samples, replace=False)
    synth_idx = rng.choice(len(synth_X), n_samples, replace=False)
    
    R = real_X[real_idx]
    S = synth_X[synth_idx]
    
    combined = np.vstack([R, S])
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)
    
    coords_real = coords[:n_samples]
    coords_synth = coords[n_samples:]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords_real[:, 0], coords_real[:, 1], color=TEAL, alpha=0.5, label="Real Traffic", edgecolors="none", s=15)
    ax.scatter(coords_synth[:, 0], coords_synth[:, 1], color=PURP, alpha=0.5, label="Synthetic Traffic", edgecolors="none", s=15)
    
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Synthetic Data Quality (PCA Projection)", fontsize=14, fontweight="bold", color=TEXT, pad=12)
    ax.legend()
    ax.grid(True)
    
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, filename))


# ── NEW: Threshold sweep plot ──────────────────────────────────────────────────
def plot_threshold_sweep(score_table: list, optimal_threshold: float, save_dir: str):
    """
    Plot Precision, Recall, and F1 as a function of decision threshold.
    Marks the chosen optimal threshold with a vertical dashed line.

    This lets you visually see WHY the optimal threshold was chosen and
    verify the precision/recall trade-off is sensible.
    """
    if not score_table:
        return

    _apply_dark_style()
    df   = pd.DataFrame(score_table)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["threshold"], df["precision"],        color=PURP, lw=2,   label="Precision")
    ax.plot(df["threshold"], df["recall"],           color=PINK, lw=2,   label="Recall")
    ax.plot(df["threshold"], df["f1"],               color=TEAL, lw=2.5, label="F1-Score")
    ax.plot(df["threshold"], df["balanced_accuracy"], color=GOLD, lw=1.5,
            linestyle="--", label="Balanced Accuracy")

    ax.axvline(x=optimal_threshold, color=RED, lw=2, linestyle=":",
               label=f"Optimal Threshold = {optimal_threshold:.3f}")
    ax.fill_betweenx([0, 1], optimal_threshold - 0.02, optimal_threshold + 0.02,
                     color=RED, alpha=0.1)

    ax.set_title("Threshold Sweep — Precision / Recall / F1 vs Decision Threshold",
                 fontsize=14, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlim([df["threshold"].min() - 0.02, df["threshold"].max() + 0.02])
    ax.set_ylim([-0.02, 1.05])
    ax.legend(loc="center right", fontsize=10)
    ax.grid(True)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "threshold_sweep.png"))


# ── NEW: Attack-type detection breakdown bar chart ─────────────────────────────
def plot_attack_type_breakdown(attack_metrics: dict, save_dir: str):
    """
    Horizontal bar chart: for each attack type, show # detected vs # missed.
    Also shows the false-positive count for benign traffic.
    """
    per_attack = attack_metrics.get("per_attack", [])
    if not per_attack:
        return

    _apply_dark_style()
    labels   = [d["attack_type"] for d in per_attack]
    detected = [d["true_positives"]  for d in per_attack]
    missed   = [d["false_negatives"] for d in per_attack]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 1.2)))

    bars_d = ax.barh(y + 0.2, detected, 0.4, label="Detected (TP)",  color=TEAL, alpha=0.9)
    bars_m = ax.barh(y - 0.2, missed,   0.4, label="Missed (FN)",    color=RED,  alpha=0.9)

    for bar in bars_d:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 1, bar.get_y() + bar.get_height() / 2,
                    f"{int(w):,}", va="center", fontsize=9, color=TEAL, fontweight="bold")
    for bar in bars_m:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 1, bar.get_y() + bar.get_height() / 2,
                    f"{int(w):,}", va="center", fontsize=9, color=RED, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Number of Samples", fontsize=11)
    ax.set_title("Attack Type Detection Breakdown (TP vs FN)", fontsize=14,
                 fontweight="bold", color=TEXT, pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.4)

    # Annotate detection rates
    for i, d in enumerate(per_attack):
        rate = d["detection_rate"] * 100
        ax.text(ax.get_xlim()[1] * 0.98, i,
                f"{rate:.1f}%", va="center", ha="right",
                fontsize=10, color=GOLD, fontweight="bold")

    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "attack_type_breakdown.png"))


# ── Embed helper ───────────────────────────────────────────────────────────────
def _embed_plot(plots_dir: str, filename: str) -> str:
    """Return base64 data URI so charts display in any HTML viewer."""
    plot_path = os.path.join(plots_dir, filename)
    if not os.path.isfile(plot_path):
        logger.warning(" Plot not found: %s", plot_path)
        return ""
    with open(plot_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ── Main HTML report ───────────────────────────────────────────────────────────
def save_explanation_report(explanations: list, metrics: dict,
                             comparison_df: pd.DataFrame, reports_dir: str,
                             attack_metrics: dict = None,
                             optimal_threshold: float = None,
                             threshold_score_table: list = None):
    os.makedirs(reports_dir, exist_ok=True)
    path      = os.path.join(reports_dir, "explanation_report.html")
    plots_dir = os.path.normpath(os.path.join(reports_dir, "..", "plots"))

    acc      = metrics.get("accuracy",          0)
    bal_acc  = metrics.get("balanced_accuracy", 0)
    prec     = metrics.get("precision",         0)
    rec      = metrics.get("recall",            0)
    f1       = metrics.get("f1_score",          0)
    roc      = metrics.get("roc_auc",           0)
    n_pred   = metrics.get("n_predicted_anomaly", "N/A")
    n_true   = metrics.get("n_true_anomaly",      "N/A")
    fp_rate  = metrics.get("false_positive_rate", 0)

    # ── Threshold info block ─────────────────────────────────────────────────
    if optimal_threshold is not None:
        thresh_html = f"""
        <div class="info-block">
          <h2>&#9881; Optimal Decision Threshold</h2>
          <div class="threshold-banner">
            <span class="thresh-val">{optimal_threshold:.3f}</span>
            <span class="thresh-note">
              Auto-tuned on validation set to maximise F1
              (replaces hard-coded 0.5 that caused over-prediction)
            </span>
          </div>
        </div>"""
    else:
        thresh_html = ""

    # ── Attack type breakdown section ────────────────────────────────────────
    attack_section_html = ""
    if attack_metrics:
        per_attack    = attack_metrics.get("per_attack", [])
        total_att     = attack_metrics.get("total_attacks_in_test", 0)
        total_det     = attack_metrics.get("total_detected", 0)
        overall_rate  = attack_metrics.get("overall_detection_rate", 0) * 100
        fp_count      = attack_metrics.get("false_positives", 0)
        n_benign      = attack_metrics.get("n_benign_in_test", 0)
        fp_pct        = attack_metrics.get("false_positive_rate", 0) * 100

        rows_html = ""
        for d in per_attack:
            rate_color = TEAL if d["detection_rate"] >= 0.8 else (GOLD if d["detection_rate"] >= 0.5 else RED)
            rows_html += f"""
            <tr>
              <td><strong>{d['attack_type']}</strong></td>
              <td>{d['total_in_test']:,}</td>
              <td style="color:{TEAL}">{d['true_positives']:,}</td>
              <td style="color:{RED}">{d['false_negatives']:,}</td>
              <td style="color:{rate_color};font-weight:700">{d['detection_rate']*100:.1f}%</td>
            </tr>"""

        attack_breakdown_img = _embed_plot(plots_dir, "attack_type_breakdown.png")
        attack_img_tag = f'<img src="{attack_breakdown_img}" alt="Attack Breakdown" style="width:100%;border-radius:8px;margin-top:1rem">' if attack_breakdown_img else ""

        attack_section_html = f"""
        <h2>&#127919; Attack Detection Summary</h2>
        <div class="attack-summary-cards">
          <div class="atk-card">
            <div class="atk-val" style="color:{TEAL}">{total_det:,}</div>
            <div class="atk-lbl">Attacks Detected</div>
          </div>
          <div class="atk-card">
            <div class="atk-val" style="color:{PURP}">{total_att:,}</div>
            <div class="atk-lbl">Total Attacks in Test</div>
          </div>
          <div class="atk-card">
            <div class="atk-val" style="color:{'#43AA8B' if overall_rate>=80 else '#F9C74F'}">{overall_rate:.1f}%</div>
            <div class="atk-lbl">Overall Detection Rate</div>
          </div>
          <div class="atk-card">
            <div class="atk-val" style="color:{RED}">{fp_count:,}</div>
            <div class="atk-lbl">False Positives (Benign→Anomaly)</div>
          </div>
          <div class="atk-card">
            <div class="atk-val" style="color:{'#F94144' if fp_pct>10 else '#F9C74F' if fp_pct>5 else '#43AA8B'}">{fp_pct:.1f}%</div>
            <div class="atk-lbl">False Positive Rate</div>
          </div>
        </div>
        <table>
          <thead><tr>
            <th>Attack Type</th>
            <th>In Test Set</th>
            <th>Detected (TP)</th>
            <th>Missed (FN)</th>
            <th>Detection Rate</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        {attack_img_tag}"""

    # ── Explanation cards ────────────────────────────────────────────────────
    exp_html = ""
    for e in explanations:
        badge_color = "#F94144" if e["prediction"] == "Botnet" else ("#FF6584" if e["prediction"] == "Malware" else "#43AA8B")
        exp_lines   = e["explanation"].replace("\n", "<br>")
        exp_html   += f"""
        <div class="exp-card">
          <div class="exp-header">
            <span class="badge" style="background:{badge_color}">{e["prediction"]}</span>
            <span class="conf">Confidence: {e["confidence"]}</span>
            <span class="attack">Type: {e["attack_type"]}</span>
          </div>
          <div class="exp-body">{exp_lines}</div>
        </div>"""

    # ── Comparison table ─────────────────────────────────────────────────────
    comp_rows    = ""
    for model, row in comparison_df.iterrows():
        comp_rows += f"<tr><td>{model}</td>"
        for v in row.values:
            comp_rows += f"<td>{v}</td>"
        comp_rows += "</tr>"
    comp_headers = "".join(f"<th>{c}</th>" for c in comparison_df.columns)

    # ── Plot cards ───────────────────────────────────────────────────────────
    plot_specs = [
        ("threshold_sweep.png",             "Threshold Sweep", "Decision Threshold Sweep — Precision / Recall / F1"),
        ("attack_type_breakdown.png",        "Attack Breakdown", "Attack Type Detection Breakdown"),
        ("class_distribution_comparison.png","Class Distribution", "Before vs After GAN Class Distribution"),
        ("pca_synthetic.png",               "PCA Quality",     "Synthetic Data Quality (PCA Projection)"),
        ("gan_loss_curves.png",              "GAN Losses",      "GAN Generator vs Discriminator Loss"),
        ("classifier_training_history.png", "Training History", "Classifier Training History (Loss / Accuracy / F1)"),
        ("confusion_matrix.png",            "Confusion Matrix", "Confusion Matrix"),
        ("roc_pr_curves.png",               "ROC PR",          "ROC & Precision-Recall Curves"),
        ("metric_comparison.png",           "Comparison",      "Model Comparison (Baseline vs GAN-Augmented)"),
        ("per_class_heatmap.png",           "Per-Class",       "Per-Class Metrics Heatmap"),
    ]
    plot_cards = ""
    for fname, alt, caption in plot_specs:
        src = _embed_plot(plots_dir, fname)
        if src:
            plot_cards += f'<div class="plot-card"><img src="{src}" alt="{alt}"><p>{caption}</p></div>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GAN+BERT IoT Anomaly Detection Results Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:#0F0F1A; --surf:#1A1A2E; --purp:#6C63FF;
    --pink:#FF6584; --teal:#43AA8B; --gold:#F9C74F;
    --text:#E0E0FF; --red:#F94144;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:"Inter",sans-serif; padding:2rem; }}
  h1 {{ font-size:2.2rem; font-weight:900; background:linear-gradient(135deg,var(--purp),var(--pink));
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:.4rem; }}
  .subtitle {{ color:#888; margin-bottom:2.5rem; font-size:.95rem; }}
  h2 {{ font-size:1.3rem; font-weight:700; margin:2rem 0 1rem; color:var(--purp);
        border-left:4px solid var(--purp); padding-left:.75rem; }}
  .info-block {{ background:var(--surf); border-radius:14px; padding:1.4rem; margin:1.5rem 0;
                  border:1px solid #2A2A4A; }}
  .threshold-banner {{ display:flex; align-items:center; gap:1.5rem; margin-top:.8rem; flex-wrap:wrap; }}
  .thresh-val {{ font-size:3rem; font-weight:900; color:var(--gold); }}
  .thresh-note {{ font-size:.9rem; color:#BCC0D6; max-width:480px; line-height:1.6; }}
  .metrics-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
                   gap:1.2rem; margin-bottom:2rem; }}
  .metric-card {{ background:var(--surf); border-radius:14px; padding:1.4rem 1rem; text-align:center;
                  border:1px solid #2A2A4A; box-shadow:0 4px 24px rgba(108,99,255,.15); }}
  .metric-card .val {{ font-size:2rem; font-weight:900; }}
  .metric-card .lbl {{ font-size:.75rem; color:#888; margin-top:.3rem; text-transform:uppercase;
                       letter-spacing:1px; }}
  .attack-summary-cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
                            gap:1rem; margin-bottom:1.5rem; }}
  .atk-card {{ background:var(--surf); border-radius:12px; padding:1.2rem; text-align:center;
               border:1px solid #2A2A4A; }}
  .atk-val {{ font-size:1.8rem; font-weight:900; }}
  .atk-lbl {{ font-size:.75rem; color:#888; margin-top:.3rem; text-transform:uppercase; }}
  .plot-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr));
                gap:1.5rem; margin-bottom:2rem; }}
  .plot-card {{ background:var(--surf); border-radius:14px; padding:1rem; border:1px solid #2A2A4A;
                text-align:center; }}
  .plot-card img {{ width:100%; border-radius:8px; }}
  .plot-card p {{ margin-top:.6rem; font-size:.85rem; color:#888; }}
  table {{ width:100%; border-collapse:collapse; background:var(--surf); border-radius:12px;
           overflow:hidden; margin-bottom:2rem; }}
  th {{ background:var(--purp); color:#fff; padding:.85rem 1rem; font-size:.9rem; text-align:left; }}
  td {{ padding:.75rem 1rem; font-size:.9rem; border-bottom:1px solid #2A2A4A; }}
  tr:hover {{ background:#22224A; }}
  .exp-card {{ background:var(--surf); border-radius:14px; padding:1.4rem; margin-bottom:1.2rem;
               border:1px solid #2A2A4A; border-left:4px solid var(--purp); }}
  .exp-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem; flex-wrap:wrap; }}
  .badge {{ padding:.3rem .9rem; border-radius:20px; font-size:.8rem; font-weight:700; color:#fff; }}
  .conf {{ font-size:.85rem; color:var(--gold); font-weight:600; }}
  .attack {{ font-size:.85rem; color:var(--teal); font-weight:600; }}
  .exp-body {{ font-size:.88rem; line-height:1.8; color:#BCC0D6; font-family:monospace;
               background:#0A0A18; border-radius:8px; padding:1rem; white-space:pre-wrap; }}
  footer {{ text-align:center; margin-top:3rem; color:#444; font-size:.8rem; }}
</style>
</head>
<body>
<h1>&#128737; GAN + DistilBERT Hybrid IoT Anomaly Detection</h1>
<p class="subtitle">CIC-IDS-2018 Dataset &nbsp;|&nbsp; Botnet &amp; Malware Traffic Analysis &nbsp;|&nbsp; DistilBERT-Powered Explainability</p>

{thresh_html}

<h2>&#128200; Overall Performance Metrics</h2>
<div class="metrics-grid">
  <div class="metric-card"><div class="val" style="color:var(--purp)">{acc:.4f}</div><div class="lbl">Accuracy</div></div>
  <div class="metric-card"><div class="val" style="color:var(--teal)">{bal_acc:.4f}</div><div class="lbl">Balanced Accuracy</div></div>
  <div class="metric-card"><div class="val" style="color:var(--teal)">{prec:.4f}</div><div class="lbl">Precision</div></div>
  <div class="metric-card"><div class="val" style="color:var(--gold)">{rec:.4f}</div><div class="lbl">Recall</div></div>
  <div class="metric-card"><div class="val" style="color:var(--pink)">{f1:.4f}</div><div class="lbl">F1-Score</div></div>
  <div class="metric-card"><div class="val" style="color:var(--red)">{roc:.4f}</div><div class="lbl">ROC-AUC</div></div>
  <div class="metric-card"><div class="val" style="color:var(--gold)">{n_pred}</div><div class="lbl">Predicted Anomaly</div></div>
  <div class="metric-card"><div class="val" style="color:var(--purp)">{n_true}</div><div class="lbl">True Anomaly</div></div>
  <div class="metric-card"><div class="val" style="color:{'var(--red)' if isinstance(fp_rate,float) and fp_rate>0.1 else 'var(--teal)'}">{fp_rate:.4f}</div><div class="lbl">False Positive Rate</div></div>
</div>

{attack_section_html}

<h2>&#128202; Training &amp; Performance Plots</h2>
<div class="plot-grid">
{plot_cards}
</div>

<h2>&#128301; Model Comparison Table</h2>
<table>
  <thead><tr><th>Model</th>{comp_headers}</tr></thead>
  <tbody>{comp_rows}</tbody>
</table>

<h2>&#129504; OpenAI LLM-Based Explanations (Sample Predictions)</h2>
{exp_html}
<footer>Generated by GAN+LLM IoT Anomaly Detection Pipeline &nbsp;&bull;&nbsp; CIC-IDS-2018</footer>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(" HTML Report saved: %s", path)
    return path


def save_gan_ctgan_comparison_report(
    comparison_df: pd.DataFrame,
    simple_gan_metrics: dict,
    ctgan_metrics: dict,
    reports_dir: str,
    simple_plots_dir: str,
    ctgan_plots_dir: str,
):
    """HTML report comparing Simple GAN vs CTGAN in separate sections."""
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "gan_vs_ctgan_comparison.html")

    comp_rows = ""
    for model, row in comparison_df.iterrows():
        comp_rows += f"<tr><td><strong>{model}</strong></td>"
        for v in row.values:
            comp_rows += f"<td>{v}</td>"
        comp_rows += "</tr>"
    comp_headers = "".join(f"<th>{c}</th>" for c in comparison_df.columns)

    def _metric_cards(metrics: dict, accent: str) -> str:
        keys = [
            ("accuracy",          "Accuracy"),
            ("balanced_accuracy", "Balanced Acc"),
            ("precision",         "Precision"),
            ("recall",            "Recall"),
            ("f1_score",          "F1-Score"),
            ("roc_auc",           "ROC-AUC"),
        ]
        cards = ""
        for k, lbl in keys:
            val = metrics.get(k, 0)
            cards += (f'<div class="metric-card">'
                      f'<div class="val" style="color:{accent}">{val:.4f}</div>'
                      f'<div class="lbl">{lbl}</div></div>')
        return cards

    plot_files = [
        ("threshold_sweep.png",             "Decision Threshold Sweep"),
        ("attack_type_breakdown.png",        "Attack Type Breakdown"),
        ("gan_loss_curves.png",              "Generator vs Discriminator Loss"),
        ("classifier_training_history.png", "Classifier Training History"),
        ("confusion_matrix.png",            "Confusion Matrix"),
        ("roc_pr_curves.png",               "ROC & PR Curves"),
        ("metric_comparison.png",           "Metrics Bar Chart"),
        ("per_class_heatmap.png",           "Per-Class Heatmap"),
    ]

    def _plot_section(plots_dir: str, heading: str) -> str:
        html = f'<h2>{heading}</h2><div class="plot-grid">'
        for fname, caption in plot_files:
            src = _embed_plot(plots_dir, fname)
            if src:
                html += (f'<div class="plot-card"><img src="{src}" alt="{caption}">'
                         f'<p>{caption}</p></div>')
        html += "</div>"
        return html

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Simple GAN vs CTGAN Comparison</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{ --bg:#0F0F1A; --surf:#1A1A2E; --purp:#6C63FF; --teal:#43AA8B; --pink:#FF6584; --text:#E0E0FF; }}
  body {{ background:var(--bg); color:var(--text); font-family:Inter,sans-serif; padding:2rem; max-width:1400px; margin:0 auto; }}
  h1 {{ font-size:2rem; color:var(--purp); }}
  h2 {{ font-size:1.2rem; margin:2rem 0 1rem; border-left:4px solid var(--teal); padding-left:.75rem; }}
  h3 {{ color:var(--pink); margin:1.5rem 0 .75rem; }}
  .subtitle {{ color:#888; margin-bottom:2rem; }}
  .metrics-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:1rem; margin:1rem 0 2rem; }}
  .metric-card {{ background:var(--surf); border-radius:12px; padding:1rem; text-align:center; border:1px solid #2A2A4A; }}
  .metric-card .val {{ font-size:1.5rem; font-weight:700; }}
  .metric-card .lbl {{ font-size:.75rem; color:#888; text-transform:uppercase; }}
  .plot-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(400px,1fr)); gap:1.2rem; }}
  .plot-card {{ background:var(--surf); border-radius:12px; padding:1rem; border:1px solid #2A2A4A; }}
  .plot-card img {{ width:100%; border-radius:8px; }}
  .plot-card p {{ font-size:.85rem; color:#888; margin-top:.5rem; }}
  table {{ width:100%; border-collapse:collapse; background:var(--surf); border-radius:12px; overflow:hidden; margin:1.5rem 0; }}
  th {{ background:var(--purp); color:#fff; padding:.75rem 1rem; text-align:left; }}
  td {{ padding:.7rem 1rem; border-bottom:1px solid #2A2A4A; }}
  .note {{ background:#1A1A2E; border-left:4px solid var(--pink); padding:1rem; margin:1rem 0; font-size:.9rem; line-height:1.6; }}
</style>
</head>
<body>
<h1>Simple GAN vs CTGAN — Comparison Report</h1>
<p class="subtitle">Same MLP classifier and test set; only the synthetic augmentation method differs.</p>

<div class="note">
  <strong>Simple GAN:</strong> MLP generator + discriminator (vanilla GAN) with residual skip + D-score quality gate.<br>
  <strong>CTGAN:</strong> Conditional Tabular GAN with mode-specific normalisation + Mahalanobis quality gate.
</div>

<h2>All Models — Metrics Table</h2>
<table>
  <thead><tr><th>Model</th>{comp_headers}</tr></thead>
  <tbody>{comp_rows}</tbody>
</table>

<h3>Simple GAN — Augmented Classifier Metrics</h3>
<div class="metrics-grid">{_metric_cards(simple_gan_metrics, "var(--purp)")}</div>

<h3>CTGAN — Augmented Classifier Metrics</h3>
<div class="metrics-grid">{_metric_cards(ctgan_metrics, "var(--teal)")}</div>

{_plot_section(simple_plots_dir, "Simple GAN — Plots (results/simple_gan/plots/)")}
{_plot_section(ctgan_plots_dir, "CTGAN — Plots (results/ctgan/plots/)")}

<footer style="text-align:center;margin-top:3rem;color:#555;">GAN vs CTGAN comparison &bull; IoT Anomaly Detection</footer>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(" GAN vs CTGAN comparison report: %s", path)
    return path


def save_augmentation_comparison_json(
    comparison_df: pd.DataFrame,
    simple_gan_metrics: dict,
    ctgan_metrics: dict,
    reports_dir: str,
):
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "gan_vs_ctgan_metrics.json")
    payload = {
        "comparison_table": comparison_df.reset_index().to_dict(orient="records"),
        "simple_gan_augmented_classifier": {k: round(v, 6) if isinstance(v, float) else v
                                            for k, v in simple_gan_metrics.items()},
        "ctgan_augmented_classifier": {k: round(v, 6) if isinstance(v, float) else v
                                       for k, v in ctgan_metrics.items()},
        "simple_gan_plots_dir": "results/simple_gan/plots/",
        "ctgan_plots_dir":      "results/ctgan/plots/",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(" GAN vs CTGAN JSON saved: %s", path)
    return path


def save_metrics_json(metrics: dict, comparison_df: pd.DataFrame,
                      explanations: list, reports_dir: str,
                      attack_metrics: dict = None,
                      optimal_threshold: float = None):
    os.makedirs(reports_dir, exist_ok=True)
    payload = {
        "optimal_threshold":    optimal_threshold,
        "final_metrics":        {k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in metrics.items()},
        "attack_type_breakdown": attack_metrics,
        "model_comparison":     comparison_df.reset_index().to_dict(orient="records"),
        "bert_explanations":    explanations,
    }
    path = os.path.join(reports_dir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(" JSON Results saved: %s", path)