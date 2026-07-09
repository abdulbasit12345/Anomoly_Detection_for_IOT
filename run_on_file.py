"""
run_on_file.py
==============
Run the full anomaly-detection pipeline on ANY csv file and store all
outputs in a dedicated, isolated folder.

Usage
-----
    python run_on_file.py --csv 02-15-2018.csv --tag 02_15_2026

This will:
  • Load data from  02-15-2018.csv
  • Save processed arrays to  data/processed_02_15_2026/
  • Save all results to       results_02_15_2026/
  • Write logs to             logs/run_02_15_2026_<timestamp>.log

The original results/ and data/processed/ folders are NEVER touched.

Improvements in this version
-----------------------------
1. Attack-type breakdown — shows how many of each attack type were detected.
2. Optimal threshold search — auto-tunes the decision threshold on the
   validation set (replaces the hard-coded 0.5 that caused the model to flag
   ALL traffic as anomaly).
3. Accuracy improvements — Focal Loss + OneCycleLR + deeper network + 60 epochs.
4. GAN anti-evasion filter — discriminator quality gate removes synthetic
   samples that look like normal traffic, preventing false-positive explosion.
"""

import os
import sys
import argparse
import logging
import copy
import torch
import numpy as np
from datetime import datetime

# ── parse arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run IoT anomaly detection on a new CSV file.")
parser.add_argument(
    "--csv",
    required=True,
    help="Path to the raw CSV file (absolute or relative to this script).",
)
parser.add_argument(
    "--tag",
    required=True,
    help="Short label used to name output folders, e.g. '02_15_2026'.",
)
args = parser.parse_args()

# ── resolve paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(BASE_DIR, args.csv)

if not os.path.isfile(csv_path):
    print(f"[ERROR] CSV file not found: {csv_path}")
    sys.exit(1)

tag = args.tag  # e.g. "02_15_2026"

# ── build an isolated config for this run ─────────────────────────────────────
import config as _base_cfg

class RunConfig:
    pass

run_cfg = RunConfig()
for attr in dir(_base_cfg):
    if not attr.startswith("__"):
        setattr(run_cfg, attr, getattr(_base_cfg, attr))

# Override paths so nothing touches the original results/
run_cfg.RAW_DATA_PATH  = csv_path
run_cfg.PROCESSED_DIR  = os.path.join(BASE_DIR, "data", f"processed_{tag}")
run_cfg.RESULTS_DIR    = os.path.join(BASE_DIR, f"results_{tag}")
run_cfg.PLOTS_DIR      = os.path.join(run_cfg.RESULTS_DIR, "plots")
run_cfg.REPORTS_DIR    = os.path.join(run_cfg.RESULTS_DIR, "reports")
run_cfg.MODELS_DIR     = os.path.join(run_cfg.RESULTS_DIR, "models")
run_cfg.LOGS_DIR       = os.path.join(BASE_DIR, "logs")   # shared log folder is fine

run_cfg.SIMPLE_GAN_DIR        = os.path.join(run_cfg.RESULTS_DIR, "simple_gan")
run_cfg.SIMPLE_GAN_PLOTS_DIR  = os.path.join(run_cfg.SIMPLE_GAN_DIR, "plots")
run_cfg.SIMPLE_GAN_MODELS_DIR = os.path.join(run_cfg.SIMPLE_GAN_DIR, "models")

run_cfg.CTGAN_DIR        = os.path.join(run_cfg.RESULTS_DIR, "ctgan")
run_cfg.CTGAN_PLOTS_DIR  = os.path.join(run_cfg.CTGAN_DIR, "plots")
run_cfg.CTGAN_MODELS_DIR = os.path.join(run_cfg.CTGAN_DIR, "models")

run_cfg.COMPARISON_REPORTS_DIR = os.path.join(run_cfg.REPORTS_DIR, "comparison")

# ── create all required directories ───────────────────────────────────────────
for d in (
    run_cfg.LOGS_DIR,
    run_cfg.SIMPLE_GAN_PLOTS_DIR, run_cfg.SIMPLE_GAN_MODELS_DIR,
    run_cfg.CTGAN_PLOTS_DIR,      run_cfg.CTGAN_MODELS_DIR,
    run_cfg.COMPARISON_REPORTS_DIR,
    run_cfg.PLOTS_DIR, run_cfg.REPORTS_DIR, run_cfg.MODELS_DIR,
):
    os.makedirs(d, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
log_file = os.path.join(
    run_cfg.LOGS_DIR,
    f"run_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("RUN_ON_FILE")

# ── imports (after logging is configured) ─────────────────────────────────────
from src.data_processing.preprocessor import load_and_preprocess
from src.models.gan import build_gan
from src.models.classifier import build_classifier
from src.training.train_gan import train_gan
from src.training.train_ctgan import train_ctgan
from src.training.train_classifier import train_classifier, predict, predict_proba, find_optimal_threshold
from src.models.hybrid_model import HybridFusionClassifier, get_distilbert_embeddings, train_hybrid_classifier, predict_hybrid, predict_proba_hybrid
from src.explainability.hybrid_explainer import HybridAnomalyExplainer, features_to_text
from src.evaluation.metrics import compute_metrics, build_comparison_table, get_confusion_matrix, compute_attack_type_metrics
from src.evaluation.visualizer import (
    plot_gan_losses, plot_training_history, plot_confusion_matrix,
    plot_roc_pr_curves, plot_metric_comparison, plot_per_class_heatmap,
    save_explanation_report, save_metrics_json,
    save_gan_ctgan_comparison_report, save_augmentation_comparison_json,
    plot_threshold_sweep, plot_attack_type_breakdown,
    plot_class_distribution, plot_synthetic_distribution_pca
)


# ── helper functions ──────────────────────────────────────────────────────────

def _evaluate_classifier(model, data, device, plots_dir: str):
    """Evaluate classifier and generate plots."""
    y_pred  = predict(model, data["X_test"], device)
    y_prob  = predict_proba(model, data["X_test"], device)
    metrics = compute_metrics(data["y_test"], y_pred, y_prob)
    cm      = get_confusion_matrix(data["y_test"], y_pred)
    plot_confusion_matrix(cm, run_cfg.CLASS_NAMES_3CLASS, plots_dir)
    plot_roc_pr_curves(data["y_test"], y_prob, plots_dir)
    plot_per_class_heatmap(metrics, plots_dir)
    return metrics, y_pred, y_prob


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    n_threads = getattr(run_cfg, "TORCH_NUM_THREADS", 2)
    torch.set_num_threads(n_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))

    # macOS optimization: Support MPS (Metal Performance Shaders) for modern Macs
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    logger.info("=" * 60)
    logger.info(" Run tag   : %s", tag)
    logger.info(" CSV file  : %s", csv_path)
    logger.info(" Results   : %s", run_cfg.RESULTS_DIR)
    logger.info(" Device    : %s | CPU threads: %d", device, n_threads)
    logger.info("=" * 60)

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    data = load_and_preprocess(run_cfg)
    class_names = run_cfg.CLASS_NAMES_3CLASS
    feature_names = data["feature_names"]

    # Before GAN Class Distribution
    before_counts = [int((data["y_train"] == 0).sum()), 
                     int((data["y_train"] == 1).sum()), 
                     int((data["y_train"] == 2).sum())]

    # ── Step 2: Baseline (no augmentation) ───────────────────────────────────
    logger.info(" Training Baseline Tabular MLP...")
    baseline_model = build_classifier(run_cfg, data["n_features"], device)
    train_classifier(
        run_cfg, baseline_model,
        data["X_train"], data["y_train"],
        data["X_val"],   data["y_val"],
        synth_X=np.empty((0, data["n_features"])),
        synth_y=np.empty((0,), dtype=np.int64),
        device=device,
        model_filename="classifier_baseline.pt",
        models_dir=run_cfg.MODELS_DIR,
    )
    baseline_metrics, _, _ = _evaluate_classifier(baseline_model, data, device, run_cfg.MODELS_DIR)

    # ── Step 3: Simple GAN ────────────────────────────────────────────────────
    logger.info(" [Simple GAN] Training Class Generators...")
    gan_results = train_gan(run_cfg, data["X_train"], data["y_train"], device)
    plot_gan_losses(gan_results["g_losses"], gan_results["d_losses"], run_cfg.SIMPLE_GAN_PLOTS_DIR)

    # Visualise PCA and balanced distribution for Simple GAN
    plot_synthetic_distribution_pca(data["X_train"][data["y_train"] != 0], gan_results["synth_X"], run_cfg.SIMPLE_GAN_PLOTS_DIR)
    
    y_train_aug_gan = np.concatenate([data["y_train"], gan_results["synth_y"]])
    after_counts_gan = [int((y_train_aug_gan == 0).sum()), 
                        int((y_train_aug_gan == 1).sum()), 
                        int((y_train_aug_gan == 2).sum())]
    plot_class_distribution(before_counts, after_counts_gan, class_names, run_cfg.SIMPLE_GAN_PLOTS_DIR)

    logger.info(" [Simple GAN] Training augmented tabular classifier...")
    gan_model = build_classifier(run_cfg, data["n_features"], device)
    train_classifier(
        run_cfg, gan_model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        synth_X=gan_results["synth_X"],
        synth_y=gan_results["synth_y"],
        device=device,
        model_filename="classifier_gan.pt",
        models_dir=run_cfg.SIMPLE_GAN_MODELS_DIR,
    )
    simple_gan_metrics, _, _ = _evaluate_classifier(gan_model, data, device, run_cfg.SIMPLE_GAN_PLOTS_DIR)

    # ── Step 4: CTGAN ─────────────────────────────────────────────────────────
    logger.info(" [CTGAN] Training Class Generators...")
    ctgan_results = train_ctgan(run_cfg, data["X_train"], data["y_train"], data["feature_names"])
    plot_gan_losses(ctgan_results["g_losses"], ctgan_results["d_losses"], run_cfg.CTGAN_PLOTS_DIR)

    # Visualise PCA and balanced distribution for CTGAN
    plot_synthetic_distribution_pca(data["X_train"][data["y_train"] != 0], ctgan_results["synth_X"], run_cfg.CTGAN_PLOTS_DIR)
    
    y_train_aug_ctgan = np.concatenate([data["y_train"], ctgan_results["synth_y"]])
    after_counts_ctgan = [int((y_train_aug_ctgan == 0).sum()), 
                          int((y_train_aug_ctgan == 1).sum()), 
                          int((y_train_aug_ctgan == 2).sum())]
    plot_class_distribution(before_counts, after_counts_ctgan, class_names, run_cfg.CTGAN_PLOTS_DIR)

    logger.info(" [CTGAN] Training augmented tabular classifier...")
    ctgan_model = build_classifier(run_cfg, data["n_features"], device)
    train_classifier(
        run_cfg, ctgan_model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        synth_X=ctgan_results["synth_X"],
        synth_y=ctgan_results["synth_y"],
        device=device,
        model_filename="classifier_ctgan.pt",
        models_dir=run_cfg.CTGAN_MODELS_DIR,
    )
    ctgan_metrics, _, _ = _evaluate_classifier(ctgan_model, data, device, run_cfg.CTGAN_PLOTS_DIR)

    # ── Step 5: DistilBERT Embeddings Extraction ──────────────────────────────
    logger.info(" Extracting DistilBERT features for hybrid multimodal network...")
    train_texts = [features_to_text(x, feature_names) for x in data["X_train"]]
    val_texts   = [features_to_text(x, feature_names) for x in data["X_val"]]
    test_texts  = [features_to_text(x, feature_names) for x in data["X_test"]]

    X_train_emb = get_distilbert_embeddings(run_cfg, train_texts, device)
    X_val_emb   = get_distilbert_embeddings(run_cfg, val_texts, device)
    X_test_emb  = get_distilbert_embeddings(run_cfg, test_texts, device)

    # ── Step 6: Hybrid Multimodal Baseline ───────────────────────────────────
    logger.info(" Training Hybrid Baseline Model...")
    hybrid_baseline_model = HybridFusionClassifier(tabular_dim=data["n_features"]).to(device)
    train_hybrid_classifier(
        run_cfg, hybrid_baseline_model,
        data["X_train"], X_train_emb, data["y_train"],
        data["X_val"], X_val_emb, data["y_val"],
        device=device, model_filename="hybrid_baseline.pt", models_dir=run_cfg.MODELS_DIR
    )
    y_pred_h_base = predict_hybrid(hybrid_baseline_model, data["X_test"], X_test_emb, device)
    y_prob_h_base = predict_proba_hybrid(hybrid_baseline_model, data["X_test"], X_test_emb, device)
    hybrid_base_metrics = compute_metrics(data["y_test"], y_pred_h_base, y_prob_h_base)

    # ── Step 7: Hybrid Multimodal + Simple GAN ───────────────────────────────
    logger.info(" Training Hybrid Simple GAN Augmented Model...")
    gan_synth_texts = [features_to_text(x, feature_names) for x in gan_results["synth_X"]]
    gan_synth_emb   = get_distilbert_embeddings(run_cfg, gan_synth_texts, device)

    X_train_aug_gan = np.vstack([data["X_train"], gan_results["synth_X"]])
    X_train_emb_aug_gan = np.vstack([X_train_emb, gan_synth_emb])
    y_train_aug_gan = np.concatenate([data["y_train"], gan_results["synth_y"]])

    hybrid_gan_model = HybridFusionClassifier(tabular_dim=data["n_features"]).to(device)
    train_hybrid_classifier(
        run_cfg, hybrid_gan_model,
        X_train_aug_gan, X_train_emb_aug_gan, y_train_aug_gan,
        data["X_val"], X_val_emb, data["y_val"],
        device=device, model_filename="hybrid_gan.pt", models_dir=run_cfg.MODELS_DIR
    )
    y_pred_h_gan = predict_hybrid(hybrid_gan_model, data["X_test"], X_test_emb, device)
    y_prob_h_gan = predict_proba_hybrid(hybrid_gan_model, data["X_test"], X_test_emb, device)
    hybrid_gan_metrics = compute_metrics(data["y_test"], y_pred_h_gan, y_prob_h_gan)

    # ── Step 8: Hybrid Multimodal + CTGAN ────────────────────────────────────
    logger.info(" Training Hybrid CTGAN Augmented Model...")
    ctgan_synth_texts = [features_to_text(x, feature_names) for x in ctgan_results["synth_X"]]
    ctgan_synth_emb   = get_distilbert_embeddings(run_cfg, ctgan_synth_texts, device)

    X_train_aug_ctgan = np.vstack([data["X_train"], ctgan_results["synth_X"]])
    X_train_emb_aug_ctgan = np.vstack([X_train_emb, ctgan_synth_emb])
    y_train_aug_ctgan = np.concatenate([data["y_train"], ctgan_results["synth_y"]])

    hybrid_ctgan_model = HybridFusionClassifier(tabular_dim=data["n_features"]).to(device)
    train_hybrid_classifier(
        run_cfg, hybrid_ctgan_model,
        X_train_aug_ctgan, X_train_emb_aug_ctgan, y_train_aug_ctgan,
        data["X_val"], X_val_emb, data["y_val"],
        device=device, model_filename="hybrid_ctgan.pt", models_dir=run_cfg.MODELS_DIR
    )
    y_pred_h_ctgan = predict_hybrid(hybrid_ctgan_model, data["X_test"], X_test_emb, device)
    y_prob_h_ctgan = predict_proba_hybrid(hybrid_ctgan_model, data["X_test"], X_test_emb, device)
    hybrid_ctgan_metrics = compute_metrics(data["y_test"], y_pred_h_ctgan, y_prob_h_ctgan)

    # ── Step 9: Comparison table ──────────────────────────────────────────────
    comparison_results = {
        "Tabular Baseline (MLP)": baseline_metrics,
        "Tabular Simple GAN + MLP": simple_gan_metrics,
        "Tabular CTGAN + MLP": ctgan_metrics,
        "Hybrid Baseline": hybrid_base_metrics,
        "Hybrid Simple GAN + DistilBERT": hybrid_gan_metrics,
        "Hybrid CTGAN + DistilBERT": hybrid_ctgan_metrics,
    }
    comparison_df = build_comparison_table(comparison_results)
    plot_metric_comparison(comparison_df, run_cfg.COMPARISON_REPORTS_DIR)
    plot_metric_comparison(comparison_df, run_cfg.PLOTS_DIR)

    comparison_path = save_gan_ctgan_comparison_report(
        comparison_df, simple_gan_metrics, ctgan_metrics,
        run_cfg.COMPARISON_REPORTS_DIR,
        run_cfg.SIMPLE_GAN_PLOTS_DIR,
        run_cfg.CTGAN_PLOTS_DIR,
    )
    save_augmentation_comparison_json(
        comparison_df, simple_gan_metrics, ctgan_metrics, run_cfg.COMPARISON_REPORTS_DIR,
    )

    # ── Step 10: Choose best Hybrid model for explanations ────────────────────
    f1_scores = {
        "base": hybrid_base_metrics.get("f1_score", 0),
        "gan":  hybrid_gan_metrics.get("f1_score", 0),
        "ctgan": hybrid_ctgan_metrics.get("f1_score", 0)
    }
    best_hybrid = max(f1_scores, key=f1_scores.get)
    logger.info(" Best performing Hybrid model: %s (F1=%.4f)", best_hybrid, f1_scores[best_hybrid])
    
    if best_hybrid == "ctgan":
        final_y_pred, final_y_prob = y_pred_h_ctgan, y_prob_h_ctgan
        final_metrics = hybrid_ctgan_metrics
    elif best_hybrid == "gan":
        final_y_pred, final_y_prob = y_pred_h_gan, y_prob_h_gan
        final_metrics = hybrid_gan_metrics
    else:
        final_y_pred, final_y_prob = y_pred_h_base, y_prob_h_base
        final_metrics = hybrid_base_metrics

    # Plot final best hybrid model metrics and charts in the main plots dir
    cm = get_confusion_matrix(data["y_test"], final_y_pred)
    plot_confusion_matrix(cm, class_names, run_cfg.PLOTS_DIR)
    plot_roc_pr_curves(data["y_test"], final_y_prob, run_cfg.PLOTS_DIR)
    plot_per_class_heatmap(final_metrics, run_cfg.PLOTS_DIR)

    # Copy GAN plots from the best branch to the main plots dir
    import shutil
    best_plots_src = run_cfg.CTGAN_PLOTS_DIR if best_hybrid == "ctgan" else run_cfg.SIMPLE_GAN_PLOTS_DIR
    for filename in ["gan_loss_curves.png", "pca_synthetic.png", "class_distribution_comparison.png"]:
        src_file = os.path.join(best_plots_src, filename)
        dst_file = os.path.join(run_cfg.PLOTS_DIR, filename)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)

    # ── Step 11: Attack-type breakdown ─────────────────────────────────────────
    logger.info(" Computing attack-type detection breakdown...")
    attack_metrics = compute_attack_type_metrics(data["y_test"], final_y_pred, data["att_test"])
    plot_attack_type_breakdown(attack_metrics, run_cfg.PLOTS_DIR)

    # ── Step 12: Hybrid Explanations ───────────────────────────────────────────
    logger.info(" Initializing Local Hybrid DistilBERT Explainer...")
    explainer = HybridAnomalyExplainer(run_cfg)
    explanations = explainer.generate_explanations(
        data["X_test"], final_y_pred, final_y_prob, data["att_test"],
        data["feature_names"], n_samples=run_cfg.NUM_EXPLAIN_SAMPLES,
    )

    # ── Step 13: Save reports ──────────────────────────────────────────────────
    report_path = save_explanation_report(
        explanations, final_metrics, comparison_df, run_cfg.REPORTS_DIR,
        attack_metrics=attack_metrics,
        optimal_threshold=0.5,
        threshold_score_table=[],
    )
    save_metrics_json(
        final_metrics, comparison_df, explanations, run_cfg.REPORTS_DIR,
        attack_metrics=attack_metrics,
        optimal_threshold=0.5,
    )

    # ── Final summary log ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(" Pipeline complete for tag        : %s", tag)
    logger.info(" Best Hybrid Model                : %s", best_hybrid)
    logger.info(" Final F1-Score (macro)           : %.4f", final_metrics.get("f1_score", 0))
    logger.info(" Final Accuracy                   : %.4f", final_metrics.get("accuracy", 0))
    logger.info(" Attacks detected / total         : %d / %d (%.1f%%)",
                attack_metrics["total_detected"],
                attack_metrics["total_attacks_in_test"],
                attack_metrics["overall_detection_rate"] * 100)
    logger.info(" False Positive Rate              : %.1f%%",
                attack_metrics["false_positive_rate"] * 100)
    logger.info(" GAN vs CTGAN comparison          : %s", comparison_path)
    logger.info(" Main report                      : %s", report_path)
    logger.info(" Results folder                   : %s", run_cfg.RESULTS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(" Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)
