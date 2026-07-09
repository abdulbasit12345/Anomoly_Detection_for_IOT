import os
import logging
import torch
import numpy as np
from datetime import datetime
import pandas as pd

import config
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

for d in (
    config.LOGS_DIR, config.SIMPLE_GAN_PLOTS_DIR, config.SIMPLE_GAN_MODELS_DIR,
    config.CTGAN_PLOTS_DIR, config.CTGAN_MODELS_DIR, config.COMPARISON_REPORTS_DIR,
    config.REPORTS_DIR, config.PLOTS_DIR, config.MODELS_DIR
):
    os.makedirs(d, exist_ok=True)

log_file = os.path.join(config.LOGS_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MAIN")


def _evaluate_classifier(model, data, device, plots_dir: str):
    """Test tabular classifier and write plots."""
    y_pred = predict(model, data["X_test"], device)
    y_prob = predict_proba(model, data["X_test"], device)
    metrics = compute_metrics(data["y_test"], y_pred, y_prob)
    cm = get_confusion_matrix(data["y_test"], y_pred)
    plot_confusion_matrix(cm, config.CLASS_NAMES_3CLASS, plots_dir)
    plot_roc_pr_curves(data["y_test"], y_prob, plots_dir)
    plot_per_class_heatmap(metrics, plots_dir)
    return metrics, y_pred, y_prob


def main():
    n_threads = getattr(config, "TORCH_NUM_THREADS", 2)
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
        
    logger.info(" Starting GAN + CTGAN + DistilBERT IoT Anomaly Detection Pipeline")
    logger.info(" Device: %s | CPU threads: %d", device, n_threads)

    data = load_and_preprocess(config)
    class_names = config.CLASS_NAMES_3CLASS
    feature_names = data["feature_names"]

    # ── Class balance validation ────────────────────────────────────────────────
    # Verify that both Botnet (1) and Malware (2) appear in train AND test.
    # With SAMPLE_SIZE=50 this check would have caught the single-class problem.
    for split_name, y_split in [("train", data["y_train"]), ("test", data["y_test"])]:
        unique, counts = np.unique(y_split, return_counts=True)
        dist_str = "  ".join(f"{class_names[u]}={c}" for u, c in zip(unique, counts))
        logger.info(" Class distribution in %s: %s", split_name, dist_str)
        for cls_idx, cls_name in enumerate(class_names):
            if cls_idx not in unique:
                logger.warning(
                    "\u26a0\ufe0f  Class '%s' (idx=%d) is MISSING from %s split! "
                    "Model cannot learn to detect this attack type. "
                    "Increase SAMPLE_SIZE or use combined_dataset.csv.",
                    cls_name, cls_idx, split_name
                )

    # Before GAN Class Distribution
    before_counts = [int((data["y_train"] == 0).sum()),
                     int((data["y_train"] == 1).sum()),
                     int((data["y_train"] == 2).sum())]

    # --- Baseline Tabular Classifier (no augmentation) ---
    logger.info(" Training Baseline Tabular MLP...")
    baseline_model = build_classifier(config, data["n_features"], device)
    baseline_history = train_classifier(
        config, baseline_model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        synth_X=np.empty((0, data["n_features"])),
        synth_y=np.empty((0,), dtype=np.int64),
        device=device,
        model_filename="classifier_baseline.pt",
        models_dir=config.MODELS_DIR,
    )
    plot_training_history(baseline_history, config.MODELS_DIR)
    baseline_metrics, _, _ = _evaluate_classifier(baseline_model, data, device, config.MODELS_DIR)

    # --- Simple GAN branch ---
    logger.info(" [Simple GAN] Training Class Generators...")
    gan_results = train_gan(config, data["X_train"], data["y_train"], device)
    plot_gan_losses(gan_results["g_losses"], gan_results["d_losses"], config.SIMPLE_GAN_PLOTS_DIR)

    # Visualise PCA and balanced distribution for Simple GAN
    plot_synthetic_distribution_pca(data["X_train"][data["y_train"] != 0], gan_results["synth_X"], config.SIMPLE_GAN_PLOTS_DIR)
    
    y_train_aug_gan = np.concatenate([data["y_train"], gan_results["synth_y"]])
    after_counts_gan = [int((y_train_aug_gan == 0).sum()), 
                        int((y_train_aug_gan == 1).sum()), 
                        int((y_train_aug_gan == 2).sum())]
    plot_class_distribution(before_counts, after_counts_gan, class_names, config.SIMPLE_GAN_PLOTS_DIR)

    logger.info(" [Simple GAN] Training augmented tabular classifier...")
    gan_model = build_classifier(config, data["n_features"], device)
    gan_clf_history = train_classifier(
        config, gan_model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        synth_X=gan_results["synth_X"],
        synth_y=gan_results["synth_y"],
        device=device,
        model_filename="classifier_gan.pt",
        models_dir=config.SIMPLE_GAN_MODELS_DIR,
    )
    plot_training_history(gan_clf_history, config.SIMPLE_GAN_PLOTS_DIR)
    simple_gan_metrics, _, _ = _evaluate_classifier(gan_model, data, device, config.SIMPLE_GAN_PLOTS_DIR)

    # --- CTGAN branch ---
    logger.info(" [CTGAN] Training Class Generators...")
    ctgan_results = train_ctgan(config, data["X_train"], data["y_train"], data["feature_names"])
    plot_gan_losses(ctgan_results["g_losses"], ctgan_results["d_losses"], config.CTGAN_PLOTS_DIR)

    # Visualise PCA and balanced distribution for CTGAN
    plot_synthetic_distribution_pca(data["X_train"][data["y_train"] != 0], ctgan_results["synth_X"], config.CTGAN_PLOTS_DIR)
    
    y_train_aug_ctgan = np.concatenate([data["y_train"], ctgan_results["synth_y"]])
    after_counts_ctgan = [int((y_train_aug_ctgan == 0).sum()), 
                          int((y_train_aug_ctgan == 1).sum()), 
                          int((y_train_aug_ctgan == 2).sum())]
    plot_class_distribution(before_counts, after_counts_ctgan, class_names, config.CTGAN_PLOTS_DIR)

    logger.info(" [CTGAN] Training augmented tabular classifier...")
    ctgan_model = build_classifier(config, data["n_features"], device)
    ctgan_clf_history = train_classifier(
        config, ctgan_model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        synth_X=ctgan_results["synth_X"],
        synth_y=ctgan_results["synth_y"],
        device=device,
        model_filename="classifier_ctgan.pt",
        models_dir=config.CTGAN_MODELS_DIR,
    )
    plot_training_history(ctgan_clf_history, config.CTGAN_PLOTS_DIR)
    ctgan_metrics, _, _ = _evaluate_classifier(ctgan_model, data, device, config.CTGAN_PLOTS_DIR)

    # --- DistilBERT Embeddings Extraction ---
    logger.info(" Extracting DistilBERT features for hybrid multimodal network...")
    train_texts = [features_to_text(x, feature_names) for x in data["X_train"]]
    val_texts   = [features_to_text(x, feature_names) for x in data["X_val"]]
    test_texts  = [features_to_text(x, feature_names) for x in data["X_test"]]

    X_train_emb = get_distilbert_embeddings(config, train_texts, device)
    X_val_emb   = get_distilbert_embeddings(config, val_texts, device)
    X_test_emb  = get_distilbert_embeddings(config, test_texts, device)

    # --- Hybrid Multimodal Baseline ---
    logger.info(" Training Hybrid Baseline Model...")
    hybrid_baseline_model = HybridFusionClassifier(tabular_dim=data["n_features"]).to(device)
    hybrid_base_history = train_hybrid_classifier(
        config, hybrid_baseline_model,
        data["X_train"], X_train_emb, data["y_train"],
        data["X_val"], X_val_emb, data["y_val"],
        device=device, model_filename="hybrid_baseline.pt"
    )
    plot_training_history(hybrid_base_history, config.MODELS_DIR)
    y_pred_h_base = predict_hybrid(hybrid_baseline_model, data["X_test"], X_test_emb, device)
    y_prob_h_base = predict_proba_hybrid(hybrid_baseline_model, data["X_test"], X_test_emb, device)
    hybrid_base_metrics = compute_metrics(data["y_test"], y_pred_h_base, y_prob_h_base)

    # --- Hybrid Multimodal + Simple GAN ---
    logger.info(" Training Hybrid Simple GAN Augmented Model...")
    gan_synth_texts = [features_to_text(x, feature_names) for x in gan_results["synth_X"]]
    gan_synth_emb   = get_distilbert_embeddings(config, gan_synth_texts, device)

    X_train_aug_gan = np.vstack([data["X_train"], gan_results["synth_X"]])
    X_train_emb_aug_gan = np.vstack([X_train_emb, gan_synth_emb])
    y_train_aug_gan = np.concatenate([data["y_train"], gan_results["synth_y"]])

    hybrid_gan_model = HybridFusionClassifier(tabular_dim=data["n_features"]).to(device)
    hybrid_gan_history = train_hybrid_classifier(
        config, hybrid_gan_model,
        X_train_aug_gan, X_train_emb_aug_gan, y_train_aug_gan,
        data["X_val"], X_val_emb, data["y_val"],
        device=device, model_filename="hybrid_gan.pt"
    )
    plot_training_history(hybrid_gan_history, config.SIMPLE_GAN_PLOTS_DIR)
    y_pred_h_gan = predict_hybrid(hybrid_gan_model, data["X_test"], X_test_emb, device)
    y_prob_h_gan = predict_proba_hybrid(hybrid_gan_model, data["X_test"], X_test_emb, device)
    hybrid_gan_metrics = compute_metrics(data["y_test"], y_pred_h_gan, y_prob_h_gan)

    # --- Hybrid Multimodal + CTGAN ---
    logger.info(" Training Hybrid CTGAN Augmented Model...")
    ctgan_synth_texts = [features_to_text(x, feature_names) for x in ctgan_results["synth_X"]]
    ctgan_synth_emb   = get_distilbert_embeddings(config, ctgan_synth_texts, device)

    X_train_aug_ctgan = np.vstack([data["X_train"], ctgan_results["synth_X"]])
    X_train_emb_aug_ctgan = np.vstack([X_train_emb, ctgan_synth_emb])
    y_train_aug_ctgan = np.concatenate([data["y_train"], ctgan_results["synth_y"]])

    hybrid_ctgan_model = HybridFusionClassifier(tabular_dim=data["n_features"]).to(device)
    hybrid_ctgan_history = train_hybrid_classifier(
        config, hybrid_ctgan_model,
        X_train_aug_ctgan, X_train_emb_aug_ctgan, y_train_aug_ctgan,
        data["X_val"], X_val_emb, data["y_val"],
        device=device, model_filename="hybrid_ctgan.pt"
    )
    plot_training_history(hybrid_ctgan_history, config.CTGAN_PLOTS_DIR)
    y_pred_h_ctgan = predict_hybrid(hybrid_ctgan_model, data["X_test"], X_test_emb, device)
    y_prob_h_ctgan = predict_proba_hybrid(hybrid_ctgan_model, data["X_test"], X_test_emb, device)
    hybrid_ctgan_metrics = compute_metrics(data["y_test"], y_pred_h_ctgan, y_prob_h_ctgan)

    # --- Comparison Table ---
    comparison_results = {
        "Tabular Baseline (MLP)": baseline_metrics,
        "Tabular Simple GAN + MLP": simple_gan_metrics,
        "Tabular CTGAN + MLP": ctgan_metrics,
        "Hybrid Baseline": hybrid_base_metrics,
        "Hybrid Simple GAN + DistilBERT": hybrid_gan_metrics,
        "Hybrid CTGAN + DistilBERT": hybrid_ctgan_metrics,
    }
    comparison_df = build_comparison_table(comparison_results)
    plot_metric_comparison(comparison_df, config.COMPARISON_REPORTS_DIR)
    plot_metric_comparison(comparison_df, config.PLOTS_DIR)

    save_gan_ctgan_comparison_report(
        comparison_df, simple_gan_metrics, ctgan_metrics,
        config.COMPARISON_REPORTS_DIR,
        config.SIMPLE_GAN_PLOTS_DIR,
        config.CTGAN_PLOTS_DIR,
    )
    save_augmentation_comparison_json(
        comparison_df, simple_gan_metrics, ctgan_metrics, config.COMPARISON_REPORTS_DIR,
    )

    # Choose best Hybrid model for report & explanations
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
    plot_confusion_matrix(cm, class_names, config.PLOTS_DIR)
    plot_roc_pr_curves(data["y_test"], final_y_prob, config.PLOTS_DIR)
    plot_per_class_heatmap(final_metrics, config.PLOTS_DIR)

    # Copy GAN plots from the best branch to the main plots dir
    import shutil
    best_plots_src = config.CTGAN_PLOTS_DIR if best_hybrid == "ctgan" else config.SIMPLE_GAN_PLOTS_DIR
    for filename in [
        "gan_loss_curves.png",
        "pca_synthetic.png",
        "class_distribution_comparison.png",
        "classifier_training_history.png",  # Training curves (fixed: now shows 50 epochs)
    ]:
        src_file = os.path.join(best_plots_src, filename)
        dst_file = os.path.join(config.PLOTS_DIR, filename)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)

    # Compute attack breakdown for the final model
    logger.info(" Computing attack-type detection breakdown...")
    attack_metrics = compute_attack_type_metrics(data["y_test"], final_y_pred, data["att_test"])
    plot_attack_type_breakdown(attack_metrics, config.PLOTS_DIR)

    # Initialize Explainer
    logger.info(" Initializing Local Hybrid DistilBERT Explainer...")
    explainer = HybridAnomalyExplainer(config)
    explanations = explainer.generate_explanations(
        data["X_test"], final_y_pred, final_y_prob, data["att_test"],
        data["feature_names"], n_samples=config.NUM_EXPLAIN_SAMPLES,
    )

    report_path = save_explanation_report(
        explanations, final_metrics, comparison_df, config.REPORTS_DIR,
        attack_metrics=attack_metrics,
        optimal_threshold=0.5,
        threshold_score_table=[],
    )
    save_metrics_json(final_metrics, comparison_df, explanations, config.REPORTS_DIR, attack_metrics=attack_metrics, optimal_threshold=0.5)

    logger.info(" Pipeline complete!")
    logger.info(" Reports generated in results/reports/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(" Pipeline failed: %s", e, exc_info=True)
