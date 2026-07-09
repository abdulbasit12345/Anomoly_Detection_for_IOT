import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    balanced_accuracy_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray = None, prefix: str = "") -> dict:

    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    prec     = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec      = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1       = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # False Positive Rate (FPR): proportion of BENIGN samples predicted as Anomaly (Botnet or Malware)
    benign_mask = y_true == 0
    n_benign = int(benign_mask.sum())
    n_fp = int(((y_true == 0) & (y_pred != 0)).sum())
    fpr = n_fp / n_benign if n_benign > 0 else 0.0

    n_pred_anomaly = int((y_pred != 0).sum())
    n_true_anomaly = int((y_true != 0).sum())

    metrics = {
        f"{prefix}accuracy":          acc,
        f"{prefix}balanced_accuracy": bal_acc,
        f"{prefix}precision":         prec,
        f"{prefix}recall":            rec,
        f"{prefix}f1_score":          f1,
        f"{prefix}n_predicted_anomaly": n_pred_anomaly,
        f"{prefix}n_true_anomaly":      n_true_anomaly,
        f"{prefix}false_positive_rate": round(fpr, 4),
    }

    if y_proba is not None:
        try:
            # Multi-class ROC-AUC using One-vs-Rest (ovr) strategy
            metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except Exception:
            pass

    prec_pc = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    rec_pc  = recall_score(y_true, y_pred,    average=None, labels=[0, 1, 2], zero_division=0)
    f1_pc   = f1_score(y_true, y_pred,         average=None, labels=[0, 1, 2], zero_division=0)
    
    class_names = ["Benign", "Botnet", "Malware"]
    for i, cls in enumerate(class_names):
        metrics[f"{prefix}{cls}_precision"] = prec_pc[i] if i < len(prec_pc) else 0.0
        metrics[f"{prefix}{cls}_recall"]    = rec_pc[i]  if i < len(rec_pc)  else 0.0
        metrics[f"{prefix}{cls}_f1"]        = f1_pc[i]   if i < len(f1_pc)   else 0.0

    logger.info(" %sMetrics:", prefix or "")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("   %-40s %.4f", k, v)
        else:
            logger.info("   %-40s %s", k, v)

    return metrics


def compute_attack_type_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 att_labels: np.ndarray) -> dict:
    """
    Break down detection results per actual attack type.
    """
    results = {}

    per_attack = []
    total_attacks = 0
    total_detected = 0

    attack_mask = y_true != 0
    for attack_name in sorted(set(att_labels[attack_mask])):
        if attack_name == "Benign":
            continue
        mask = (att_labels == attack_name)
        n_total   = int(mask.sum())
        
        # Flagged as any anomaly (prediction != 0)
        n_detected = int((y_pred[mask] != 0).sum())
        n_missed  = n_total - n_detected
        det_rate  = n_detected / n_total if n_total > 0 else 0.0

        per_attack.append({
            "attack_type":      attack_name,
            "total_in_test":    n_total,
            "true_positives":   n_detected,
            "false_negatives":  n_missed,
            "detection_rate":   round(det_rate, 4),
        })
        total_attacks  += n_total
        total_detected += n_detected

        logger.info(
            "   [Attack] %-25s  %d/%d detected  (%.1f%%)",
            attack_name, n_detected, n_total, det_rate * 100,
        )

    # False positive summary
    benign_mask  = y_true == 0
    n_benign     = int(benign_mask.sum())
    n_fp         = int((y_pred[benign_mask] != 0).sum())
    fp_rate      = n_fp / n_benign if n_benign > 0 else 0.0

    results["per_attack"]              = per_attack
    results["total_attacks_in_test"]   = total_attacks
    results["total_detected"]          = total_detected
    results["overall_detection_rate"]  = round(total_detected / total_attacks, 4) if total_attacks > 0 else 0.0
    results["false_positives"]         = n_fp
    results["n_benign_in_test"]        = n_benign
    results["false_positive_rate"]     = round(fp_rate, 4)

    logger.info(
        " Attack Summary: %d/%d detected (%.1f%%) | FP: %d/%d benign (%.1f%%)",
        total_detected, total_attacks,
        100.0 * results["overall_detection_rate"],
        n_fp, n_benign, 100.0 * fp_rate,
    )
    return results


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: list = None) -> str:
    target_names = class_names or ["Benign", "Botnet", "Malware"]
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def build_comparison_table(results_dict: dict) -> pd.DataFrame:

    rows = []
    for model_name, metrics in results_dict.items():
        row = {"Model": model_name}
        for col in ["accuracy", "balanced_accuracy", "precision", "recall",
                    "f1_score", "roc_auc"]:
            row[col.replace("_", " ").title()] = f"{metrics.get(col, 0.0):.4f}"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    logger.info("\n Model Comparison Table:\n%s", df.to_string())
    return df