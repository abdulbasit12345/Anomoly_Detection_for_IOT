# Evaluation and Reports

**Metrics:** `src/evaluation/metrics.py`  
**Plots & reports:** `src/evaluation/visualizer.py`

---

## Metrics computed (`compute_metrics`)

### Binary metrics (overall)

| Metric | Description |
|--------|-------------|
| `accuracy` | Fraction of correct predictions |
| `precision` | TP / (TP + FP) for anomaly class |
| `recall` | TP / (TP + FN) |
| `f1_score` | Harmonic mean of precision and recall |
| `roc_auc` | Area under ROC (needs `y_proba`) |
| `pr_auc` | Area under precision-recall curve |

### Per-class metrics

For `Benign` and `Anomaly`:

- `{Class}_precision`  
- `{Class}_recall`  
- `{Class}_f1`  

### Prefix for baseline

Baseline run uses `prefix="base_"` → keys like `base_accuracy`, `base_f1_score`.

Final model uses no prefix → `accuracy`, `f1_score`, etc.

---

## Confusion matrix

`get_confusion_matrix(y_true, y_pred)` → 2×2 array:

```text
                Predicted
              Benign  Anomaly
Actual Benign    TN      FP
       Anomaly   FN      TP
```

Plotted with counts and row percentages → `confusion_matrix.png`.

---

## Model comparison table

`build_comparison_table()` builds a DataFrame:

| Model | Accuracy | Precision | Recall | F1 Score | Roc Auc | Pr Auc |
|-------|----------|-----------|--------|----------|---------|--------|
| Baseline (MLP) | … | … | … | … | … | … |
| GAN-Augmented Hybrid | … | … | … | … | … | … |

> Note: Table name says “GAN+BERT” in plot titles, but comparison is **Baseline MLP vs GAN-augmented MLP** — BERT is not a separate row in metrics.

---

## Plots generated

All saved under `results/plots/` with dark theme styling.

| File | Function | Content |
|------|----------|---------|
| `gan_loss_curves.png` | `plot_gan_losses` | Generator vs discriminator loss per epoch |
| `classifier_training_history.png` | `plot_training_history` | Loss, accuracy, F1 train/val |
| `confusion_matrix.png` | `plot_confusion_matrix` | Heatmap with counts |
| `roc_pr_curves.png` | `plot_roc_pr_curves` | ROC and PR curves with AUC |
| `metric_comparison.png` | `plot_metric_comparison` | Bar chart baseline vs final |
| `per_class_heatmap.png` | `plot_per_class_heatmap` | Precision/recall/F1 per class |

### Color palette (from config)

- Primary purple `#6C63FF`  
- Pink `#FF6584`  
- Teal `#43AA8B`  
- Background `#0F0F1A`  

---

## HTML report

**Path:** `results/reports/explanation_report.html`

### Sections

1. **Title** — GAN + BERT Hybrid IoT Anomaly Detection  
2. **Metric cards** — Accuracy, precision, recall, F1, ROC-AUC  
3. **Embedded plots** — relative paths to `../plots/*.png`  
4. **Comparison table** — HTML table from DataFrame  
5. **Explanation cards** — one per sample (badge, confidence, attack type, monospace explanation body)  
6. **Footer** — pipeline credit line  

Open locally:

```bash
open results/reports/explanation_report.html    # macOS
xdg-open results/reports/explanation_report.html  # Linux
```

---

## JSON results

**Path:** `results/reports/results.json`

Structure:

```json
{
  "final_metrics": { "accuracy": 0.0, ... },
  "model_comparison": [ { "Model": "...", ... } ],
  "bert_explanations": [
    {
      "sample_idx": 1234,
      "prediction": "Anomaly",
      "confidence": "87.3%",
      "attack_type": "SSH-Bruteforce",
      "explanation": "..."
    }
  ]
}
```

Useful for dashboards or downstream tools.

---

## Functions called from `main.py`

```python
baseline_metrics = compute_metrics(..., prefix="base_")
final_metrics = compute_metrics(...)
cm = get_confusion_matrix(...)
plot_confusion_matrix(cm, ...)
plot_roc_pr_curves(...)
plot_per_class_heatmap(final_metrics, ...)
comparison_df = build_comparison_table({...})
plot_metric_comparison(comparison_df, ...)
report_path = save_explanation_report(explanations, final_metrics, comparison_df, ...)
save_metrics_json(final_metrics, comparison_df, explanations, ...)
```

---

## Interpreting results

| Observation | Possible meaning |
|-------------|------------------|
| High recall, low precision | Model flags many false positives |
| ROC-AUC ≈ 1.0 with low accuracy | Ranking good but threshold 0.5 wrong |
| GAN loss G up, D down | Discriminator dominating — synthetic quality may be weak |
| Improved F1 after GAN | Synthetic oversampling helped minority class |

Always review **confusion matrix** and **per-class heatmap**, not accuracy alone.
