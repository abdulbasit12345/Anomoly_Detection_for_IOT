# Simple GAN vs CTGAN Comparison

The pipeline now trains **two augmentation methods** and stores results separately.

## Output folders

| Path | Contents |
|------|----------|
| `results/simple_gan/plots/` | Charts for vanilla GAN + classifier |
| `results/simple_gan/models/` | `generator.pt`, `discriminator.pt`, `classifier_gan.pt` |
| `results/ctgan/plots/` | Charts for CTGAN + classifier |
| `results/ctgan/models/` | `ctgan_model.pkl`, `classifier_ctgan.pt` |
| `results/reports/comparison/` | Side-by-side comparison report |

## Comparison reports

- **HTML:** `results/reports/comparison/gan_vs_ctgan_comparison.html`
- **JSON:** `results/reports/comparison/gan_vs_ctgan_metrics.json`

## Models compared

1. **Baseline (MLP)** — no synthetic data  
2. **Simple GAN + MLP** — vanilla GAN synthetic anomalies  
3. **CTGAN + MLP** — CTGAN synthetic anomalies (mode-specific normalization, conditional sampling)

## Config (`config.py`)

```python
CTGAN_EPOCHS = 25
CTGAN_SYNTHETIC_SAMPLES = 1000
CTGAN_MAX_TRAIN_ROWS = 8000   # subsample for speed
GAN_SYNTHETIC_SAMPLES = 1000  # keep equal for fair comparison
```

## Run

```bash
source .venv/bin/activate
pip install ctgan threadpoolctl
python main.py
```

## View comparison

```bash
open results/reports/comparison/gan_vs_ctgan_comparison.html
```
