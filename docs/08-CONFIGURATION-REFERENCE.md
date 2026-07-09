# Configuration Reference

**File:** `config.py` (project root)

All paths are built from `BASE_DIR` = directory containing `config.py`.

---

## Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_DIR` | auto | Project root |
| `DATA_DIR` | `data/` | Data storage |
| `RAW_DATA_PATH` | `02-14-2018.csv` | Input CSV |
| `PROCESSED_DIR` | `data/processed/` | Scaler + numpy arrays |
| `RESULTS_DIR` | `results/` | All outputs |
| `PLOTS_DIR` | `results/plots/` | PNG figures |
| `REPORTS_DIR` | `results/reports/` | HTML + JSON |
| `MODELS_DIR` | `results/models/` | `.pt` and BERT folder |
| `LOGS_DIR` | `logs/` | Run logs |

---

## Data and splits

| Variable | Default | Description |
|----------|---------|-------------|
| `LABEL_COL` | `"Label"` | CSV column for attack name |
| `SAMPLE_SIZE` | `200_000` | Max rows after stratified sample; `None` = use all |
| `RANDOM_STATE` | `42` | Reproducibility seed |
| `TEST_SIZE` | `0.20` | Fraction for test set |
| `VAL_SIZE` | `0.10` | Fraction of **original** data for validation |
| `LABEL_MAP` | Benign→0, attacks→1 | Which labels to keep |
| `CLASS_NAMES` | `["Benign", "Anomaly"]` | Plot/report labels |
| `ATTACK_TYPES` | FTP + SSH brute-force | Documentation list |

---

## GAN hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `LATENT_DIM` | `64` | Noise vector size for generator |
| `GAN_EPOCHS` | `200` | **Largest time cost** — training epochs |
| `GAN_BATCH_SIZE` | `256` | Minibatch size |
| `GAN_LR_G` | `2e-4` | Generator learning rate |
| `GAN_LR_D` | `2e-4` | Discriminator learning rate |
| `GAN_BETAS` | `(0.5, 0.999)` | Adam betas |
| `GAN_SYNTHETIC_SAMPLES` | `5000` | Fake anomalies added to classifier train set |

---

## Classifier hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `CLF_EPOCHS` | `30` | Max epochs (early stop may end sooner) |
| `CLF_BATCH_SIZE` | `512` | Minibatch size |
| `CLF_LR` | `1e-3` | AdamW learning rate |
| `CLF_HIDDEN` | `[256, 128, 64]` | MLP hidden layer sizes |
| `CLF_DROPOUT` | `0.3` | Dropout rate |
| `CLF_WEIGHT_DECAY` | `1e-4` | L2 penalty |

---

## BERT hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `BERT_MODEL` | `bert-base-uncased` | Hugging Face model id |
| `BERT_MAX_LEN` | `128` | Max tokens per flow text |
| `BERT_BATCH_SIZE` | `16` | Fine-tune batch size |
| `BERT_EPOCHS` | `3` | Fine-tune epochs |
| `BERT_LR` | `2e-5` | Typical BERT learning rate |
| `NUM_EXPLAIN_SAMPLES` | `10` | Test flows in HTML report |

---

## Evaluation and plotting

| Variable | Default | Description |
|----------|---------|-------------|
| `METRICS` | accuracy, precision, recall, f1 | Listed for reference |
| `FIGURE_DPI` | `150` | matplotlib DPI |
| `PALETTE` | hex colors | Used in some styling (plots use local constants too) |

---

## Recommended profiles

### Fast debug (~15–30 min CPU)

```python
SAMPLE_SIZE = 20_000
GAN_EPOCHS = 20
GAN_SYNTHETIC_SAMPLES = 1000
CLF_EPOCHS = 10
BERT_EPOCHS = 1
NUM_EXPLAIN_SAMPLES = 3
```

### Balanced (~45–60 min CPU)

```python
SAMPLE_SIZE = 100_000
GAN_EPOCHS = 50
GAN_SYNTHETIC_SAMPLES = 3000
CLF_EPOCHS = 20
BERT_EPOCHS = 2
```

### Full quality (default, 1–2+ hours CPU)

Keep defaults in `config.py`.

---

## Environment variables (optional)

Not in `config.py` but useful:

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `HF_HOME` | Hugging Face cache directory |
| `TOKENIZERS_PARALLELISM` | Set `false` to avoid fork warnings |

Device is chosen in `main.py`:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
