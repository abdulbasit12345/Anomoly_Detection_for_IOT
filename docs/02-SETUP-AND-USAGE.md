# Setup and Usage

## Requirements

- **Python 3.9+** (tested with 3.12)
- **~4–8 GB RAM** minimum (more helps for BERT)
- **Disk space:** ~2–3 GB for Python packages + BERT model download
- **Dataset:** `02-14-2018.csv` must be in the project root (already included)

---

## Install dependencies

### Option A: Standard virtual environment

```bash
cd ANOMOLY_DETECTION_FOR_IOT
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Reuse Anaconda packages (faster if you already have PyTorch)

```bash
/opt/anaconda3/bin/python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install transformers tqdm joblib   # only if missing
```

### What `requirements.txt` installs

| Package | Role |
|---------|------|
| `numpy`, `pandas` | Data arrays and tables |
| `scikit-learn` | Scaling, splits, metrics |
| `torch`, `torchvision`, `torchaudio` | Neural networks (vision/audio not used directly) |
| `transformers` | BERT tokenizer and model |
| `matplotlib`, `seaborn` | Plots |
| `joblib` | Save scaler |
| `tqdm` | Progress bars |

---

## Run the full pipeline

```bash
source .venv/bin/activate
python main.py
```

### What happens when you run

1. Creates a log file: `logs/run_YYYYMMDD_HHMMSS.log`
2. Prints progress to terminal (tqdm bars for training)
3. Writes artifacts under `data/processed/`, `results/models/`, `results/plots/`, `results/reports/`

### View results

Open in a browser:

```text
results/reports/explanation_report.html
```

Raw metrics JSON:

```text
results/reports/results.json
```

---

## Monitor a long run

```bash
tail -f logs/run_*.log
```

Check if process is running:

```bash
pgrep -fl main.py
```

---

## Speed up for testing

Edit `config.py`:

```python
SAMPLE_SIZE = 50_000      # default 200_000
GAN_EPOCHS = 50           # default 200
CLF_EPOCHS = 15           # default 30
BERT_EPOCHS = 1           # default 3
NUM_EXPLAIN_SAMPLES = 5   # default 10
```

Smaller data + fewer epochs can finish in **15–30 minutes** on CPU instead of 1–2 hours.

---

## Troubleshooting

### `ReduceLROnPlateau ... unexpected keyword argument 'verbose'`

Older PyTorch allowed `verbose=True` on schedulers; newer versions removed it. The current `train_classifier.py` in this repo does **not** use `verbose` — if you see this error, update `train_classifier.py` or upgrade/downgrade PyTorch to match the code.

### `ModuleNotFoundError: No module named 'torch'`

Activate the venv and install requirements:

```bash
source .venv/bin/activate
pip install torch transformers
```

### `FileNotFoundError` for CSV

Ensure `02-14-2018.csv` is in the project root. Path is set in `config.py` as `RAW_DATA_PATH`.

### BERT download fails

First run downloads `bert-base-uncased` from Hugging Face (~400 MB). Needs internet. Set proxy if required:

```bash
export HF_ENDPOINT=https://hf-mirror.com   # example mirror
```

### Run stops at GAN epoch 60 / process killed

Long CPU jobs may be killed if the terminal closes. Use `nohup` or `screen`:

```bash
nohup python main.py > run.out 2>&1 &
```

### Low baseline accuracy (~36%)

The baseline often predicts mostly **Anomaly** (high recall, low benign precision) due to class imbalance and threshold 0.5. The GAN-augmented model is intended to improve this — check final metrics in the report.

---

## Configuration

All tunable values are in `config.py`. See [08-CONFIGURATION-REFERENCE.md](08-CONFIGURATION-REFERENCE.md).

---

## Logs

Each run appends to a new file:

```text
logs/run_20260524_232434.log
```

Log levels: `INFO` for milestones, `ERROR` with traceback if `main.py` crashes.
