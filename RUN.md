# IoT Anomaly Detection — Run Guide

---

## 1. Install Dependencies

**macOS (Conda):**
```bash
pip install -r requirements.txt
pip install ctgan --no-deps
pip install rdt tqdm
```

**macOS (Virtual Environment):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ctgan --no-deps
pip install rdt tqdm
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install ctgan --no-deps
pip install rdt tqdm
```

---

## 2. Run

```bash
python main.py
```

---

## 3. Results

All output is saved to the `results/` folder:

```
results/
├── plots/                        ← All charts (confusion matrix, ROC, training history, etc.)
├── reports/
│   ├── explanation_report.html   ← Open this in your browser (main dashboard)
│   └── comparison/
│       └── gan_vs_ctgan_comparison.html
├── models/                       ← Saved model weights (.pt files)
├── simple_gan/                   ← Simple GAN plots and models
└── ctgan/                        ← CTGAN plots and models
```

Open `results/reports/explanation_report.html` in any browser to see all detections and metrics.

---

## 4. Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'ctgan'` | `pip install ctgan --no-deps && pip install rdt tqdm` |
| `No module named 'torch'` | `pip install torch torchvision torchaudio` |
| `FileNotFoundError: combined_dataset.csv` | Place `combined_dataset.csv` in the project root |
| `source: no such file or directory: .venv/bin/activate` | Run `python3 -m venv .venv` first |
| `.venv\Scripts\activate` fails on PowerShell | Use: `.venv\Scripts\Activate.ps1` |
