# Documentation Index

Complete documentation for the **GAN + BERT Hybrid IoT Anomaly Detection** project.

## Quick links

| Document | What it covers |
|----------|----------------|
| [01-PROJECT-OVERVIEW.md](01-PROJECT-OVERVIEW.md) | Goals, architecture, end-to-end pipeline, folder structure |
| [02-SETUP-AND-USAGE.md](02-SETUP-AND-USAGE.md) | Virtual environment, install, run, troubleshooting |
| [03-DATA-AND-PREPROCESSING.md](03-DATA-AND-PREPROCESSING.md) | Dataset, labels, cleaning, scaling, train/val/test splits |
| [04-GAN.md](04-GAN.md) | Simple GAN (not CT-GAN), architecture, training, synthetic data |
| [05-CLASSIFIER.md](05-CLASSIFIER.md) | MLP classifier, baseline vs GAN-augmented training |
| [06-BERT-EXPLAINABILITY.md](06-BERT-EXPLAINABILITY.md) | Text conversion, BERT fine-tuning, human-readable explanations |
| [07-EVALUATION-AND-REPORTS.md](07-EVALUATION-AND-REPORTS.md) | Metrics, plots, HTML report, JSON output |
| [08-CONFIGURATION-REFERENCE.md](08-CONFIGURATION-REFERENCE.md) | Every setting in `config.py` explained |
| [09-MAIN-PIPELINE.md](09-MAIN-PIPELINE.md) | Step-by-step walkthrough of `main.py` |
| [10-GAN-VS-CTGAN.md](10-GAN-VS-CTGAN.md) | Simple GAN vs CTGAN separate results and comparison report |

## One-line summary

This project detects **malicious IoT network traffic** (FTP/SSH brute-force) using a **GAN-augmented neural classifier**, then produces **BERT-style natural language explanations** for flagged flows.

## Run the project

```bash
cd /path/to/ANOMOLY_DETECTION_FOR_IOT
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

See [02-SETUP-AND-USAGE.md](02-SETUP-AND-USAGE.md) for details.
