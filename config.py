import os
from dotenv import load_dotenv

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
# Load environment variables from .env
load_dotenv(os.path.join(BASE_DIR, ".env"))

DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH   = os.path.join(BASE_DIR, "combined_dataset.csv")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR     = os.path.join(RESULTS_DIR, "reports")
MODELS_DIR      = os.path.join(RESULTS_DIR, "models")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

# Separate outputs: Simple GAN vs CTGAN (for side-by-side comparison)
SIMPLE_GAN_DIR          = os.path.join(RESULTS_DIR, "simple_gan")
SIMPLE_GAN_PLOTS_DIR    = os.path.join(SIMPLE_GAN_DIR, "plots")
SIMPLE_GAN_MODELS_DIR   = os.path.join(SIMPLE_GAN_DIR, "models")

CTGAN_DIR               = os.path.join(RESULTS_DIR, "ctgan")
CTGAN_PLOTS_DIR         = os.path.join(CTGAN_DIR, "plots")
CTGAN_MODELS_DIR        = os.path.join(CTGAN_DIR, "models")

COMPARISON_REPORTS_DIR  = os.path.join(REPORTS_DIR, "comparison")

LABEL_COL       = "Label"
# 5000 rows gives a statistically meaningful train/test split with all attack types.
# Set to None to use the FULL 375MB combined_dataset.csv (takes ~1-2 hrs on CPU).
SAMPLE_SIZE     = 5000
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
VAL_SIZE        = 0.10

# ── Class Balancing ───────────────────────────────────────────────────
# Cap Benign rows to MAX_BENIGN_RATIO × the LARGEST minority class count.
# Without this, combined_dataset.csv has a 94:4:1 Benign:Botnet:Malware ratio
# which causes models to get 94%+ accuracy simply by predicting everything Benign.
# Setting MAX_BENIGN_RATIO=3 forces the model to actually learn attack patterns.
MAX_BENIGN_RATIO = 3

# NOTE: The preprocessor now auto-detects ALL non-Benign rows as anomaly (label=1).
# This LABEL_MAP is kept for reference / legacy compatibility only.
# It no longer filters rows — any CSV attack type will be detected.
LABEL_MAP = {
    # 02-14-2018.csv
    "Benign":                   0,
    "FTP-BruteForce":           1,
    "SSH-Bruteforce":           1,
    # 02-15-2018.csv
    "DoS attacks-GoldenEye":    1,
    "DoS attacks-Slowloris":    1,
    "DoS attacks-Hulk":         1,
    "DoS attacks-SlowHTTPTest": 1,
    # Generic fallback — all non-Benign rows map to 1 in the preprocessor
}

CLASS_NAMES  = ["Benign", "Anomaly"]
ATTACK_TYPES = [
    # 02-14-2018.csv
    "FTP-BruteForce", "SSH-Bruteforce",
    # 02-15-2018.csv
    "DoS attacks-GoldenEye", "DoS attacks-Slowloris",
    "DoS attacks-Hulk", "DoS attacks-SlowHTTPTest",
]

# ── GAN (WGAN-GP) ──────────────────────────────────────────────────────────────
# Switched to WGAN-GP to fix mode collapse observed in previous runs.
# The old BCE GAN had Generator loss diverging to 4.0 while Discriminator
# collapsed to 0 (textbook mode collapse). WGAN-GP is mathematically stable.
LATENT_DIM      = 64
# WGAN-GP converges faster than BCE GAN — 50 epochs is sufficient.
GAN_EPOCHS      = 50
GAN_BATCH_SIZE  = 256
GAN_LR_G        = 1e-4   # lower LR for WGAN-GP stability
GAN_LR_D        = 1e-4
GAN_BETAS       = (0.0, 0.9)  # WGAN-GP recommended betas (no momentum)
GAN_SYNTHETIC_SAMPLES = 10_000
# Train discriminator (critic) N_CRITIC times per generator step.
# Standard WGAN-GP practice: D sees more data, converges before G overfits.
GAN_N_CRITIC    = 5
# Gradient penalty weight — standard WGAN-GP value
GAN_GP_LAMBDA   = 10.0

# ── GAN quality gate ──────────────────────────────────────────────────────────────
# WGAN-GP critic outputs are unbounded real values (not 0-1 probabilities).
# Threshold must be 0.0 so all generated samples are kept (filtered by
# Mahalanobis distance in post-processing instead).
DISC_QUALITY_THRESHOLD = 0.0

# ── CTGAN ──────────────────────────────────────────────────────────────────────
# CTGAN (WGAN-GP) also requires many epochs. Negative loss values are NORMAL
# for Wasserstein GANs — they represent the critic score, not BCE.
CTGAN_EPOCHS            = 100
CTGAN_BATCH_SIZE        = 500
CTGAN_LR                = 2e-4
CTGAN_PAC               = 10
CTGAN_SYNTHETIC_SAMPLES = 10_000
CTGAN_MAX_TRAIN_ROWS    = None    # None = use all anomaly rows
# CTGAN quality gate: discard synthetic samples whose Mahalanobis distance
# from the real anomaly distribution exceeds this percentile.
CTGAN_MAHAL_PERCENTILE  = 95

# ── Classifier ─────────────────────────────────────────────────────────────────
# 50 epochs gives the OneCycleLR scheduler enough steps to warm up and
# converge. 2 epochs produced only 2 data points on the training history plot
# (flatline), and the model never moved past random initialization accuracy.
CLF_EPOCHS       = 50
CLF_BATCH_SIZE   = 256          # smaller batches → better gradient signal
CLF_LR           = 3e-4        # stable convergence
CLF_HIDDEN       = [512, 256, 128, 64]
CLF_DROPOUT      = 0.4
CLF_WEIGHT_DECAY = 1e-4
# Early stopping patience: must be > OneCycleLR warmup period.
# With pct_start=0.1 and 50 epochs, warmup = 5 epochs. Patience=15 ensures
# the model trains well past warmup before stopping.
CLF_EARLY_STOP_PATIENCE = 15

# Label smoothing prevents the model from becoming overconfident (100% confidence).
# Smoothing=0.1 means the target distribution is 90% correct class + 10% spread.
# This produced calibrated probabilities (e.g., 82%) instead of always 100%.
LABEL_SMOOTHING  = 0.1

# OneCycleLR warmup fraction: 10% = 5 epochs warm-up out of 50 total.
# Previous setting (30%) caused early stopping to fire before warmup finished.
CLF_PCT_START    = 0.1

# Focal Loss parameters (kept for reference)
FOCAL_GAMMA      = 2.0
FOCAL_ALPHA      = 0.75

# ── Threshold search ───────────────────────────────────────────────────────────
# After training, sweep thresholds on the validation set to find the one that
# maximises F1 (or balances precision/recall). This replaces the hard-coded 0.5
# and stops the model from flagging ALL traffic as anomaly.
THRESHOLD_SEARCH_MIN   = 0.05
THRESHOLD_SEARCH_MAX   = 0.95
THRESHOLD_SEARCH_STEP  = 0.01
THRESHOLD_METRIC       = "accuracy"   # "f1" | "precision" | "recall" | "balanced_accuracy" | "accuracy"

# ── 3-Class Configuration ──────────────────────────────────────────────────────
CLASS_MAP_3CLASS = {
    "Benign": 0,
    "Botnet": 1,
    "Malware": 2
}
CLASS_NAMES_3CLASS = ["Benign", "Botnet", "Malware"]

# ── DistilBERT & Hybrid Model ──────────────────────────────────────────────────
BERT_MODEL          = "distilbert-base-uncased"
BERT_MAX_LEN        = 64
BERT_BATCH_SIZE     = 64
BERT_LR             = 2e-5
BERT_EPOCHS         = 1
FREEZE_BERT         = True       # Frozen backbone is extremely fast on CPU
# Generate explanations for samples from EACH class (Benign, Botnet, Malware)
NUM_EXPLAIN_SAMPLES = 6

# ── PCA & t-SNE Plotting ───────────────────────────────────────────────────────
PCA_SAMPLE_SIZE     = 1000       # Number of samples to plot in PCA/t-SNE for speed

# ── OpenAI Explainability ──────────────────────────────────────────────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL        = "gpt-4o-mini"

# ── System ─────────────────────────────────────────────────────────────────────
TORCH_NUM_THREADS = 2
PROCESS_NICE      = 10

METRICS         = ["accuracy", "precision", "recall", "f1", "roc_auc"]

FIGURE_DPI      = 150
PALETTE         = {
    "primary":      "#6C63FF",
    "secondary":    "#FF6584",
    "success":      "#43AA8B",
    "warning":      "#F9C74F",
    "danger":       "#F94144",
    "background":   "#0F0F1A",
    "surface":      "#1A1A2E",
    "text":         "#E0E0FF",
}