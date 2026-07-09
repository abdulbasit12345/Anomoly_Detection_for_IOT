import os
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Binary Focal Loss — designed for class-imbalanced datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0 down-weights easy (well-classified) examples, forcing the model
    to focus on hard examples (i.e., the rare anomaly class in IoT traffic).
    alpha weights the positive class to further counteract imbalance.

    Why this replaces BCEWithLogitsLoss:
      BCE treats every sample equally → on highly imbalanced IoT data the model
      learns to predict "Benign" for everything because that trivially minimises
      BCE. Focal Loss penalises confident correct predictions lightly and hard
      misclassifications heavily, pushing the model to actually learn anomalies.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha       # weight for the positive (anomaly) class
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs      = torch.sigmoid(logits)
        bce        = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t        = probs * targets + (1 - probs) * (1 - targets)
        alpha_t    = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_w    = alpha_t * (1 - p_t) ** self.gamma
        loss       = focal_w * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ── Early stopping ─────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 8, delta: float = 1e-4):
        self.patience   = patience
        self.delta      = delta
        self.best_val   = -np.inf
        self.counter    = 0
        self.best_state = None

    def __call__(self, val_metric, model):
        if val_metric > self.best_val + self.delta:
            self.best_val   = val_metric
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ── Training ───────────────────────────────────────────────────────────────────
def train_classifier(config, model, X_train, y_train, X_val, y_val,
                     synth_X: np.ndarray, synth_y: np.ndarray, device: torch.device,
                     model_filename: str = "classifier.pt",
                     models_dir: str = None) -> dict:

    if synth_y is None:
        synth_y = np.ones(len(synth_X), dtype=np.int64)
        
    X_aug   = np.vstack([X_train, synth_X]).astype(np.float32)
    y_aug   = np.concatenate([y_train, synth_y]).astype(np.int64)
    logger.info(" Classifier Training | Augmented train size: %d  (real=%d  synth=%d)",
                len(X_aug), len(X_train), len(synth_X))

    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.tensor(X_aug, dtype=torch.float32),
        torch.tensor(y_aug, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=config.CLF_BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=config.CLF_BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.CLF_LR,
        weight_decay=config.CLF_WEIGHT_DECAY,
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.CLF_LR * 10,
        steps_per_epoch=len(train_loader),
        epochs=config.CLF_EPOCHS,
        pct_start=0.3,
    )
    early_stop = EarlyStopping(patience=8)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "train_f1":   [], "val_f1":   [],
    }

    for epoch in range(1, config.CLF_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        correct, total = 0, 0
        all_tr_preds = []
        all_tr_true = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.CLF_EPOCHS} [Train]", leave=False)
        for Xb, yb in pbar:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            tr_loss += loss.item() * len(Xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
            
            all_tr_preds.extend(preds.cpu().numpy())
            all_tr_true.extend(yb.cpu().numpy())

        tr_loss /= len(train_ds)
        tr_acc  = correct / total
        tr_f1   = f1_score(all_tr_true, all_tr_preds, average="macro", zero_division=0)

        model.eval()
        vl_loss = 0.0
        vl_correct, vl_total = 0, 0
        all_vl_preds = []
        all_vl_true = []
        
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits  = model(Xb)
                loss    = criterion(logits, yb)
                vl_loss += loss.item() * len(Xb)
                preds = logits.argmax(dim=1)
                vl_correct += (preds == yb).sum().item()
                vl_total += len(yb)
                
                all_vl_preds.extend(preds.cpu().numpy())
                all_vl_true.extend(yb.cpu().numpy())

        vl_loss /= len(val_ds)
        vl_acc  = vl_correct / vl_total
        vl_f1   = f1_score(all_vl_true, all_vl_preds, average="macro", zero_division=0)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(vl_f1)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "  [Epoch %2d/%d]  tr_loss=%.4f  vl_loss=%.4f  "
                "tr_acc=%.4f  vl_acc=%.4f  vl_f1_macro=%.4f",
                epoch, config.CLF_EPOCHS,
                tr_loss, vl_loss, tr_acc, vl_acc, vl_f1
            )

        if early_stop(vl_f1, model):
            logger.info("   Early stopping at epoch %d (best val_f1=%.4f)",
                        epoch, early_stop.best_val)
            break

    early_stop.restore(model)

    save_dir = models_dir or config.MODELS_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), save_path)
    logger.info(" Classifier saved to: %s", save_path)

    return history


# ── Optimal Threshold Search (Stubbed for 3-Class) ──────────────────────────────
def find_optimal_threshold(model, X_val: np.ndarray, y_val: np.ndarray,
                           device: torch.device, *args, **kwargs) -> tuple:
    """Stubbed version for multi-class classifier."""
    return 0.5, []


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(model, X: np.ndarray, device: torch.device,
            batch_size: int = 512, threshold: float = 0.5) -> np.ndarray:
    """Run inference (returns class prediction indices 0, 1, or 2)."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb     = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            logits = model(Xb)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


def predict_proba(model, X: np.ndarray, device: torch.device,
                  batch_size: int = 512) -> np.ndarray:
    """Return softmax probabilities."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb    = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            logits = model(Xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)