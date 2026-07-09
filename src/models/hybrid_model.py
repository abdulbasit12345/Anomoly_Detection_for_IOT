import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HybridDataset(Dataset):
    """
    Dataset that returns both tabular features and pre-computed/computed LLM embeddings.
    """
    def __init__(self, X_tab: np.ndarray, X_emb: np.ndarray, y: np.ndarray):
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.X_emb = torch.tensor(X_emb, dtype=torch.float32)
        self.y     = torch.tensor(y,     dtype=torch.long)

    def __len__(self):
        return len(self.X_tab)

    def __getitem__(self, idx):
        return {
            "tab_x": self.X_tab[idx],
            "emb_x": self.X_emb[idx],
            "y":     self.y[idx]
        }


class HybridFusionClassifier(nn.Module):
    """
    Multimodal Fusion Network:
    - Tabular branch (MLP): raw network features -> 128-d embedding
    - Text branch: 768-d pre-computed DistilBERT sequence embedding
    - Fusion layer: Concatenates both embeddings and classifies into 3 classes
    """
    def __init__(self, tabular_dim: int, embedding_dim: int = 768, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        
        # Tabular network
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Text representation is already 768-d from DistilBERT
        
        # Fusion Classifier Head
        self.fusion_dim = 128 + embedding_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, tab_x: torch.Tensor, emb_x: torch.Tensor) -> torch.Tensor:
        tab_feats = self.tabular_net(tab_x)
        fused = torch.cat([tab_feats, emb_x], dim=1)
        logits = self.fusion_head(fused)
        return logits


# ── DistilBERT Embedding Extraction helper ──────────────────────────────────────
def get_distilbert_embeddings(config, texts: list, device: torch.device) -> np.ndarray:
    """
    Run frozen DistilBERT on a list of texts and return the CLS sequence embeddings.
    This runs in batches and is extremely fast.
    """
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    logger.info(" Initializing DistilBERT tokenizer & model for feature extraction...")
    tokenizer = DistilBertTokenizer.from_pretrained(config.BERT_MODEL)
    model     = DistilBertModel.from_pretrained(config.BERT_MODEL).to(device)
    model.eval()

    logger.info(" Extracting embeddings for %d sequences...", len(texts))
    embeddings = []
    
    batch_size = config.BERT_BATCH_SIZE
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                max_length=config.BERT_MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            # CLS token is at index 0 of sequence length
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            
            if (i // batch_size) % 100 == 0 and i > 0:
                logger.info("   Extracted: %d/%d", i, len(texts))

    return np.vstack(embeddings)


# ── Hybrid Model Training Loop ──────────────────────────────────────────────────
def train_hybrid_classifier(config, model, X_train, X_train_emb, y_train,
                            X_val, X_val_emb, y_val,
                            device: torch.device, model_filename: str = "hybrid_fusion.pt",
                            models_dir: str = None) -> dict:
    """
    Train the Hybrid Multimodal Fusion classifier using tabular and text embeddings.
    Uses Standard CrossEntropyLoss for 3-class target.
    """
    logger.info(" Training Hybrid Multimodal Fusion Classifier (3 classes)...")
    
    train_ds = HybridDataset(X_train, X_train_emb, y_train)
    val_ds   = HybridDataset(X_val,   X_val_emb,   y_val)
    
    train_loader = DataLoader(train_ds, batch_size=config.CLF_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.CLF_BATCH_SIZE, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.CLF_LR, weight_decay=config.CLF_WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.CLF_LR * 5,
        steps_per_epoch=len(train_loader),
        epochs=config.CLF_EPOCHS,
    )
    
    best_val_f1 = 0.0
    best_state = None
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    
    for epoch in range(1, config.CLF_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        correct, total = 0, 0
        
        for batch in train_loader:
            tab_x = batch["tab_x"].to(device)
            emb_x = batch["emb_x"].to(device)
            y     = batch["y"].to(device)
            
            optimizer.zero_grad()
            logits = model(tab_x, emb_x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            tr_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
            
        tr_loss /= len(train_ds)
        tr_acc  = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                tab_x = batch["tab_x"].to(device)
                emb_x = batch["emb_x"].to(device)
                y     = batch["y"].to(device)
                
                logits = model(tab_x, emb_x)
                loss   = criterion(logits, y)
                val_loss += loss.item() * len(y)
                
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += len(y)
                
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y.cpu().numpy())
                
        val_loss /= len(val_ds)
        val_acc  = val_correct / val_total
        
        # We check Macro F1-score to choose the best model
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
        if epoch % 10 == 0 or epoch == 1:
            logger.info("  [Hybrid Epoch %2d/%d] loss=%.4f val_loss=%.4f val_acc=%.4f val_f1_macro=%.4f",
                        epoch, config.CLF_EPOCHS, tr_loss, val_loss, val_acc, val_f1)
                        
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        
    save_dir = models_dir or config.MODELS_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), save_path)
    logger.info(" Saved best Hybrid model (val_f1=%.4f) to: %s", best_val_f1, save_path)
    
    return history


def predict_hybrid(model, X_tab: np.ndarray, X_emb: np.ndarray, device: torch.device) -> np.ndarray:
    """Run prediction on hybrid input (returns argmax labels)."""
    model.eval()
    all_preds = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X_tab), batch_size):
            tx = torch.tensor(X_tab[i:i + batch_size], dtype=torch.float32).to(device)
            ex = torch.tensor(X_emb[i:i + batch_size], dtype=torch.float32).to(device)
            logits = model(tx, ex)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


def predict_proba_hybrid(model, X_tab: np.ndarray, X_emb: np.ndarray, device: torch.device) -> np.ndarray:
    """Run prediction on hybrid input (returns probabilities)."""
    model.eval()
    all_probs = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X_tab), batch_size):
            tx = torch.tensor(X_tab[i:i + batch_size], dtype=torch.float32).to(device)
            ex = torch.tensor(X_emb[i:i + batch_size], dtype=torch.float32).to(device)
            logits = model(tx, ex)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)
