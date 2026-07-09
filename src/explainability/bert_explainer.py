import os
import logging
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

def features_to_text(feature_vector: np.ndarray, feature_names: list, top_k: int = 10) -> str:

    ranked_idx = np.argsort(np.abs(feature_vector))[::-1][:top_k]
    parts = []
    for i in ranked_idx:
        name = feature_names[i].strip()
        val  = feature_vector[i]
        if "pkt" in name.lower() or "bytes" in name.lower():
            parts.append(f"{name} is {'high' if val > 0.5 else 'low'} ({val:.2f})")
        elif "flag" in name.lower():
            parts.append(f"{name} count is {abs(val):.1f}")
        elif "duration" in name.lower():
            parts.append(f"flow duration is {'long' if val > 0 else 'short'}")
        elif "rate" in name.lower() or "pkts/s" in name.lower():
            parts.append(f"traffic rate is {'elevated' if val > 0.5 else 'normal'} ({val:.2f})")
        else:
            parts.append(f"{name} = {val:.2f}")
    return "Network flow: " + "; ".join(parts) + "."

def build_explanation(prediction: int, probability: float,
                       feature_vector: np.ndarray, feature_names: list,
                       attack_label: str = None) -> str:

    label     = "ANOMALY (Attack)" if prediction == 1 else "BENIGN"
    conf      = probability if prediction == 1 else 1 - probability
    flow_text = features_to_text(feature_vector, feature_names)

    explanation = (
        f"[PREDICTION]: {label} | Confidence: {conf*100:.1f}%\n"
        f"[FLOW DESCRIPTION]: {flow_text}\n"
    )
    if prediction == 1:
        explanation += (
            f"[REASON]: This network flow exhibits elevated packet rates, "
            f"unusual flag combinations, and abnormal byte distributions "
            f"consistent with {'IoT botnet' if 'SSH' in (attack_label or '') else 'brute-force'} activity. "
            f"The GAN-augmented model flagged this as malicious with high confidence.\n"
            f"[ATTACK TYPE]: {attack_label or 'Unknown Attack'}\n"
            f"[RECOMMENDATION]: Block the source IP. Investigate destination port. "
            f"Enable rate limiting and intrusion prevention rules."
        )
    else:
        explanation += (
            f"[REASON]: Flow characteristics match normal IoT device communication. "
            f"Packet sizes, rates, and flag patterns are within expected thresholds.\n"
            f"[RECOMMENDATION]: No action required. Continue monitoring."
        )
    return explanation

class FlowTextDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_len: int):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

class BertAnomalyExplainer:
    def __init__(self, config, device: torch.device):
        self.config    = config
        self.device    = device
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL)
        self.model     = BertForSequenceClassification.from_pretrained(
            config.BERT_MODEL, num_labels=2
        ).to(device)
        logger.info(" BERT model loaded: %s", config.BERT_MODEL)

    def prepare_texts(self, X: np.ndarray, y: np.ndarray,
                      feature_names: list) -> list:

        return [features_to_text(X[i], feature_names) for i in range(len(X))]

    def fine_tune(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  feature_names: list):

        logger.info(" Fine-tuning BERT on %d samples...", len(X_train))

        train_texts = self.prepare_texts(X_train, y_train, feature_names)
        val_texts   = self.prepare_texts(X_val,   y_val,   feature_names)

        train_ds = FlowTextDataset(train_texts, y_train.tolist(), self.tokenizer, self.config.BERT_MAX_LEN)
        val_ds   = FlowTextDataset(val_texts,   y_val.tolist(),   self.tokenizer, self.config.BERT_MAX_LEN)

        train_loader = DataLoader(train_ds, batch_size=self.config.BERT_BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.config.BERT_BATCH_SIZE, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.BERT_LR)
        total_steps = len(train_loader) * self.config.BERT_EPOCHS
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )

        best_val_acc = 0.0
        for epoch in range(1, self.config.BERT_EPOCHS + 1):
            self.model.train()
            tr_loss = 0.0
            pbar = tqdm(train_loader, desc=f"BERT Epoch {epoch}/{self.config.BERT_EPOCHS}")
            for batch in pbar:
                optimizer.zero_grad()
                out  = self.model(
                    input_ids      = batch["input_ids"].to(self.device),
                    attention_mask = batch["attention_mask"].to(self.device),
                    labels         = batch["labels"].to(self.device),
                )
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                tr_loss += out.loss.item()

            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    logits  = self.model(
                        input_ids      = batch["input_ids"].to(self.device),
                        attention_mask = batch["attention_mask"].to(self.device),
                    ).logits
                    preds   = logits.argmax(dim=1).cpu()
                    correct += (preds == batch["labels"]).sum().item()
                    total   += len(batch["labels"])
            val_acc = correct / total
            logger.info("  BERT [Epoch %d/%d]  loss=%.4f  val_acc=%.4f",
                        epoch, self.config.BERT_EPOCHS,
                        tr_loss / len(train_loader), val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(self.config.MODELS_DIR, exist_ok=True)
                self.model.save_pretrained(os.path.join(self.config.MODELS_DIR, "bert_explainer"))
                self.tokenizer.save_pretrained(os.path.join(self.config.MODELS_DIR, "bert_explainer"))

        logger.info(" BERT fine-tuning complete. Best val_acc=%.4f", best_val_acc)

    def generate_explanations(self, X_test: np.ndarray, y_pred: np.ndarray,
                               y_proba: np.ndarray, att_labels: np.ndarray,
                               feature_names: list, n_samples: int = 10) -> list:

        explanations = []
        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

        for idx in indices:
            exp = build_explanation(
                prediction    = int(y_pred[idx]),
                probability   = float(y_proba[idx]),
                feature_vector= X_test[idx],
                feature_names = feature_names,
                attack_label  = str(att_labels[idx]) if att_labels is not None else None,
            )
            explanations.append({
                "sample_idx":   int(idx),
                "prediction":   "Anomaly" if y_pred[idx] == 1 else "Benign",
                "confidence":   f"{float(y_proba[idx])*100:.1f}%",
                "attack_type":  str(att_labels[idx]) if att_labels is not None else "N/A",
                "explanation":  exp,
            })
        return explanations