# MLP Classifier (Anomaly Detector)

**Model file:** `src/models/classifier.py`  
**Training file:** `src/training/train_classifier.py`

---

## Role in the pipeline

The classifier is a **binary neural network**:

- **Input:** scaled feature vector (~78 dimensions)  
- **Output:** probability of **Anomaly** (class 1)  
- **Threshold:** 0.5 on sigmoid output  

Two models are trained in `main.py`:

| Model | Training data | Purpose |
|-------|---------------|---------|
| **Baseline** | Real `X_train`, `y_train` only | Comparison benchmark |
| **GAN-augmented** | Real train + 5000 synthetic anomalies | Main production model |

---

## Architecture: `AnomalyClassifier`

```text
Input (n_features)
    ↓
Linear → BatchNorm → ReLU → Dropout   # repeat for each hidden dim
    ↓
[256 → 128 → 64]  (from config.CLF_HIDDEN)
    ↓
ResidualBlock(64)   # skip connection: x + block(x)
    ↓
Linear(64 → 1) → Sigmoid
    ↓
Output: P(anomaly)
```

### Residual block

```17:18:src/models/classifier.py
    def forward(self, x):
        return self.act(x + self.block(x))
```

Adds capacity without very deep stacks — helps gradient flow.

---

## Loss function

**`BCEWithLogitsLoss`** with **positive class weight**:

```python
pos_weight = n_negative / n_positive
```

When many synthetic anomalies are added, `pos_weight` adjusts for the new imbalance so the model does not always predict anomaly.

---

## Training details

| Setting | Default |
|---------|---------|
| Optimizer | AdamW (`lr=1e-3`, `weight_decay=1e-4`) |
| Scheduler | `ReduceLROnPlateau` on validation F1 (max mode) |
| Early stopping | Patience 7 epochs on val F1 |
| Max epochs | 30 (`CLF_EPOCHS`) |
| Batch size | 512 |
| Gradient clip | max norm 1.0 |

### Augmented training set (GAN model only)

```python
X_aug = vstack([X_train, synth_X])
y_aug = concatenate([y_train, ones(len(synth_X))])
```

Baseline run passes empty `synth_X`:

```python
synth_X = np.empty((0, n_features))
```

---

## Metrics tracked per epoch

| Key | Meaning |
|-----|---------|
| `train_loss` / `val_loss` | BCE loss |
| `train_acc` / `val_acc` | Accuracy at threshold 0.5 |
| `train_f1` / `val_f1` | Binary F1 on validation |

Logged every 5 epochs and on epoch 1.

---

## Prediction functions

### `predict(model, X, device)`

- Returns hard labels `0` or `1`  
- `sigmoid(logit) >= 0.5` → anomaly  

### `predict_proba(model, X, device)`

- Returns float in [0, 1] = P(anomaly)  
- Used for ROC-AUC and PR-AUC  

---

## Saved artifact

```text
results/models/classifier.pt
```

Note: baseline and final model **overwrite the same file** — the last trained model in a run is what’s saved. Baseline is evaluated before GAN training; final classifier is saved after GAN-augmented training.

---

## Baseline behavior (typical)

From an actual run log:

- Accuracy ~0.36  
- Recall ~1.0 (predicts almost everything as anomaly)  
- Benign precision/recall ~0  

This happens because:

1. Class imbalance pushes the model toward predicting the majority attack pattern in the metric space  
2. Threshold 0.5 may not be optimal  
3. Only 8 epochs before early stopping  

The **GAN-augmented** model is expected to improve balance — check `results/reports/` for final numbers.

---

## Training history plot

```text
results/plots/classifier_training_history.png
```

Three subplots: Loss, Accuracy, F1 (train vs validation).

---

## Build function

```47:53:src/models/classifier.py
def build_classifier(config, n_features: int, device: torch.device) -> AnomalyClassifier:
    model = AnomalyClassifier(
        input_dim=n_features,
        hidden_dims=config.CLF_HIDDEN,
        dropout=config.CLF_DROPOUT,
    ).to(device)
    return model
```

---

## Hyperparameters (config.py)

| Variable | Default | Description |
|----------|---------|-------------|
| `CLF_EPOCHS` | 30 | Max training epochs |
| `CLF_BATCH_SIZE` | 512 | Minibatch size |
| `CLF_LR` | 1e-3 | Learning rate |
| `CLF_HIDDEN` | [256, 128, 64] | Hidden layer sizes |
| `CLF_DROPOUT` | 0.3 | Dropout probability |
| `CLF_WEIGHT_DECAY` | 1e-4 | L2 regularization |

See [08-CONFIGURATION-REFERENCE.md](08-CONFIGURATION-REFERENCE.md).
