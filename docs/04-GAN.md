# GAN (Generative Adversarial Network)

**Model file:** `src/models/gan.py`  
**Training file:** `src/training/train_gan.py`

---

## Type: Simple vanilla GAN (not CT-GAN)

This project uses a **basic MLP GAN**, not **CT-GAN**, **CTGAN**, or **TableGAN**.

### What “simple GAN” means here

- **Generator (G):** random noise → fully connected layers → fake feature vector  
- **Discriminator (D):** real or fake vector → probability real  
- **Loss:** binary cross-entropy (standard GAN objective)  
- **Optimizer:** Adam with betas `(0.5, 0.999)` (common GAN setting)  
- **Training data:** only rows where `y_train == 1` (anomaly class)

### What CT-GAN would add (not present)

CT-GAN / CTGAN for tabular data typically includes:

- Variational autoencoder (VAE) structure  
- **Mode-specific normalization** per continuous column  
- **Conditional** generation on categorical columns  
- Wasserstein loss with gradient penalty  
- Complex preprocessing of mixed data types  

**None of that exists in this codebase.**

---

## Why use a GAN in this project?

IoT datasets are **imbalanced**: many more benign flows than attacks.

The GAN learns the **distribution of anomaly feature vectors** and generates **5,000 synthetic anomaly samples** (`GAN_SYNTHETIC_SAMPLES`). These are concatenated with real training data before training the final classifier.

```text
Final training set = X_train (real) + synth_X (fake anomalies)
                   y_train (real)  + ones (all synthetic labeled anomaly)
```

---

## Generator architecture

```text
Input:  z ∈ R^64   (LATENT_DIM random Gaussian noise)

Linear(64 → 128) → BatchNorm → LeakyReLU(0.2)
Linear(128 → 256) → BatchNorm → LeakyReLU(0.2)
Linear(256 → 512) → BatchNorm → LeakyReLU(0.2)
Linear(512 → n_features) → Tanh

Output: fake flow feature vector (same dimension as real data, ~78)
```

- **Xavier** weight init on linear layers  
- **Tanh** squashes outputs to [-1, 1] (matches scaled real data range roughly)

---

## Discriminator architecture

```text
Input: x ∈ R^n_features

SpectralNorm(Linear(n → 512)) → LeakyReLU → Dropout(0.3)
SpectralNorm(Linear(512 → 256)) → LeakyReLU → Dropout(0.3)
SpectralNorm(Linear(256 → 128)) → LeakyReLU
Linear(128 → 1) → Sigmoid

Output: probability in [0, 1] (real vs fake)
```

**Spectral normalization** stabilizes discriminator training (Lipschitz constraint).

---

## Training loop (one batch)

For each batch of **real anomaly** vectors:

### 1. Train Discriminator

```text
loss_real = BCE(D(real), 1)
loss_fake = BCE(D(G(z)), 0)   # z ~ N(0,I), detach G for D step
loss_D = (loss_real + loss_fake) / 2
```

### 2. Train Generator

```text
loss_G = BCE(D(G(z)), 1)   # fool D into thinking fake is real
```

### 3. Schedulers

- `CosineAnnealingLR` on both G and D over `GAN_EPOCHS`

---

## Training configuration (defaults)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `GAN_EPOCHS` | 200 | **Main reason training is slow** |
| `GAN_BATCH_SIZE` | 256 | Batches per epoch |
| `GAN_LR_G`, `GAN_LR_D` | 2e-4 | Learning rates |
| Anomaly train rows | ~50,861 | Only class 1 from training split |
| Batches per epoch | ~198 | floor(50861/256) with drop_last |
| **Total batch updates** | ~39,600 | 200 × 198 |

### Why GAN takes ~25–40 minutes on CPU

- ~200 epochs × ~198 batches × (D forward/backward + G forward/backward)  
- Each batch: matrix ops on 256 × 78 tensors  
- No GPU → all on CPU  
- Logged every 20 epochs: `G_loss`, `D_loss`

Typical log pattern:

```text
[Epoch   1/200]  G_loss=0.88  D_loss=0.54
[Epoch  20/200]  G_loss=2.76  D_loss=0.07
[Epoch  60/200]  G_loss=5.23  D_loss=0.006
```

Rising G loss with falling D loss often means D is winning strongly — common in vanilla GANs without advanced stabilization.

---

## After training: synthetic sample generation

```python
z = randn(5000, 64)
synth_X = G(z)   # shape (5000, n_features)
```

Saved models:

```text
results/models/generator.pt
results/models/discriminator.pt
```

Plot:

```text
results/plots/gan_loss_curves.png
```

---

## GAN vs conditional GAN (clarification)

| Question | Answer |
|----------|--------|
| Does G receive the label? | No — only noise `z` |
| Does G generate benign traffic? | No — trained only on anomalies |
| Is it conditional GAN? | No |
| Is it CT-GAN? | No |

It is an **unconditional generator** trained on one class (anomaly) for **oversampling** that class.

---

## Code reference

Generator forward:

```33:34:src/models/gan.py
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)
```

Training uses only anomalies:

```13:14:src/training/train_gan.py
    X_anomaly = X_train[y_train == 1]
    logger.info(" GAN Training | Anomaly samples: %d | Epochs: %d",
```

---

## Tuning tips

| Goal | Change |
|------|--------|
| Faster runs | `GAN_EPOCHS = 30–50` |
| More synthetic data | `GAN_SYNTHETIC_SAMPLES = 10000` |
| Smaller model | Reduce hidden layers in `gan.py` |
| More stable GAN | Consider WGAN-GP or lower `GAN_LR_D` (requires code change) |

See [08-CONFIGURATION-REFERENCE.md](08-CONFIGURATION-REFERENCE.md).
