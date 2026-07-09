import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.models.gan import build_gan, compute_gradient_penalty
from src.training.train_ctgan import _mahalanobis_filter

logger = logging.getLogger(__name__)


def train_gan(config, X_train: np.ndarray, y_train: np.ndarray,
              device: torch.device) -> dict:
    """
    Train separate WGAN-GP models on each minority class (Botnet=1, Malware=2).

    WGAN-GP vs old BCE GAN:
    ─────────────────────────────────────────────────────────────────────────────
    Old BCE GAN problem: Generator loss diverged to 4.0 while Discriminator
    collapsed to ~0. This is textbook GAN mode collapse — the discriminator
    learned to perfectly reject all fakes so the generator got zero gradient.

    WGAN-GP fixes this by:
    1. Using unbounded real-valued critic scores (no sigmoid).
    2. Adding Gradient Penalty to enforce 1-Lipschitz constraint.
    3. Training the critic N_CRITIC (=5) times per generator step so the
       critic is always near-optimal but never over-powers the generator.
    4. Using (0.0, 0.9) Adam betas instead of (0.5, 0.999) for stability.

    Expected behaviour:
    - Critic loss ≈ -(real_score - fake_score) → converges to a small negative value
    - Generator loss = -fake_score → decreases as generator improves
    - Both curves should gradually stabilise, NOT diverge
    """
    unique_classes   = np.unique(y_train)
    minority_classes = [c for c in [1, 2] if c in unique_classes]

    logger.info(" Starting WGAN-GP Training. Minority classes: %s", minority_classes)

    all_synth_X: list = []
    all_synth_y: list = []
    g_losses_dict: dict = {}
    d_losses_dict: dict = {}

    n_features = X_train.shape[1]

    n_critic  = getattr(config, "GAN_N_CRITIC",  5)
    gp_lambda = getattr(config, "GAN_GP_LAMBDA", 10.0)

    for c in minority_classes:
        X_class = X_train[y_train == c]
        logger.info(" --- WGAN-GP for Class %d | Samples: %d | Epochs: %d ---",
                    c, len(X_class), config.GAN_EPOCHS)

        if len(X_class) < 10:
            logger.warning(" Skipping WGAN-GP Class %d — too few samples (%d)",
                           c, len(X_class))
            continue

        G, C = build_gan(config, n_features, device)

        batch_size = min(config.GAN_BATCH_SIZE, len(X_class))
        # Ensure even batch size for stability
        if batch_size % 2 != 0:
            batch_size = max(2, batch_size - 1)

        tensor_X = torch.tensor(X_class, dtype=torch.float32)
        loader   = DataLoader(
            TensorDataset(tensor_X),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,          # must drop last for GP to work (fixed batch size)
        )

        # WGAN-GP recommended: Adam with β1=0.0, β2=0.9, no momentum
        opt_G = torch.optim.Adam(G.parameters(), lr=config.GAN_LR_G, betas=config.GAN_BETAS)
        opt_C = torch.optim.Adam(C.parameters(), lr=config.GAN_LR_D, betas=config.GAN_BETAS)

        g_losses: list = []
        d_losses: list = []
        G.train()
        C.train()

        for epoch in range(1, config.GAN_EPOCHS + 1):
            g_epoch_loss  = 0.0
            c_epoch_loss  = 0.0
            n_g_steps     = 0
            n_c_steps     = 0

            pbar = tqdm(loader, desc=f"WGAN-GP Class {c} Ep {epoch}/{config.GAN_EPOCHS}", leave=False)

            # We iterate through the data; critic trained n_critic times, generator once
            real_buffer = []
            for (real_batch,) in pbar:
                real_buffer.append(real_batch)

                # ── Critic update ───────────────────────────────────────────────
                for _ in range(n_critic):
                    real_data = real_batch.to(device)
                    bsz       = real_data.size(0)

                    z    = torch.randn(bsz, config.LATENT_DIM, device=device)
                    fake = G(z).detach()

                    opt_C.zero_grad()
                    real_score = C(real_data).mean()
                    fake_score = C(fake).mean()
                    gp         = compute_gradient_penalty(C, real_data, fake, device, gp_lambda)

                    # WGAN-GP critic loss: minimise -(real - fake) + GP
                    loss_C = fake_score - real_score + gp
                    loss_C.backward()
                    opt_C.step()

                    c_epoch_loss += loss_C.item()
                    n_c_steps    += 1

                # ── Generator update ────────────────────────────────────────────
                opt_G.zero_grad()
                z    = torch.randn(bsz, config.LATENT_DIM, device=device)
                fake = G(z)
                # Generator loss: wants critic to score fakes highly
                loss_G = -C(fake).mean()
                loss_G.backward()
                opt_G.step()

                g_epoch_loss += loss_G.item()
                n_g_steps    += 1

            g_losses.append(g_epoch_loss / max(n_g_steps, 1))
            d_losses.append(c_epoch_loss / max(n_c_steps, 1))

            if epoch % 10 == 0 or epoch == 1 or epoch == config.GAN_EPOCHS:
                logger.info("  [WGAN-GP Class %d | Ep %3d/%d]  G=%.4f  C=%.4f",
                            c, epoch, config.GAN_EPOCHS,
                            g_losses[-1], d_losses[-1])

        g_losses_dict[c] = g_losses
        d_losses_dict[c] = d_losses

        # ── Generate synthetic samples ─────────────────────────────────────────
        G.eval()
        with torch.no_grad():
            z_synth = torch.randn(config.GAN_SYNTHETIC_SAMPLES,
                                  config.LATENT_DIM, device=device)
            synth_X = G(z_synth).cpu().numpy()

        # ── Quality filter via Mahalanobis distance ────────────────────────────
        # WGAN-GP critic outputs unbounded scores, so we use Mahalanobis (same
        # as CTGAN) instead of the old DISC_QUALITY_THRESHOLD which assumed 0-1.
        keep_mask    = _mahalanobis_filter(synth_X, X_class, percentile=95)
        synth_X_kept = synth_X[keep_mask]

        n_total = len(synth_X)
        n_kept  = len(synth_X_kept)
        logger.info(" WGAN-GP Class %d: kept %d / %d synthetic samples (%.1f%%)",
                    c, n_kept, n_total, 100.0 * n_kept / max(n_total, 1))

        if n_kept == 0:
            logger.warning(" Mahalanobis filter removed ALL samples! Keeping full set.")
            synth_X_kept = synth_X

        all_synth_X.append(synth_X_kept)
        all_synth_y.append(np.full(len(synth_X_kept), c, dtype=np.int64))

        # ── Save model ─────────────────────────────────────────────────────────
        models_dir = getattr(config, "SIMPLE_GAN_MODELS_DIR", config.MODELS_DIR)
        os.makedirs(models_dir, exist_ok=True)
        torch.save(G.state_dict(), os.path.join(models_dir, f"generator_class_{c}.pt"))
        torch.save(C.state_dict(), os.path.join(models_dir, f"critic_class_{c}.pt"))

    if all_synth_X:
        synth_X_final = np.vstack(all_synth_X)
        synth_y_final = np.concatenate(all_synth_y)
    else:
        synth_X_final = np.empty((0, n_features), dtype=np.float32)
        synth_y_final = np.empty((0,), dtype=np.int64)

    logger.info(" WGAN-GP training complete. Total synthetic: %d samples", len(synth_y_final))

    rep_class    = minority_classes[0] if minority_classes else 1
    rep_g_losses = g_losses_dict.get(rep_class, [0.0])
    rep_d_losses = d_losses_dict.get(rep_class, [0.0])

    return {
        "synth_X":      synth_X_final,
        "synth_y":      synth_y_final,
        "g_losses":     rep_g_losses,
        "d_losses":     rep_d_losses,
        "g_losses_all": g_losses_dict,
        "d_losses_all": d_losses_dict,
    }