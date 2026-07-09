import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.models.gan import build_gan

logger = logging.getLogger(__name__)


def train_gan(config, X_train: np.ndarray, y_train: np.ndarray,
              device: torch.device) -> dict:
    """
    Train separate GANs on each present minority class (Botnet=1, Malware=2)
    and generate synthetic samples to balance the dataset.
    """
    unique_classes = np.unique(y_train)
    minority_classes = [c for c in [1, 2] if c in unique_classes]
    
    logger.info(" Starting GAN Training. Minority classes to balance: %s", minority_classes)
    
    all_synth_X = []
    all_synth_y = []
    
    # Store losses for reporting/plotting (we can save/average them)
    g_losses_dict = {}
    d_losses_dict = {}
    
    n_features = X_train.shape[1]
    
    for c in minority_classes:
        X_class = X_train[y_train == c]
        logger.info(" --- Training GAN for Class %d (Samples: %d, Epochs: %d) ---", 
                    c, len(X_class), config.GAN_EPOCHS)
        
        if len(X_class) < 5:
            logger.warning(" Skipping GAN training for Class %d because sample count is too low (%d)", c, len(X_class))
            continue
            
        G, D = build_gan(config, n_features, device)
        
        tensor_X = torch.tensor(X_class, dtype=torch.float32)
        loader   = DataLoader(
            TensorDataset(tensor_X),
            batch_size=min(config.GAN_BATCH_SIZE, len(X_class)),
            shuffle=True,
            drop_last=True if len(X_class) >= config.GAN_BATCH_SIZE else False,
        )
        
        opt_G = torch.optim.Adam(G.parameters(), lr=config.GAN_LR_G, betas=config.GAN_BETAS)
        opt_D = torch.optim.Adam(D.parameters(), lr=config.GAN_LR_D, betas=config.GAN_BETAS)
        criterion = nn.BCELoss()
        
        sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=config.GAN_EPOCHS)
        sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=config.GAN_EPOCHS)
        
        g_losses, d_losses = [], []
        G.train(); D.train()
        
        for epoch in range(1, config.GAN_EPOCHS + 1):
            g_epoch, d_epoch = 0.0, 0.0
            n_batches = 0
            pbar = tqdm(loader, desc=f"GAN Class {c} Epoch {epoch}/{config.GAN_EPOCHS}", leave=False)
            for (real_batch,) in pbar:
                real_batch = real_batch.to(device)
                bsz        = real_batch.size(0)
                
                real_labels = torch.ones(bsz,  1, device=device)
                fake_labels = torch.zeros(bsz, 1, device=device)
                
                # ── Train Discriminator ─────────────────────────────────────────
                opt_D.zero_grad()
                z    = torch.randn(bsz, config.LATENT_DIM, device=device)
                fake = G(z).detach()
                
                loss_real = criterion(D(real_batch), real_labels)
                loss_fake = criterion(D(fake),       fake_labels)
                loss_D    = (loss_real + loss_fake) * 0.5
                loss_D.backward()
                opt_D.step()
                
                # ── Train Generator ─────────────────────────────────────────────
                opt_G.zero_grad()
                z    = torch.randn(bsz, config.LATENT_DIM, device=device)
                fake = G(z)
                loss_G = criterion(D(fake), real_labels)
                loss_G.backward()
                opt_G.step()
                
                g_epoch += loss_G.item()
                d_epoch += loss_D.item()
                n_batches += 1
                
            if n_batches > 0:
                g_losses.append(g_epoch / n_batches)
                d_losses.append(d_epoch / n_batches)
            else:
                g_losses.append(0.0)
                d_losses.append(0.0)
                
            sched_G.step(); sched_D.step()
            
            if epoch % 50 == 0 or epoch == 1:
                logger.info("  [Class %d - Epoch %3d/%d]  G_loss=%.4f  D_loss=%.4f",
                            c, epoch, config.GAN_EPOCHS, g_losses[-1], d_losses[-1])
                            
        g_losses_dict[c] = g_losses
        d_losses_dict[c] = d_losses
        
        # ── Generate synthetic samples for this class ───────────────────────────
        G.eval()
        with torch.no_grad():
            z_synth  = torch.randn(config.GAN_SYNTHETIC_SAMPLES, config.LATENT_DIM, device=device)
            synth_X  = G(z_synth).cpu().numpy()
            
        # ── Discriminator filter ────────────────────────────────────────────────
        disc_threshold = getattr(config, "DISC_QUALITY_THRESHOLD", 0.55)
        D.eval()
        with torch.no_grad():
            synth_tensor  = torch.tensor(synth_X, dtype=torch.float32).to(device)
            disc_scores   = D(synth_tensor).squeeze(1).cpu().numpy()
            
        keep_mask = disc_scores >= disc_threshold
        n_total   = len(synth_X)
        synth_X_kept = synth_X[keep_mask]
        n_kept    = len(synth_X_kept)
        
        logger.info(" GAN Class %d filter (D-score >= %.2f): kept %d / %d samples (%.1f%%)",
                    c, disc_threshold, n_kept, n_total, 100.0 * n_kept / max(n_total, 1))
                    
        if n_kept == 0:
            logger.warning(" Anti-evasion filter removed ALL samples! Keeping top-50%% by D-score.")
            top_half = disc_scores >= np.median(disc_scores)
            synth_X_kept = synth_X[top_half]
            
        all_synth_X.append(synth_X_kept)
        all_synth_y.append(np.full(len(synth_X_kept), c, dtype=np.int64))
        
        # Save model weights
        models_dir = getattr(config, "SIMPLE_GAN_MODELS_DIR", config.MODELS_DIR)
        os.makedirs(models_dir, exist_ok=True)
        torch.save(G.state_dict(), os.path.join(models_dir, f"generator_class_{c}.pt"))
        torch.save(D.state_dict(), os.path.join(models_dir, f"discriminator_class_{c}.pt"))
        
    if all_synth_X:
        synth_X_final = np.vstack(all_synth_X)
        synth_y_final = np.concatenate(all_synth_y)
    else:
        synth_X_final = np.empty((0, n_features), dtype=np.float32)
        synth_y_final = np.empty((0,), dtype=np.int64)
        
    logger.info(" GAN training complete. Total synthetic samples: %d", len(synth_y_final))
    
    # Return average loss of present classes for visualization or first class losses
    rep_class = minority_classes[0] if minority_classes else 1
    rep_g_losses = g_losses_dict.get(rep_class, [0.0])
    rep_d_losses = d_losses_dict.get(rep_class, [0.0])
    
    return {
        "synth_X":    synth_X_final,
        "synth_y":    synth_y_final,
        "g_losses":   rep_g_losses,
        "d_losses":   rep_d_losses,
        "g_losses_all": g_losses_dict,
        "d_losses_all": d_losses_dict,
    }