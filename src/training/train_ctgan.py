import os
import logging
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def _mahalanobis_filter(synth_X: np.ndarray, real_X: np.ndarray,
                         percentile: float = 95) -> np.ndarray:
    """
    Filter CTGAN synthetic samples by Mahalanobis distance from the real
    anomaly distribution.

    CTGAN has no explicit discriminator score we can query.  Instead we check
    whether each synthetic sample is geometrically close to the real anomaly
    cluster in feature space.  Samples beyond the ``percentile``-th percentile
    of distance (i.e., outliers) are discarded because they likely represent
    out-of-distribution noise that would confuse the classifier.

    Returns a boolean mask (True = keep).
    """
    try:
        # Use only columns with non-zero variance
        stds = real_X.std(axis=0)
        valid = stds > 1e-8
        R = real_X[:, valid]
        S = synth_X[:, valid]

        mean  = R.mean(axis=0)
        cov   = np.cov(R.T)

        # Regularise covariance to ensure invertibility
        cov  += np.eye(cov.shape[0]) * 1e-6
        cov_inv = np.linalg.pinv(cov)

        diffs = S - mean
        dists = np.array([d @ cov_inv @ d for d in diffs])

        cutoff   = np.percentile(dists, percentile)
        keep_mask = dists <= cutoff

        logger.info(
            " CTGAN Mahalanobis filter (p%.0f): kept %d / %d samples (%.1f%%)",
            percentile, keep_mask.sum(), len(synth_X),
            100.0 * keep_mask.mean(),
        )
        return keep_mask

    except Exception as exc:
        logger.warning(" Mahalanobis filter failed (%s). Keeping all samples.", exc)
        return np.ones(len(synth_X), dtype=bool)


def train_ctgan(config, X_train: np.ndarray, y_train: np.ndarray,
                feature_names: list) -> dict:
    """Train separate CTGANs on each minority class, generate synthetic samples, and filter."""
    from ctgan import CTGAN

    unique_classes = np.unique(y_train)
    minority_classes = [c for c in [1, 2] if c in unique_classes]
    
    logger.info(" Starting CTGAN Training. Minority classes to balance: %s", minority_classes)
    
    all_synth_X = []
    all_synth_y = []
    
    g_losses_dict = {}
    d_losses_dict = {}
    
    n_features = X_train.shape[1]
    
    for c in minority_classes:
        X_class = X_train[y_train == c]
        logger.info(" --- Training CTGAN for Class %d (Samples: %d) ---", c, len(X_class))
        
        if len(X_class) < 5:
            logger.warning(" Skipping CTGAN training for Class %d because sample count is too low (%d)", c, len(X_class))
            continue
            
        max_rows = getattr(config, "CTGAN_MAX_TRAIN_ROWS", None)
        if max_rows is not None and len(X_class) > max_rows:
            rng     = np.random.default_rng(config.RANDOM_STATE)
            idx     = rng.choice(len(X_class), max_rows, replace=False)
            X_class = X_class[idx]
            logger.info(" CTGAN subsampled Class %d anomalies: %d -> %d", c, len(X_train[y_train == c]), max_rows)

        df         = pd.DataFrame(X_class, columns=feature_names)
        
        # Enforce pac constraint: pac must be at most len(df) and even (default 10)
        pac = getattr(config, "CTGAN_PAC", 10)
        if len(df) < pac:
            pac = 1
            
        batch_size = min(config.CTGAN_BATCH_SIZE, len(df))
        
        # Round down batch_size to nearest multiple of pac
        if pac > 1:
            batch_size = batch_size - (batch_size % pac)
            # Ensure it is also even (in case pac is configured odd)
            if batch_size % 2 != 0:
                batch_size = max(2, batch_size - pac)
        else:
            # pac = 1, make batch_size even
            if batch_size % 2 != 0:
                batch_size = max(2, batch_size - 1)
                
        # Final safety bounds
        if batch_size < 2:
            batch_size = 2
            pac = 1

        logger.info(
            " CTGAN Class %d Training | Epochs: %d | Batch: %d | Pac: %d",
            c, config.CTGAN_EPOCHS, batch_size, pac
        )

        ctgan = CTGAN(
            epochs=config.CTGAN_EPOCHS,
            batch_size=batch_size,
            generator_lr=config.CTGAN_LR,
            discriminator_lr=config.CTGAN_LR,
            pac=pac,
            cuda=torch.cuda.is_available(),
            verbose=True,
        )
        ctgan.fit(df)

        synthetic_df = ctgan.sample(config.CTGAN_SYNTHETIC_SAMPLES)
        synth_X      = synthetic_df[feature_names].values.astype(np.float32)

        # ── Mahalanobis quality gate ─────────────────────────────────────────────
        mahal_pct = getattr(config, "CTGAN_MAHAL_PERCENTILE", 95)
        keep_mask = _mahalanobis_filter(synth_X, X_class, percentile=mahal_pct)
        synth_X_kept = synth_X[keep_mask]

        logger.info(" CTGAN Class %d done. Final synthetic samples: %d", c, len(synth_X_kept))

        models_dir = config.CTGAN_MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"ctgan_model_class_{c}.pkl")
        ctgan.save(model_path)
        logger.info(" CTGAN Class %d model saved: %s", c, model_path)

        loss_df  = ctgan.loss_values
        g_losses = loss_df["Generator Loss"].tolist() if "Generator Loss" in loss_df.columns else [0.0]
        d_losses = loss_df["Discriminator Loss"].tolist() if "Discriminator Loss" in loss_df.columns else [0.0]
        
        g_losses_dict[c] = g_losses
        d_losses_dict[c] = d_losses
        
        all_synth_X.append(synth_X_kept)
        all_synth_y.append(np.full(len(synth_X_kept), c, dtype=np.int64))

    if all_synth_X:
        synth_X_final = np.vstack(all_synth_X)
        synth_y_final = np.concatenate(all_synth_y)
    else:
        synth_X_final = np.empty((0, n_features), dtype=np.float32)
        synth_y_final = np.empty((0,), dtype=np.int64)

    logger.info(" CTGAN training complete. Total synthetic samples: %d", len(synth_y_final))

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
