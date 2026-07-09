import os
import logging
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def _mahalanobis_filter(synth_X: np.ndarray, real_X: np.ndarray,
                         percentile: float = 95) -> np.ndarray:
    try:
        stds = real_X.std(axis=0)
        valid = stds > 1e-8
        R = real_X[:, valid]
        S = synth_X[:, valid]
        mean    = R.mean(axis=0)
        cov     = np.cov(R.T) + np.eye(R.shape[1]) * 1e-6
        cov_inv = np.linalg.pinv(cov)
        diffs   = S - mean
        dists   = np.array([d @ cov_inv @ d for d in diffs])
        cutoff  = np.percentile(dists, percentile)
        keep    = dists <= cutoff
        logger.info(
            " CTGAN Mahalanobis filter (p%.0f): kept %d / %d samples (%.1f%%)",
            percentile, keep.sum(), len(synth_X), 100.0 * keep.mean(),
        )
        return keep
    except Exception as exc:
        logger.warning(" Mahalanobis filter failed (%s). Keeping all samples.", exc)
        return np.ones(len(synth_X), dtype=bool)


def train_ctgan(config, X_train: np.ndarray, y_train: np.ndarray,
                feature_names: list) -> dict:
    """
    Try the external ctgan package first. If unavailable or incompatible
    (e.g. Python 3.12 + torch 2.2 conflict), fall back to the internal WGAN-GP.
    """
    try:
        from ctgan import CTGAN as _CTGAN
        logger.info(" [CTGAN] Using external ctgan package.")
        return _train_ctgan_external(config, X_train, y_train, feature_names, _CTGAN)
    except Exception as e:
        logger.warning(
            " External ctgan unavailable (%s). Falling back to internal WGAN-GP.", e
        )
        return _train_ctgan_internal(config, X_train, y_train)


# ── External CTGAN implementation ────────────────────────────────────────────
def _train_ctgan_external(config, X_train, y_train, feature_names, CTGAN):
    unique_classes  = np.unique(y_train)
    minority_classes = [c for c in [1, 2] if c in unique_classes]
    n_features      = X_train.shape[1]

    all_synth_X, all_synth_y = [], []
    g_losses_dict,  d_losses_dict = {}, {}

    for c in minority_classes:
        X_class = X_train[y_train == c]
        logger.info(" --- CTGAN Class %d | Samples: %d ---", c, len(X_class))
        if len(X_class) < 5:
            continue

        max_rows = getattr(config, "CTGAN_MAX_TRAIN_ROWS", None)
        if max_rows and len(X_class) > max_rows:
            idx     = np.random.default_rng(config.RANDOM_STATE).choice(len(X_class), max_rows, replace=False)
            X_class = X_class[idx]

        df         = pd.DataFrame(X_class, columns=feature_names)
        pac        = getattr(config, "CTGAN_PAC", 10)
        if len(df) < pac:
            pac = 1

        batch_size = min(config.CTGAN_BATCH_SIZE, len(df))
        if pac > 1:
            batch_size = batch_size - (batch_size % pac)
            if batch_size % 2 != 0:
                batch_size = max(2, batch_size - pac)
        else:
            if batch_size % 2 != 0:
                batch_size = max(2, batch_size - 1)
        if batch_size < 2:
            batch_size, pac = 2, 1

        model = CTGAN(
            epochs=config.CTGAN_EPOCHS,
            batch_size=batch_size,
            generator_lr=config.CTGAN_LR,
            discriminator_lr=config.CTGAN_LR,
            pac=pac,
            cuda=torch.cuda.is_available(),
            verbose=True,
        )
        model.fit(df)

        synth_X = model.sample(config.CTGAN_SYNTHETIC_SAMPLES)[feature_names].values.astype(np.float32)
        keep    = _mahalanobis_filter(synth_X, X_class, getattr(config, "CTGAN_MAHAL_PERCENTILE", 95))
        synth_X = synth_X[keep]

        models_dir = config.CTGAN_MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)
        model.save(os.path.join(models_dir, f"ctgan_model_class_{c}.pkl"))

        try:
            ldf      = model.loss_values
            g_losses = ldf["Generator Loss"].tolist() if "Generator Loss" in ldf.columns else [0.0]
            d_losses = ldf["Discriminator Loss"].tolist() if "Discriminator Loss" in ldf.columns else [0.0]
        except Exception:
            g_losses, d_losses = [0.0], [0.0]

        g_losses_dict[c] = g_losses
        d_losses_dict[c] = d_losses
        all_synth_X.append(synth_X)
        all_synth_y.append(np.full(len(synth_X), c, dtype=np.int64))

    if all_synth_X:
        synth_X_final = np.vstack(all_synth_X)
        synth_y_final = np.concatenate(all_synth_y)
    else:
        synth_X_final = np.empty((0, n_features), dtype=np.float32)
        synth_y_final = np.empty((0,),            dtype=np.int64)

    rep = minority_classes[0] if minority_classes else 1
    return {
        "synth_X":      synth_X_final,
        "synth_y":      synth_y_final,
        "g_losses":     g_losses_dict.get(rep, [0.0]),
        "d_losses":     d_losses_dict.get(rep, [0.0]),
        "g_losses_all": g_losses_dict,
        "d_losses_all": d_losses_dict,
    }


# ── Internal WGAN-GP fallback (no external package needed) ───────────────────
def _train_ctgan_internal(config, X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Reuse the proven WGAN-GP from train_gan when ctgan package is unavailable."""
    from src.training.train_gan import train_gan as _train_wgan

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(" [CTGAN fallback] Running WGAN-GP on device: %s", device)
    results = _train_wgan(config, X_train, y_train, device)

    return {
        "synth_X":      results["synth_X"],
        "synth_y":      results["synth_y"],
        "g_losses":     results["g_losses"],
        "d_losses":     results["d_losses"],
        "g_losses_all": {},
        "d_losses_all": {},
    }
