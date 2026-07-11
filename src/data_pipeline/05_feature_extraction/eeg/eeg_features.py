import numpy as np
import antropy as ant

from eeg.bandpower import compute_bandpower
from eeg.hjorth     import hjorth_parameters

from config.settings import EEG_BANDS, EEG_SR


def extract_eeg_features(eeg_epoch):
    """
    Per-channel feature vector:
      5 bandpower values (delta, theta, alpha, beta, gamma)
      2 Hjorth parameters (mobility, complexity)
      1 sample entropy
    Total per channel : 8
    Total (20 ch)     : 160 features
    """
    features = []

    n_channels = eeg_epoch.shape[1]

    for ch in range(n_channels):

        signal = eeg_epoch[:, ch].astype(float)

        # guard: skip flat channels
        if np.std(signal) < 1e-10:
            features.extend([0.0] * 8)
            continue

        # ── Bandpower ────────────────────────────────────────
        for band_range in EEG_BANDS.values():
            bp = compute_bandpower(signal, EEG_SR, band_range)
            features.append(float(bp))

        # ── Hjorth ───────────────────────────────────────────
        mobility, complexity = hjorth_parameters(signal)
        features.append(mobility)
        features.append(complexity)

        # ── Sample entropy ───────────────────────────────────
        try:
            entropy = float(ant.sample_entropy(signal))
        except Exception:
            entropy = 0.0
        features.append(entropy)

    return np.array(features, dtype=np.float32)


def feature_names(n_channels=20):
    names = []
    for ch in range(n_channels):
        for band in EEG_BANDS:
            names.append(f"ch{ch:02d}_{band}_bp")
        names.append(f"ch{ch:02d}_hjorth_mobility")
        names.append(f"ch{ch:02d}_hjorth_complexity")
        names.append(f"ch{ch:02d}_sample_entropy")
    return names
