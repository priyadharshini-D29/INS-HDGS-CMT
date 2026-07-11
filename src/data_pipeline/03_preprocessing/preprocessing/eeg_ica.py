from sklearn.decomposition import FastICA
import numpy as np


def apply_ica(eeg, n_components=15):

    print("\nRunning ICA...")

    ica = FastICA(
        n_components=n_components,
        random_state=42,
        max_iter=1000
    )

    transformed = ica.fit_transform(eeg)

    reconstructed = ica.inverse_transform(transformed)

    print(f"ICA complete — {n_components} components used")

    return reconstructed
