import numpy as np

from config.settings import (
    EEG_EPOCHS_PATH,
    ET_EPOCHS_PATH,
    LABELS_PATH
)


def load_epochs():

    eeg = np.load(EEG_EPOCHS_PATH, allow_pickle=True)
    et  = np.load(ET_EPOCHS_PATH,  allow_pickle=True)
    labels = np.load(LABELS_PATH,  allow_pickle=True)

    print(f"\n===== LOADED EPOCHS =====")
    print(f"EEG epochs : {len(eeg)}")
    print(f"ET  epochs : {len(et)}")
    print(f"Labels     : {len(labels)}")
    print(f"EEG shape[0]: {eeg[0].shape}")
    print(f"ET  shape[0]: {et[0].shape}")

    return eeg, et, labels
