import json
import numpy as np
from pathlib import Path

_PHASE2_DIR = Path(__file__).resolve().parent.parent


def remove_bad_channels(eeg, subject_id):

    with open(_PHASE2_DIR / "metadata" / "bad_channels.json", "r") as f:
        bad_map = json.load(f)

    bad_channels = bad_map.get(subject_id, [])

    print(f"\nRemoving bad channels: {bad_channels}")

    good_idx = [
        i for i in range(eeg.shape[1])
        if i not in bad_channels
    ]

    eeg_clean = eeg[:, good_idx]

    return eeg_clean, good_idx
