import numpy as np

from config.settings import EEG_SR, PRE_TIME, POST_TIME, MIN_EEG_SAMPLES


def extract_eeg_epoch(eeg, eeg_ts, event_ts):
    """
    Extract one EEG window centred on event_ts using timestamp masking.
    Native 300 Hz — NO resampling.

    Returns (epoch, n_samples) or (None, 0) if window is too short.
    """
    t_start = event_ts + PRE_TIME
    t_end   = event_ts + POST_TIME

    mask  = (eeg_ts >= t_start) & (eeg_ts <= t_end)
    epoch = eeg[mask]

    if len(epoch) < MIN_EEG_SAMPLES:
        return None, 0

    return epoch, len(epoch)


def extract_all_eeg_epochs(eeg, eeg_ts, events):
    """
    Extract EEG epochs for all events.
    Returns list of arrays (variable length) and list of valid event indices.
    """
    epochs       = []
    valid_idx    = []
    sample_counts = []

    for i, ev in enumerate(events):
        epoch, n = extract_eeg_epoch(eeg, eeg_ts, ev["timestamp"])
        if epoch is not None:
            epochs.append(epoch)
            valid_idx.append(i)
            sample_counts.append(n)

    print(f"\n===== EEG EPOCH EXTRACTION =====")
    print(f"Requested : {len(events)}  |  Valid : {len(epochs)}")
    if sample_counts:
        print(f"Samples   : min={min(sample_counts)}  "
              f"max={max(sample_counts)}  "
              f"mean={int(sum(sample_counts)/len(sample_counts))}")

    return epochs, valid_idx
