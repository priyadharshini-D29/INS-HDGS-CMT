import numpy as np

from config.settings import ET_SR, PRE_TIME, POST_TIME, MIN_ET_SAMPLES


def extract_et_epoch(et, et_ts, event_ts):
    """
    Extract one ET window centred on event_ts using timestamp masking.
    Native 120 Hz — NO resampling or interpolation.
    Blink gaps (NaN) are preserved as-is.

    Returns (epoch, n_samples) or (None, 0) if window is too short.
    """
    t_start = event_ts + PRE_TIME
    t_end   = event_ts + POST_TIME

    mask  = (et_ts >= t_start) & (et_ts <= t_end)
    epoch = et[mask]

    if len(epoch) < MIN_ET_SAMPLES:
        return None, 0

    return epoch, len(epoch)


def extract_all_et_epochs(et, et_ts, events, valid_idx):
    """
    Extract ET epochs for the events that already passed EEG validation.
    An ET epoch that is too short causes the entire event to be dropped.

    Returns (et_epochs, final_valid_idx).
    """
    et_epochs      = []
    final_valid_idx = []
    sample_counts  = []

    for i in valid_idx:
        epoch, n = extract_et_epoch(et, et_ts, events[i]["timestamp"])
        if epoch is not None:
            et_epochs.append(epoch)
            final_valid_idx.append(i)
            sample_counts.append(n)

    print(f"\n===== ET EPOCH EXTRACTION =====")
    print(f"EEG-valid events : {len(valid_idx)}  |  Both valid : {len(et_epochs)}")
    if sample_counts:
        print(f"Samples          : min={min(sample_counts)}  "
              f"max={max(sample_counts)}  "
              f"mean={int(sum(sample_counts)/len(sample_counts))}")

    return et_epochs, final_valid_idx
