import numpy as np

from config.settings import EEG_SR, ET_SR, MIN_EEG_SAMPLES, MIN_ET_SAMPLES


def validate_eeg_epoch(epoch, event_id=None):
    """
    Return True if the EEG epoch passes quality checks.
    """
    tag = f"[event {event_id}]" if event_id is not None else ""

    if epoch is None or len(epoch) == 0:
        print(f"[REJECT]{tag} empty epoch")
        return False

    if len(epoch) < MIN_EEG_SAMPLES:
        print(f"[REJECT]{tag} too short: {len(epoch)} < {MIN_EEG_SAMPLES}")
        return False

    if np.isnan(epoch).all():
        print(f"[REJECT]{tag} all-NaN epoch")
        return False

    ch_std = np.nanstd(epoch, axis=0)
    flat   = np.sum(ch_std < 1e-6)
    if flat > epoch.shape[1] * 0.5:
        print(f"[REJECT]{tag} >{50}% flat channels ({flat})")
        return False

    return True


def validate_et_epoch(epoch, event_id=None):
    """
    Return True if the ET epoch passes quality checks.
    Blink gaps (NaN) are expected and do not cause rejection
    unless the entire epoch is NaN.
    """
    tag = f"[event {event_id}]" if event_id is not None else ""

    if epoch is None or len(epoch) == 0:
        print(f"[REJECT]{tag} empty ET epoch")
        return False

    if len(epoch) < MIN_ET_SAMPLES:
        print(f"[REJECT]{tag} ET too short: {len(epoch)} < {MIN_ET_SAMPLES}")
        return False

    nan_ratio = np.isnan(epoch[:, :2]).mean()
    if nan_ratio >= 1.0:
        print(f"[REJECT]{tag} ET epoch entirely NaN")
        return False

    return True


def validate_epoch_pair(eeg_epoch, et_epoch, event_id=None):
    return (
        validate_eeg_epoch(eeg_epoch, event_id) and
        validate_et_epoch(et_epoch,  event_id)
    )
