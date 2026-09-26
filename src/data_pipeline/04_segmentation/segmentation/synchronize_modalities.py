import numpy as np


def get_overlap_window(eeg_ts, et_ts):
    """
    Return the common time window where both streams have data.
    Uses native timestamps — no resampling.
    """
    t_start = max(eeg_ts[0],  et_ts[0])
    t_end   = min(eeg_ts[-1], et_ts[-1])

    if t_end <= t_start:
        raise ValueError(
            f"No overlap between EEG ({eeg_ts[0]:.3f}–{eeg_ts[-1]:.3f}) "
            f"and ET ({et_ts[0]:.3f}–{et_ts[-1]:.3f})"
        )

    overlap = t_end - t_start

    print(f"\n===== TEMPORAL SYNCHRONIZATION =====")
    print(f"EEG range       : {eeg_ts[0]:.3f} → {eeg_ts[-1]:.3f} s")
    print(f"ET  range       : {et_ts[0]:.3f} → {et_ts[-1]:.3f} s")
    print(f"Overlap window  : {t_start:.3f} → {t_end:.3f} s  ({overlap:.2f} s)")

    return t_start, t_end


def trim_to_overlap(eeg, eeg_ts, et, et_ts):
    """
    Trim both streams to their shared time window.
    Each stream keeps its native frequency and sample count.
    """
    t_start, t_end = get_overlap_window(eeg_ts, et_ts)

    eeg_mask = (eeg_ts >= t_start) & (eeg_ts <= t_end)
    et_mask  = (et_ts  >= t_start) & (et_ts  <= t_end)

    eeg_trim    = eeg[eeg_mask]
    eeg_ts_trim = eeg_ts[eeg_mask]
    et_trim     = et[et_mask]
    et_ts_trim  = et_ts[et_mask]

    print(f"EEG after trim  : {eeg_trim.shape}  samples={len(eeg_ts_trim)}")
    print(f"ET  after trim  : {et_trim.shape}   samples={len(et_ts_trim)}")

    return eeg_trim, eeg_ts_trim, et_trim, et_ts_trim
