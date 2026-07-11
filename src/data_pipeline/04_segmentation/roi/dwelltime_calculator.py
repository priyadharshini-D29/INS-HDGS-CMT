import numpy as np

from config.settings import ET_SR


def compute_dwell_time(hit_array, roi_index, fs=ET_SR):
    """
    Compute total dwell time (seconds) for one ROI in one epoch.

    hit_array : (T,) array of ROI indices per sample (-1 = no hit).
    """
    n_hits = int(np.sum(hit_array == roi_index))
    return n_hits / fs


def compute_all_dwell_times(hit_array, n_rois, fs=ET_SR):
    """
    Returns dict {roi_index: dwell_seconds} for every ROI.
    """
    return {
        i: compute_dwell_time(hit_array, i, fs)
        for i in range(n_rois)
    }


def dwell_time_summary(hit_array, n_rois, fs=ET_SR):
    dwell = compute_all_dwell_times(hit_array, n_rois, fs)

    total_gaze = sum(dwell.values())
    attention  = {
        k: (v / total_gaze if total_gaze > 0 else 0.0)
        for k, v in dwell.items()
    }

    top_roi = max(dwell, key=dwell.get) if dwell else -1

    return {
        "dwell_seconds":     dwell,
        "attention_ratio":   attention,
        "top_roi":           top_roi,
        "total_gaze_seconds": total_gaze
    }
