import numpy as np
import matplotlib.pyplot as plt


def plot_et_quality(et_stream, subject_id=""):

    et = np.array(et_stream["time_series"])
    timestamps = np.array(et_stream["time_stamps"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    if et.shape[1] >= 2:
        axes[0].scatter(et[:, 0], et[:, 1], s=1, alpha=0.2, c="blue")
        axes[0].set_title(f"Gaze Scatter (X vs Y) — {subject_id}")
        axes[0].set_xlabel("Gaze X")
        axes[0].set_ylabel("Gaze Y")
        axes[0].invert_yaxis()

    nan_ratio = np.isnan(et).mean(axis=0)
    axes[1].bar(range(len(nan_ratio)), nan_ratio)
    axes[1].axhline(0.10, color="red", linestyle="--", label="10% threshold")
    axes[1].set_title("ET NaN Ratio per Channel")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("NaN Ratio")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
