import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from config.settings import ET_SR


def plot_et_epoch(epoch, event_id=None, save_path=None):
    """
    Gaze scanpath for one ET epoch.
    NaN samples (blinks) create breaks in the trajectory.
    """
    gaze_x = epoch[:, 0].astype(float)
    gaze_y = epoch[:, 1].astype(float)
    times  = np.arange(len(epoch)) / ET_SR

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Scanpath ─────────────────────────────────────────────
    valid = ~np.isnan(gaze_x) & ~np.isnan(gaze_y)
    axes[0].scatter(gaze_x[valid], gaze_y[valid],
                    c=times[valid], cmap="viridis", s=3, alpha=0.6)
    axes[0].invert_yaxis()
    axes[0].set_title(f"Gaze Scanpath{f' — event {event_id}' if event_id is not None else ''}")
    axes[0].set_xlabel("X (normalised)")
    axes[0].set_ylabel("Y (normalised)")

    # ── X/Y over time ─────────────────────────────────────────
    axes[1].plot(times, gaze_x, label="X", linewidth=0.6, color="steelblue")
    axes[1].plot(times, gaze_y, label="Y", linewidth=0.6, color="tomato")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Gaze coordinate")
    axes[1].set_title("Gaze X / Y over time")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
