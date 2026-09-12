import numpy as np
import matplotlib.pyplot as plt


def plot_gaze_cleaning(
    et_raw,
    et_clean,
    subject="S01"
):

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 8)
    )

    # -----------------------------------
    # RAW GAZE
    # -----------------------------------

    raw_x = et_raw[:, 0]
    raw_y = et_raw[:, 1]

    valid_raw = (
        ~np.isnan(raw_x)
        &
        ~np.isnan(raw_y)
    )

    axes[0].scatter(
        raw_x[valid_raw],
        raw_y[valid_raw],
        s=2,
        c="red",
        alpha=0.4
    )

    axes[0].invert_yaxis()

    axes[0].set_title(
        f"Raw Gaze — {subject}"
    )

    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # -----------------------------------
    # CLEAN GAZE
    # -----------------------------------

    clean_x = et_clean[:, 0]
    clean_y = et_clean[:, 1]

    valid_clean = (
        ~np.isnan(clean_x)
        &
        ~np.isnan(clean_y)
    )

    axes[1].scatter(
        clean_x[valid_clean],
        clean_y[valid_clean],
        s=2,
        c="blue",
        alpha=0.4
    )

    axes[1].invert_yaxis()

    axes[1].set_title(
        f"Clean Gaze — {subject}"
    )

    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    plt.tight_layout()
    plt.show()