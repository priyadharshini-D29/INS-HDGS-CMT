import numpy as np


# ============================================================
# I-VT FIXATION DETECTOR
# Velocity Threshold Identification
# ============================================================

"""
Scientifically grounded fixation detection for NeuMa ET.

INPUT
-----
et_epoch : np.ndarray
    Shape = (samples, channels)

Expected channels:
    ch0 = gaze_x
    ch1 = gaze_y

PARAMETERS
----------
sampling_rate : int
    ET sampling frequency (default=120Hz)

velocity_threshold : float
    Max velocity allowed for fixation
    Units = pixels/sample

min_duration_ms : int
    Minimum fixation duration in milliseconds

OUTPUT
------
fixations : list[dict]

Each fixation:
{
    "start_idx": int,
    "end_idx": int,
    "duration_ms": float,
    "mean_x": float,
    "mean_y": float,
    "mean_velocity": float
}
"""


# ============================================================
# COMPUTE GAZE VELOCITY
# ============================================================

def compute_velocity(x, y):

    dx = np.diff(x)
    dy = np.diff(y)

    velocity = np.sqrt(dx**2 + dy**2)

    velocity = np.insert(
        velocity,
        0,
        0
    )

    return velocity


# ============================================================
# FIXATION DETECTOR
# ============================================================

def detect_fixations(
    et_epoch,
    sampling_rate=120,
    velocity_threshold=30,
    min_duration_ms=100
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if et_epoch.shape[1] < 2:

        raise ValueError(
            "ET epoch must contain "
            "at least x/y gaze channels."
        )

    # --------------------------------------------------------
    # EXTRACT GAZE
    # --------------------------------------------------------

    x = et_epoch[:, 0]
    y = et_epoch[:, 1]

    # --------------------------------------------------------
    # REMOVE NaNs
    # --------------------------------------------------------

    valid = (
        ~np.isnan(x)
        &
        ~np.isnan(y)
    )

    x_valid = x.copy()
    y_valid = y.copy()

    x_valid[~valid] = 0
    y_valid[~valid] = 0

    # --------------------------------------------------------
    # COMPUTE VELOCITY
    # --------------------------------------------------------

    velocity = compute_velocity(
        x_valid,
        y_valid
    )

    # --------------------------------------------------------
    # FIXATION MASK
    # --------------------------------------------------------

    fixation_mask = (
        velocity < velocity_threshold
    )

    # --------------------------------------------------------
    # MINIMUM FIXATION SAMPLES
    # --------------------------------------------------------

    min_samples = int(
        (min_duration_ms / 1000)
        * sampling_rate
    )

    # Example:
    # 100 ms @ 120 Hz
    # = 12 samples

    # --------------------------------------------------------
    # EXTRACT FIXATION SEGMENTS
    # --------------------------------------------------------

    fixations = []

    in_fixation = False
    start_idx = 0

    for i in range(len(fixation_mask)):

        # ----------------------------------------------------
        # ENTER FIXATION
        # ----------------------------------------------------

        if fixation_mask[i] and not in_fixation:

            in_fixation = True
            start_idx = i

        # ----------------------------------------------------
        # EXIT FIXATION
        # ----------------------------------------------------

        elif not fixation_mask[i] and in_fixation:

            end_idx = i

            duration_samples = (
                end_idx - start_idx
            )

            # -----------------------------------------------
            # VALID FIXATION
            # -----------------------------------------------

            if duration_samples >= min_samples:

                fx_x = x[start_idx:end_idx]
                fx_y = y[start_idx:end_idx]

                fx_vel = velocity[
                    start_idx:end_idx
                ]

                fixation = {

                    "start_idx": start_idx,

                    "end_idx": end_idx,

                    "duration_ms":
                        (duration_samples / sampling_rate)
                        * 1000,

                    "mean_x":
                        np.nanmean(fx_x),

                    "mean_y":
                        np.nanmean(fx_y),

                    "mean_velocity":
                        np.nanmean(fx_vel)
                }

                fixations.append(fixation)

            in_fixation = False

    # --------------------------------------------------------
    # HANDLE FINAL FIXATION
    # --------------------------------------------------------

    if in_fixation:

        end_idx = len(fixation_mask)

        duration_samples = (
            end_idx - start_idx
        )

        if duration_samples >= min_samples:

            fx_x = x[start_idx:end_idx]
            fx_y = y[start_idx:end_idx]

            fx_vel = velocity[
                start_idx:end_idx
            ]

            fixation = {

                "start_idx": start_idx,

                "end_idx": end_idx,

                "duration_ms":
                    (duration_samples / sampling_rate)
                    * 1000,

                "mean_x":
                    np.nanmean(fx_x),

                "mean_y":
                    np.nanmean(fx_y),

                "mean_velocity":
                    np.nanmean(fx_vel)
            }

            fixations.append(fixation)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n===== FIXATION DETECTION =====")

    print(
        f"Velocity threshold : "
        f"{velocity_threshold}"
    )

    print(
        f"Minimum duration   : "
        f"{min_duration_ms} ms"
    )

    print(
        f"Detected fixations : "
        f"{len(fixations)}"
    )

    return fixations