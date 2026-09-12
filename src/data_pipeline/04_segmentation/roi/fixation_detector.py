import numpy as np

from config.settings import (
    FIXATION_VELOCITY_THRESHOLD,
    FIXATION_MIN_DURATION_SAMPLES
)


# ============================================================
# SCREEN SIZE
# ============================================================

SCREEN_W = 3000
SCREEN_H = 1688


# ============================================================
# VELOCITY
# ============================================================

def compute_velocity(gaze_x, gaze_y):

    dx = np.diff(gaze_x)
    dy = np.diff(gaze_y)

    velocity = np.sqrt(dx ** 2 + dy ** 2)

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
    velocity_threshold=FIXATION_VELOCITY_THRESHOLD,
    min_duration=FIXATION_MIN_DURATION_SAMPLES,
    sampling_rate=120
):

    if et_epoch.shape[1] < 5:
        raise ValueError(
            "ET epoch must contain binocular gaze channels."
        )

    # --------------------------------------------------------
    # BINOCULAR GAZE
    # --------------------------------------------------------

    left_x = et_epoch[:, 0].astype(float)
    left_y = et_epoch[:, 1].astype(float)

    right_x = et_epoch[:, 3].astype(float)
    right_y = et_epoch[:, 4].astype(float)

    stack_x = np.vstack([
        left_x,
        right_x
    ])

    stack_y = np.vstack([
        left_y,
        right_y
    ])

    # --------------------------------------------------------
    # SAFE BINOCULAR FUSION
    # --------------------------------------------------------

    gaze_x = np.empty(
        stack_x.shape[1]
    )

    gaze_y = np.empty(
        stack_y.shape[1]
    )

    for i in range(stack_x.shape[1]):

        xvals = stack_x[:, i]
        yvals = stack_y[:, i]

        # ----------------------------------------------------
        # X
        # ----------------------------------------------------

        valid_x = xvals[
            ~np.isnan(xvals)
        ]

        if len(valid_x) == 0:
            gaze_x[i] = np.nan
        else:
            gaze_x[i] = np.mean(valid_x)

        # ----------------------------------------------------
        # Y
        # ----------------------------------------------------

        valid_y = yvals[
            ~np.isnan(yvals)
        ]

        if len(valid_y) == 0:
            gaze_y[i] = np.nan
        else:
            gaze_y[i] = np.mean(valid_y)
    # --------------------------------------------------------
    # NORMALIZED → PIXELS
    # --------------------------------------------------------

    gaze_x = gaze_x * SCREEN_W
    gaze_y = gaze_y * SCREEN_H

    # --------------------------------------------------------
    # VALID MASK
    # --------------------------------------------------------

    valid = (
        ~np.isnan(gaze_x)
        &
        ~np.isnan(gaze_y)
    )

    gaze_x_clean = gaze_x.copy()
    gaze_y_clean = gaze_y.copy()

    gaze_x_clean[~valid] = 0
    gaze_y_clean[~valid] = 0

    # --------------------------------------------------------
    # VELOCITY
    # --------------------------------------------------------

    velocity = compute_velocity(
        gaze_x_clean,
        gaze_y_clean
    )

    # --------------------------------------------------------
    # FIXATION MASK
    # --------------------------------------------------------

    is_fixation = (
        velocity < velocity_threshold
    )

    is_fixation[~valid] = False

    # --------------------------------------------------------
    # SEGMENT FIXATIONS
    # --------------------------------------------------------

    fixations = []

    i = 0

    while i < len(is_fixation):

        if is_fixation[i]:

            j = i

            while (
                j < len(is_fixation)
                and is_fixation[j]
            ):
                j += 1

            duration = j - i

            if duration >= min_duration:

                fx_x = gaze_x[i:j]
                fx_y = gaze_y[i:j]

                # ------------------------------------------------
                # SAFE FIXATION CENTROID
                # ------------------------------------------------

                if np.all(np.isnan(fx_x)):
                    cx = np.nan
                else:
                    cx = float(
                        np.nanmean(fx_x)
                    )

                if np.all(np.isnan(fx_y)):
                    cy = np.nan
                else:
                    cy = float(
                        np.nanmean(fx_y)
                    )

                # ------------------------------------------------
                # FIXATION OBJECT
                # ------------------------------------------------

                fixation = {

                    "start":
                        i,

                    "end":
                        j,

                    "cx":
                        cx,

                    "cy":
                        cy,

                    "duration":
                        duration,

                    "duration_ms":
                        (
                            duration
                            / sampling_rate
                        ) * 1000,

                    "mean_velocity":
                        float(
                            np.nanmean(
                                velocity[i:j]
                            )
                        )
                }

                fixations.append(
                    fixation
                )

            i = j

        else:

            i += 1

    return fixations


# ============================================================
# FIXATION SUMMARY
# ============================================================

def fixation_summary(fixations):

    if not fixations:

        return {

            "count": 0,

            "total_duration": 0,

            "mean_duration": 0.0,

            "mean_duration_ms": 0.0
        }

    durations = [
        f["duration"]
        for f in fixations
    ]

    durations_ms = [
        f["duration_ms"]
        for f in fixations
    ]

    return {

        "count":
            len(fixations),

        "total_duration":
            int(np.sum(durations)),

        "mean_duration":
            float(np.mean(durations)),

        "mean_duration_ms":
            float(np.mean(durations_ms))
    }