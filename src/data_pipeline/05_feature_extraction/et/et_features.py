import numpy as np

from config.settings import ET_SR


def extract_et_features(et_epoch):
    """
    ET feature vector (6 features):
      mean gaze X
      std  gaze X
      mean gaze Y
      std  gaze Y
      mean velocity
      std  velocity
    """
    gaze_x = et_epoch[:, 0].astype(float)
    gaze_y = et_epoch[:, 1].astype(float)

    valid = ~(np.isnan(gaze_x) | np.isnan(gaze_y))
    gaze_x = gaze_x[valid]
    gaze_y = gaze_y[valid]

    if len(gaze_x) < 2:
        return np.zeros(6, dtype=np.float32)

    dx = np.diff(gaze_x)
    dy = np.diff(gaze_y)
    velocity = np.sqrt(dx ** 2 + dy ** 2)

    features = [
        np.mean(gaze_x),
        np.std(gaze_x),
        np.mean(gaze_y),
        np.std(gaze_y),
        np.mean(velocity),
        np.std(velocity),
    ]

    return np.array(features, dtype=np.float32)


def et_feature_names():
    return [
        "gaze_x_mean", "gaze_x_std",
        "gaze_y_mean", "gaze_y_std",
        "velocity_mean", "velocity_std",
    ]
