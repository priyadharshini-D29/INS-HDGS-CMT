import numpy as np


def point_in_box(x, y, box):

    x1, y1, x2, y2 = box

    return (
        x >= x1 and
        x <= x2 and
        y >= y1 and
        y <= y2
    )


def map_gaze_to_roi(
    gaze_x,
    gaze_y,
    roi_boxes,
    image_shape
):
    """
    Map normalized gaze coordinates to ROI name.

    Parameters
    ----------
    gaze_x, gaze_y : float
        Normalized gaze coordinates (0–1)

    roi_boxes : dict
        {
            "roi_name": [x1, y1, x2, y2]
        }

    image_shape : tuple
        (H, W) or (H, W, C)

    Returns
    -------
    str
        ROI name or "background"
    """

    h, w = image_shape[:2]

    # normalized -> pixels
    px = gaze_x * w
    py = gaze_y * h

    for roi_name, box in roi_boxes.items():

        if len(box) != 4:
            continue

        x1, y1, x2, y2 = box

        if point_in_box(
            px,
            py,
            [x1, y1, x2, y2]
        ):
            return roi_name

    return "background"


def map_epoch_gaze_to_roi(
    et_epoch,
    roi_boxes,
    image_shape
):
    """
    Map all ET samples in an epoch to ROI names.

    Returns
    -------
    ndarray[str]
        ROI label per sample
    """

    n = len(et_epoch)

    hits = np.full(
        n,
        "background",
        dtype=object
    )

    for t in range(n):

        gx = et_epoch[t, 0]
        gy = et_epoch[t, 1]

        if np.isnan(gx) or np.isnan(gy):
            hits[t] = "invalid"
            continue

        hits[t] = map_gaze_to_roi(
            gx,
            gy,
            roi_boxes,
            image_shape
        )

    return hits


def roi_hit_statistics(hit_array):
    """
    Count samples per ROI label.
    """

    unique, counts = np.unique(
        hit_array,
        return_counts=True
    )

    stats = {
        str(k): int(v)
        for k, v in zip(unique, counts)
    }

    return stats