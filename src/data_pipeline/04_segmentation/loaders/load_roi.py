import numpy as np
import scipy.io
from pathlib import Path


def load_roi_boxes_npy(path=None):
    """
    Load ROI bounding boxes from a .npy file.

    path : explicit Path or str. If None, uses settings.ROI_BOXES_PATH.
    """
    if path is not None:
        roi_path = Path(path)
    else:
        from config.settings import ROI_BOXES_PATH
        roi_path = ROI_BOXES_PATH

    if not roi_path.exists():
        raise FileNotFoundError(f"ROI boxes not found: {roi_path}")

    boxes = np.load(roi_path, allow_pickle=True).item()
    print(f"\n===== ROI BOXES =====")
    print(f"Total ROI regions : {len(boxes)}")
    for roi_name, roi_box in boxes.items():
        print(f"{roi_name} -> {roi_box}")
    return boxes


def load_roi_boxes_mat(mat_path):
    """Load ROI boxes from a MATLAB .mat file."""
    data = scipy.io.loadmat(str(mat_path))
    keys = [k for k in data.keys() if not k.startswith("__")]
    key  = keys[0]
    raw  = data[key]

    if raw.dtype == object:
        boxes = np.vstack([
            np.array(raw[0, i]).flatten()
            for i in range(raw.shape[1])
        ])
    else:
        boxes = np.array(raw)

    print(f"\n===== ROI BOXES (mat) =====")
    print(f"Key : {key}  Shape : {boxes.shape}")
    return boxes
