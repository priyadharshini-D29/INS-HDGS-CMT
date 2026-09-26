import numpy as np

from loaders.load_brochures import load_brochure
from loaders.load_roi_boxes import load_roi_boxes

from visualization.visualize_roi_boxes import visualize_roi_boxes
from visualization.overlay_gaze_on_brochure import overlay_gaze_on_brochure

from validators.validate_roi_boxes import validate_roi_boxes


def inspect_roi_mapping(
    brochure_path,
    roi_path,
    et_stream
):

    # -----------------------------------
    # Load brochure
    # -----------------------------------

    image = load_brochure(brochure_path)

    # -----------------------------------
    # Load ROI file
    # -----------------------------------

    roi_data = load_roi_boxes(roi_path)

    # -----------------------------------
    # IMPORTANT:
    # Adjust key name after inspection
    # -----------------------------------

    possible_keys = [
        key for key in roi_data.keys()
        if not key.startswith("__")
    ]

    roi_key = possible_keys[0]

    roi_boxes_raw = roi_data[roi_key]

    print(f"\nUsing ROI key : {roi_key}")
    print(f"Raw dtype      : {roi_boxes_raw.dtype}")
    print(f"Raw shape      : {roi_boxes_raw.shape}")

    # MATLAB cell array arrives as object array of shape (1, N).
    # Flatten each cell into a (4,) vector → final shape (N, 4).
    if roi_boxes_raw.dtype == object:
        roi_boxes = np.vstack([
            np.array(roi_boxes_raw[0, i]).flatten()
            for i in range(roi_boxes_raw.shape[1])
        ])
    else:
        roi_boxes = np.array(roi_boxes_raw)

    print(f"Reshaped boxes : {roi_boxes.shape}  (each row = x1 y1 x2 y2)")
    print(f"First 3 rows:\n{roi_boxes[:3]}")

    # -----------------------------------
    # Validate ROI
    # -----------------------------------

    validate_roi_boxes(
        roi_boxes,
        image.shape
    )

    # -----------------------------------
    # Visualize ROI
    # -----------------------------------

    visualize_roi_boxes(
        image,
        roi_boxes
    )

    # -----------------------------------
    # Overlay gaze
    # -----------------------------------

    et_data = np.array(et_stream["time_series"])

    overlay_gaze_on_brochure(
        image,
        et_data
    )