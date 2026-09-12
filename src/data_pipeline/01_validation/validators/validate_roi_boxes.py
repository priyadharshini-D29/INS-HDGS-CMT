import numpy as np


def validate_roi_boxes(roi_boxes, image_shape):

    h, w = image_shape[:2]

    print("\n===== ROI VALIDATION =====")
    print(f"Image dimensions   : {w}w x {h}h")
    print(f"Boxes shape        : {roi_boxes.shape}")

    # MATLAB bounding box format: [x, y, width, height]
    print(f"Format             : x, y, width, height (MATLAB convention)")

    for idx, box in enumerate(roi_boxes):

        x, y, bw, bh = box[:4]

        errors = []

        if bw <= 0:
            errors.append(f"invalid width={bw:.1f}")

        if bh <= 0:
            errors.append(f"invalid height={bh:.1f}")

        if x < 0:
            errors.append(f"negative x={x:.1f}")

        if y < 0:
            errors.append(f"negative y={y:.1f}")

        if x + bw > w:
            errors.append(f"exceeds image width ({x+bw:.1f} > {w})")

        if y + bh > h:
            errors.append(f"exceeds image height ({y+bh:.1f} > {h})")

        if len(errors) == 0:
            print(f"[PASS] ROI {idx+1:02d} | x={x:.1f} y={y:.1f} w={bw:.1f} h={bh:.1f}")
        else:
            print(f"[FAIL] ROI {idx+1:02d}: {errors}")