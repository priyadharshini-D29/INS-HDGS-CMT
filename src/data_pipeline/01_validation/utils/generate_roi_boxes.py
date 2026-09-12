import os
import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================

# cwd-relative: run with cwd = 01_validation/ (the phase root), e.g.
# `python utils/generate_roi_boxes.py`. That directory is now nested two
# levels deeper than before (src/data_pipeline/01_validation instead of
# NEUMA_PHASE0 directly under the repo root), hence "../../../".
IMAGE_PATH = "../../../DataSource/Dependencies/Brochure_Pages/ImagePage_1.tif"

OUTPUT_DIR = "output/roi"

# ============================================================
# CREATE OUTPUT DIR
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD IMAGE
# ============================================================

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

# ============================================================
# MANUAL ROI DEFINITIONS
# (x1, y1, x2, y2)
# ============================================================

roi_boxes = {

    "title": [50, 40, 700, 140],

    "product_left": [40, 180, 380, 620],

    "product_right": [420, 180, 760, 620],

    "logo": [600, 20, 780, 120],

    "footer_text": [40, 650, 760, 760]
}

# ============================================================
# DRAW ROIs
# ============================================================

overlay = img.copy()

for name, box in roi_boxes.items():

    x1, y1, x2, y2 = box

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        overlay,
        name,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# ============================================================
# SAVE
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "roi_boxes.npy"
    ),
    roi_boxes
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "roi_overlay.png"
    ),
    overlay
)

print("\nSaved:")
print("output/roi/roi_boxes.npy")
print("output/roi/roi_overlay.png")