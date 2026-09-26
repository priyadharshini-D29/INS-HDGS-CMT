import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def validate_roi_boxes(roi_boxes, image_shape):

    h, w = image_shape[:2]

    print("\n===== ROI VALIDATION =====")

    print(f"Image dimensions : {w}w x {h}h")
    print(f"Boxes shape      : {roi_boxes.shape}")

    for idx, box in enumerate(roi_boxes):

        x, y, bw, bh = box

        errors = []

        # -----------------------------------
        # Width / height validation
        # -----------------------------------

        if bw <= 0:
            errors.append("invalid width")

        if bh <= 0:
            errors.append("invalid height")

        # -----------------------------------
        # Boundary validation
        # -----------------------------------

        if x < 0:
            errors.append("negative x")

        if y < 0:
            errors.append("negative y")

        if x + bw > w:
            errors.append("box exceeds image width")

        if y + bh > h:
            errors.append("box exceeds image height")

        # -----------------------------------
        # Output
        # -----------------------------------

        if len(errors) == 0:

            print(
                f"[PASS] ROI {idx+1:02d} | "
                f"x={x:.1f} y={y:.1f} "
                f"w={bw:.1f} h={bh:.1f}"
            )

        else:

            print(
                f"[FAIL] ROI {idx+1:02d}: {errors}"
            )

    print("\nROI validation complete.")


def visualize_roi_boxes(image, roi_boxes):
    _, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(image)

    print("\n===== DRAWING ROI BOXES =====")

    for idx, box in enumerate(roi_boxes):
        x, y, bw, bh = box

        rect = patches.Rectangle(
            (x, y), bw, bh,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x, y - 5, f"P{idx+1}", color='yellow', fontsize=8, weight='bold')

    plt.title("Brochure ROI Validation")
    plt.axis("off")
    plt.tight_layout()
    plt.show()