import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_roi_overlay(image, roi_boxes, et_epoch,
                     fixations=None, event_id=None, save_path=None):
    """
    Overlay gaze points and ROI boxes on a brochure image.

    image      : (H, W, 3) uint8 array.
    roi_boxes  : (N, 4)  — [x, y, width, height] in pixels.
    et_epoch   : (T, C)  — columns 0/1 are normalised gaze X/Y.
    fixations  : list of fixation dicts from fixation_detector.
    """
    h, w = image.shape[:2]

    _, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(image)

    # ── ROI boxes ─────────────────────────────────────────────
    for idx, box in enumerate(roi_boxes):
        x, y, bw, bh = box
        rect = patches.Rectangle(
            (x, y), bw, bh,
            linewidth=1.5, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x + 2, y + 14, f"P{idx+1}",
                color="lime", fontsize=7, weight="bold")

    # ── Gaze trajectory ───────────────────────────────────────
    gx = et_epoch[:, 0] * w
    gy = et_epoch[:, 1] * h
    valid = ~np.isnan(gx) & ~np.isnan(gy)
    ax.scatter(gx[valid], gy[valid], s=2, c="red", alpha=0.4, label="Gaze")

    # ── Fixation circles ──────────────────────────────────────
    if fixations:
        for fx in fixations:
            cx = fx["cx"] * w
            cy = fx["cy"] * h
            r  = max(8, fx["duration"] * 0.5)
            circle = plt.Circle((cx, cy), r,
                                 color="yellow", fill=False, linewidth=1.5)
            ax.add_patch(circle)

    title = f"ROI Overlay{f' — event {event_id}' if event_id is not None else ''}"
    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
