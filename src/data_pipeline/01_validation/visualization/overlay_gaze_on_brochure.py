import numpy as np
import matplotlib.pyplot as plt


def overlay_gaze_on_brochure(image, et_data, title="Gaze Overlay on Brochure"):
    if et_data.shape[1] < 2:
        print("[WARN] ET data has fewer than 2 columns — cannot overlay gaze.")
        return

    x = et_data[:, 0]
    y = et_data[:, 1]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(image)
    ax.scatter(x, y, s=2, c="cyan", alpha=0.3, label="Gaze")
    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
