import matplotlib.pyplot as plt
import numpy as np


def overlay_gaze_on_brochure(image, et_data):

    h, w = image.shape[:2]

    # Left eye coordinates
    x = et_data[:, 0]
    y = et_data[:, 1]

    # Remove NaNs
    valid = ~np.isnan(x) & ~np.isnan(y)

    x = x[valid]
    y = y[valid]

    # Convert normalized coords → image pixels
    x_pix = x * w
    y_pix = y * h

    plt.figure(figsize=(16, 10))

    plt.imshow(image)

    plt.scatter(
        x_pix,
        y_pix,
        s=2,
        alpha=0.3
    )

    plt.title("ET Gaze Overlay on Brochure")

    plt.axis("off")

    plt.show()