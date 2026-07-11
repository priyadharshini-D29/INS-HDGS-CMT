import numpy as np
import matplotlib.pyplot as plt


def plot_et_gaze(et_stream, title="Eye Tracking Gaze Points"):
    data = np.array(et_stream["time_series"])

    if data.shape[1] < 2:
        print("[WARN] ET stream has fewer than 2 columns — cannot plot gaze.")
        return

    x = data[:, 0]
    y = data[:, 1]

    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, s=1, alpha=0.3, c="blue")
    plt.title(title)
    plt.xlabel("Gaze X")
    plt.ylabel("Gaze Y")
    plt.gca().invert_yaxis()
    plt.show()
