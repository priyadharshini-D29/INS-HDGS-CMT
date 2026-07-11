import numpy as np
import matplotlib.pyplot as plt


def plot_blink_events(et_stream):

    et = np.array(et_stream["time_series"])

    blink_mask = np.isnan(et).any(axis=1)

    plt.figure(figsize=(14, 3))

    plt.plot(blink_mask.astype(int))

    plt.title("Detected Blink Events")

    plt.xlabel("Samples")
    plt.ylabel("Blink")

    plt.tight_layout()

    plt.show()