import numpy as np
import matplotlib.pyplot as plt


def plot_timestamp_drift(streams):

    fig, ax = plt.subplots(figsize=(14, 5))

    for name, stream in streams.items():

        ts = np.array(stream["time_stamps"])
        diffs = np.diff(ts)

        ax.plot(diffs, label=name, linewidth=0.5, alpha=0.8)

    ax.axhline(0.5, color="red", linestyle="--", label="0.5s warning")
    ax.set_title("Timestamp Gaps per Stream")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Gap (s)")
    ax.legend()
    plt.tight_layout()
    plt.show()
