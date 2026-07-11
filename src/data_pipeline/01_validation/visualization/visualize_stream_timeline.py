import matplotlib.pyplot as plt
import numpy as np


def plot_stream_timeline(streams_dict):
    plt.figure(figsize=(15, 5))

    ypos = 0

    for name, stream in streams_dict.items():
        if stream is None:
            continue

        timestamps = np.array(stream["time_stamps"])

        plt.plot(
            timestamps,
            np.ones_like(timestamps) * ypos,
            '.',
            markersize=1,
            label=name
        )

        ypos += 1

    plt.legend()
    plt.title("Multimodal Stream Timeline")
    plt.xlabel("Time (s)")
    plt.yticks(range(ypos), list(k for k, v in streams_dict.items() if v is not None))
    plt.tight_layout()
    plt.show()
