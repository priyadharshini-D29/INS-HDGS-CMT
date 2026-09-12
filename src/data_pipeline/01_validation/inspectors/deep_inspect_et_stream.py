import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def deep_inspect_et_stream(et_stream):

    print("\n======================================")
    print(" DEEP ET STREAM INSPECTION ")
    print("======================================")

    # -----------------------------------
    # Load Data
    # -----------------------------------

    data = np.array(et_stream["time_series"])
    timestamps = np.array(et_stream["time_stamps"])

    print(f"\nShape: {data.shape}")

    n_samples, n_channels = data.shape

    print(f"Samples : {n_samples}")
    print(f"Channels: {n_channels}")

    # -----------------------------------
    # Timestamp Analysis
    # -----------------------------------

    duration = timestamps[-1] - timestamps[0]

    effective_sr = (len(timestamps) - 1) / duration

    print("\n----- TIMESTAMPS -----")
    print(f"Start Time     : {timestamps[0]:.4f}")
    print(f"End Time       : {timestamps[-1]:.4f}")
    print(f"Duration (sec) : {duration:.4f}")
    print(f"Effective SR   : {effective_sr:.4f} Hz")

    # -----------------------------------
    # Per-Channel Analysis
    # -----------------------------------

    print("\n===== CHANNEL STATISTICS =====")

    for ch in range(n_channels):

        channel_data = data[:, ch]

        nan_count = np.isnan(channel_data).sum()
        inf_count = np.isinf(channel_data).sum()

        valid_data = channel_data[
            ~np.isnan(channel_data) &
            ~np.isinf(channel_data)
        ]

        print("\n----------------------------------")
        print(f"Channel {ch}")

        print(f"NaN Count      : {nan_count}")
        print(f"Inf Count      : {inf_count}")

        if len(valid_data) > 0:
            print(f"Min            : {np.min(valid_data):.4f}")
            print(f"Max            : {np.max(valid_data):.4f}")
            print(f"Mean           : {np.mean(valid_data):.4f}")
            print(f"Std            : {np.std(valid_data):.4f}")

    # -----------------------------------
    # Correlation Analysis
    # -----------------------------------

    print("\n===== CHANNEL CORRELATION =====")

    df = pd.DataFrame(data)

    corr = df.corr()

    print(corr)

    # -----------------------------------
    # Visualization
    # -----------------------------------

    print("\nGenerating visualizations...")

    # -----------------------------------
    # Plot Raw Channels
    # -----------------------------------

    plt.figure(figsize=(15, 10))

    for ch in range(n_channels):

        plt.subplot(n_channels, 1, ch + 1)

        plt.plot(data[:, ch], linewidth=0.5)

        plt.title(f"ET Channel {ch}")

        plt.ylabel("Value")

    plt.xlabel("Samples")

    plt.tight_layout()

    plt.show()

    # -----------------------------------
    # Histograms
    # -----------------------------------

    plt.figure(figsize=(15, 10))

    for ch in range(n_channels):

        plt.subplot(2, 3, ch + 1)

        valid_data = data[:, ch]
        valid_data = valid_data[~np.isnan(valid_data)]

        plt.hist(valid_data, bins=100)

        plt.title(f"Histogram Ch {ch}")

    plt.tight_layout()

    plt.show()

    # -----------------------------------
    # Scatter Analysis
    # -----------------------------------

    if n_channels >= 2:

        plt.figure(figsize=(8, 8))

        x = data[:, 0]
        y = data[:, 1]

        valid = ~np.isnan(x) & ~np.isnan(y)

        plt.scatter(
            x[valid],
            y[valid],
            s=1,
            alpha=0.3
        )

        plt.title("Possible Gaze Coordinates")
        plt.xlabel("Channel 0")
        plt.ylabel("Channel 1")

        plt.show()

    # -----------------------------------
    # NaN Visualization
    # -----------------------------------

    plt.figure(figsize=(15, 5))

    nan_matrix = np.isnan(data).astype(int)

    plt.imshow(
        nan_matrix.T,
        aspect="auto",
        interpolation="nearest"
    )

    plt.title("NaN Pattern Across ET Channels")
    plt.xlabel("Samples")
    plt.ylabel("Channels")

    plt.colorbar(label="NaN")

    plt.show()

    print("\n======================================")
    print(" ET INSPECTION COMPLETE ")
    print("======================================")