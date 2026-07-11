import matplotlib.pyplot as plt


def plot_filter_comparison(raw, filtered):

    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(raw[:3000, 0], linewidth=0.5)
    plt.title("Raw EEG — Channel 0")
    plt.ylabel("Amplitude (µV)")

    plt.subplot(2, 1, 2)
    plt.plot(filtered[:3000, 0], linewidth=0.5, color="green")
    plt.title("Filtered EEG — Channel 0")
    plt.ylabel("Amplitude (µV)")
    plt.xlabel("Samples")

    plt.tight_layout()
    plt.show()
