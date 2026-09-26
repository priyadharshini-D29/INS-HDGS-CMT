import matplotlib.pyplot as plt
import seaborn as sns


def plot_channel_heatmap(corr):

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0
    )

    plt.title("EEG Channel Correlation Matrix")

    plt.tight_layout()

    plt.show()