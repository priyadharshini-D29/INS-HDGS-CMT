import numpy as np
import matplotlib.pyplot as plt


def plot_feature_distributions(features, feature_names=None, n_show=20, save_path=None):
    """Histogram of the first n_show feature columns."""
    n = min(n_show, features.shape[1])
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = axes.flatten()

    for i in range(n):
        axes[i].hist(features[:, i], bins=15, color="steelblue", alpha=0.7)
        label = feature_names[i] if feature_names else f"F{i}"
        axes[i].set_title(label, fontsize=7)
        axes[i].tick_params(labelsize=6)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions", fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_feature_heatmap(features, labels, save_path=None):
    """Row-normalised heatmap of feature matrix grouped by label."""
    from sklearn.preprocessing import LabelEncoder
    le     = LabelEncoder()
    y      = le.fit_transform(labels)
    order  = np.argsort(y)
    X_sorted = features[order]

    mean = X_sorted.mean(axis=0, keepdims=True)
    std  = X_sorted.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    X_norm = (X_sorted - mean) / std

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(X_norm, aspect="auto", interpolation="nearest",
                   cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_title("Feature Matrix (z-scored, sorted by label)")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Epoch")
    plt.colorbar(im, ax=ax, label="z-score")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
