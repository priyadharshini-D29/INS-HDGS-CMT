import matplotlib.pyplot as plt


def save_figure(fig, output_path):
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"[SAVED] {output_path}")


def new_figure(figsize=(12, 6)):
    return plt.subplots(figsize=figsize)
