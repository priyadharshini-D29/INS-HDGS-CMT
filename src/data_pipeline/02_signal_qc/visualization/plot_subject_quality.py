import matplotlib.pyplot as plt


def plot_subject_quality(scores_dict):

    subjects = list(scores_dict.keys())
    scores = list(scores_dict.values())

    colors = ["green" if s >= 90 else "orange" if s >= 70 else "red" for s in scores]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(subjects, scores, color=colors)
    ax.axhline(90, color="green", linestyle="--", label="Excellent (90)")
    ax.axhline(70, color="orange", linestyle="--", label="Usable (70)")
    ax.set_title("Subject Quality Scores")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 110)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
