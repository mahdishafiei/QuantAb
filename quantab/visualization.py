"""Publication-quality figure generation."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

COLORS = {
    "quantum_minimal": "#e63946",
    "quantum_expressive": "#f4a261",
    "linear_svm": "#457b9d",
    "rbf_svm": "#1d3557",
    "poly_svm": "#2a9d8f",
    "random_forest": "#6d6875",
}


def plot_learning_curves(
    results: dict[str, dict],
    title: str = "Learning Curves",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot Spearman correlation vs training set size for all methods.

    Args:
        results: {method_name: {"train_sizes": [...], "means": [...], "stds": [...]}}.
        title: Figure title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in results.items():
        sizes = data["train_sizes"]
        means = np.array(data["means"])
        stds = np.array(data["stds"])
        color = COLORS.get(method, "#666666")
        ls = "--" if method.startswith("quantum") else "-"
        ax.plot(sizes, means, label=method, color=color, lw=2, ls=ls, marker="o", ms=5)
        ax.fill_between(sizes, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_affinity_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot of predicted vs measured affinity."""
    from scipy.stats import spearmanr
    rho = spearmanr(y_true, y_pred).statistic
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color=COLORS.get(method, "#457b9d"))
    ax.set_xlabel("Measured affinity")
    ax.set_ylabel("Predicted affinity")
    ax.set_title(f"{method}  ρ = {rho:.3f}")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
