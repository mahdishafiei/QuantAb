"""Render the regression benchmark as a bar chart for the README / slides.

Reads results/regression_grouped.json + results/regression_random.json and writes
figures/regression_benchmark.png — Spearman ρ per model under random CV (optimistic,
leaky) vs grouped-by-antigen CV (honest), antibody-only.

Usage: python scripts/make_results_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["igbert", "esm2_650M", "balm_paired"]
LABELS = ["IgBERT", "ESM-2 650M", "BALM-paired"]


def main() -> None:
    g = json.loads(Path("results/regression_grouped.json").read_text())["results"]
    r = json.loads(Path("results/regression_random.json").read_text())["results"]

    rand_m = [r[m]["antibody"]["mean"] for m in MODELS]
    rand_s = [r[m]["antibody"]["std"] for m in MODELS]
    grp_m = [g[m]["antibody"]["mean"] for m in MODELS]
    grp_s = [g[m]["antibody"]["std"] for m in MODELS]

    x = np.arange(len(MODELS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    b1 = ax.bar(x - w / 2, rand_m, w, yerr=rand_s, capsize=4,
                color="#bdbdbd", label="Random CV (optimistic — leaky)")
    b2 = ax.bar(x + w / 2, grp_m, w, yerr=grp_s, capsize=4,
                color="#1d3557", label="Grouped-by-antigen CV (honest)")
    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9)

    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Spearman ρ  (predicted vs measured affinity)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Antibody affinity prediction — honest vs leaky evaluation\n"
                 "(mean-pooled embeddings → MLP, unified_affinity_v2)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out = Path("figures/regression_benchmark.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
