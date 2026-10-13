"""Render actual native PCA/cluster assignments into a portable figure."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/clusters.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/clustering.png"))
    args = parser.parse_args()
    rows = np.genfromtxt(args.input, delimiter=",", names=True)
    figure, axes = plt.subplots(
        1, 4, figsize=(16, 4), sharex=True, sharey=True, layout="constrained"
    )
    titles = {
        "label": "UCI varieties (reference)",
        "kmeans": "KMeans++",
        "dbscan": "DBSCAN",
        "gmm": "Diagonal Gaussian mixture",
    }
    colors = plt.get_cmap("tab10")
    for axis, name in zip(axes, titles, strict=True):
        for index, label in enumerate(np.unique(rows[name])):
            selected = rows[name] == label
            noise = label == -1
            axis.scatter(
                rows["pc1"][selected],
                rows["pc2"][selected],
                s=18,
                alpha=0.8,
                color="#999999" if noise else colors(index % 10),
                marker="x" if noise else "o",
                label="noise" if noise else str(int(label)),
            )
        axis.set_title(titles[name])
        axis.set_xlabel("Principal component 1")
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Principal component 2")
    figure.suptitle("Seeds • native C++ assignments projected with PCA", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, metadata={"Creator": "cpp-ml-portfolio"})
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
