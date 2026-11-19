"""Render paired learning curves from KW-MLP training logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def read_run(path):
    result = json.loads(path.read_text())
    history = result["training"]["history"]
    if [row["epoch"] for row in history] != list(range(1, result["epochs"] + 1)):
        raise ValueError(f"Incomplete epoch history in {path}")
    if result["training"]["best_epoch"] != max(
        history, key=lambda row: row["validation"]["accuracy"]
    )["epoch"]:
        raise ValueError(f"Unexpected validation checkpoint in {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = read_run(args.run / "baseline" / "metrics.json")
    cst = read_run(args.run / "cst" / "metrics.json")
    for key in ("epochs", "seed", "batch_size", "shared_initial_sha256", "dataset"):
        if original[key] != cst[key]:
            raise ValueError(f"Training condition {key} differs between the two runs")

    figure, (loss, score) = plt.subplots(1, 2, figsize=(10.4, 3.9), layout="constrained")
    figure.suptitle("Keyword-MLP on Speech Commands V2-35", fontsize=14)
    for run, label, color, line in (
        (original, "Original KW-MLP", "#175d95", "-"),
        (cst, f"CST · {cst['atoms_per_projection']} Atoms per gate", "#ad470c", "--"),
    ):
        training = run["training"]
        rows = training["history"]
        epochs = [row["epoch"] for row in rows]
        style = dict(label=label, color=color, linestyle=line, linewidth=1.65)
        loss.plot(epochs, [row["train_loss"] for row in rows], **style)
        score.plot(epochs, [100 * row["validation"]["accuracy"] for row in rows], **style)
        best_epoch = training["best_epoch"]
        score.scatter(
            best_epoch, 100 * rows[best_epoch - 1]["validation"]["accuracy"],
            color=color, s=26, zorder=3,
        )

    loss.set(ylabel="Training loss")
    score.set(ylabel="Validation accuracy (%)")
    for axis in (loss, score):
        axis.set(xlabel="Epoch", xlim=(1, max(2, original["epochs"])))
        axis.grid(axis="y", color="#d9e1e7", linewidth=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    loss.legend(frameon=False)
    figure.savefig(args.output, dpi=190, facecolor="white")


if __name__ == "__main__":
    main()
