"""Render matched learning curves from KW-MLP training logs."""

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
    parser.add_argument("--cst-reference", type=Path, help="CST metrics JSON for the reference curve")
    parser.add_argument("--cst-compare", type=Path, help="Metrics JSON for another CST run")
    args = parser.parse_args()
    original = read_run(args.run / "baseline" / "metrics.json")
    cst = read_run(args.cst_reference or args.run / "cst" / "metrics.json")
    runs = [(original, "Original KW-MLP", "#175d95", "-")]
    candidates = [(cst, "#ad470c", "--")]
    if args.cst_compare:
        candidates.append((read_run(args.cst_compare), "#147a64", "-."))
    for run, color, line in candidates:
        if run["variant"] != "cst":
            parser.error("Expected CST metrics for the CST curves")
        label = f"CST · {run['atoms_per_projection']} Atoms · LR {run['learning_rates'][1]:g}"
        runs.append((run, label, color, line))
    for run, label, _, _ in runs[1:]:
        for key in ("epochs", "seed", "batch_size", "shared_initial_sha256", "dataset"):
            if original[key] != run[key]:
                raise ValueError(f"Training condition {key} differs in {label}")

    figure, (loss, score) = plt.subplots(1, 2, figsize=(10.4, 3.9), layout="constrained")
    figure.suptitle("Keyword-MLP on Speech Commands V2-35", fontsize=14)
    zoom = None
    if args.cst_compare and original["epochs"] >= 40:
        zoom_start = original["epochs"] // 7
        zoom = score.inset_axes([0.32, 0.16, 0.62, 0.51])
        zoom.set_title(f"Epochs {zoom_start}–{original['epochs']}", fontsize=8, pad=2)
        zoom.tick_params(labelsize=7)
        zoom.grid(axis="y", color="#d9e1e7", linewidth=0.6)
    for run, label, color, line in runs:
        training = run["training"]
        rows = training["history"]
        epochs = [row["epoch"] for row in rows]
        style = dict(label=label, color=color, linestyle=line, linewidth=1.65)
        loss.plot(epochs, [row["train_loss"] for row in rows], **style)
        validation = [100 * row["validation"]["accuracy"] for row in rows]
        score.plot(epochs, validation, **style)
        best_epoch = training["best_epoch"]
        score.scatter(best_epoch, validation[best_epoch - 1], color=color, s=26, zorder=3)
        if zoom is not None:
            zoom.plot(epochs[zoom_start - 1 :], validation[zoom_start - 1 :], **style)
            if best_epoch >= zoom_start:
                zoom.scatter(best_epoch, validation[best_epoch - 1], color=color, s=10, zorder=3)

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
