#!/usr/bin/env python3
"""Plot calibration reliability for the best v2 SFRA-RAG run."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt


SERIES = [
    ("RAG-fused", "correct", "confidence", "#1b9e77", "o"),
    ("Classifier", "classifier_correct", "classifier_confidence", "#377eb8", "s"),
    ("Retrieval", "retrieval_correct", "retrieval_confidence", "#e66101", "^"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v2 SFRA-RAG reliability diagram.")
    parser.add_argument(
        "--predictions-csv",
        default="outputs_v2/metrics/rag_fusion/sfra_v2_l005_seed42/sfra_v2_l005_seed42_rag_predictions.csv",
        help="Best SFRA-RAG prediction CSV.",
    )
    parser.add_argument("--bins", type=int, default=10, help="Number of equal-width ECE bins.")
    parser.add_argument(
        "--output",
        action="append",
        default=[],
        help="Output PDF path. Repeat to save in multiple locations.",
    )
    parser.add_argument("--pad-inches", type=float, default=0.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def calibration_curve(
    rows: list[dict[str, str]],
    correct_col: str,
    confidence_col: str,
    bins: int,
) -> tuple[list[float], list[float], list[int], float]:
    confidences = np.array([float(row[confidence_col]) for row in rows], dtype=float)
    correctness = np.array([int(row[correct_col]) for row in rows], dtype=float)
    bin_confidences, bin_accuracies, bin_counts = [], [], []
    ece = 0.0

    for idx in range(bins):
        lower = idx / bins
        upper = (idx + 1) / bins
        if idx == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        avg_confidence = float(confidences[mask].mean())
        avg_accuracy = float(correctness[mask].mean())
        count = int(mask.sum())
        ece += (count / len(rows)) * abs(avg_accuracy - avg_confidence)
        bin_confidences.append(avg_confidence * 100.0)
        bin_accuracies.append(avg_accuracy * 100.0)
        bin_counts.append(count)

    return bin_confidences, bin_accuracies, bin_counts, ece * 100.0


def plot(
    rows: list[dict[str, str]],
    outputs: list[Path],
    bins: int,
    pad_inches: float,
) -> None:
    fig, ax = plt.subplots(figsize=(3.9, 3.25))

    for label, correct_col, confidence_col, color, marker in SERIES:
        conf, acc, _, ece = calibration_curve(rows, correct_col, confidence_col, bins)
        ax.plot(
            conf,
            acc,
            marker=marker,
            linewidth=1.9,
            markersize=4.6,
            color=color,
            label=f"{label} (ECE {ece:.2f}%)",
        )

    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1.0, color="#555555", alpha=0.75)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Mean Confidence (%)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Empirical Accuracy (%)", fontsize=9, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")
    ax.tick_params(axis="both", labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.tight_layout(pad=0.2)
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=pad_inches)
        print(f"Saved: {output}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outputs = [Path(path) for path in args.output]
    if not outputs:
        outputs = [Path("outputs_v2/figures/reliability_diagram_sfra_v2.pdf")]
    rows = read_rows(Path(args.predictions_csv))
    plot(rows, outputs, args.bins, args.pad_inches)


if __name__ == "__main__":
    main()
