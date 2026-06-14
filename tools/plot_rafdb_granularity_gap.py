#!/usr/bin/env python3
"""Plot the RAF-DB granularity-gap comparison used in the manuscript."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the RAF-DB VLM comparison figure.")
    parser.add_argument(
        "--output",
        default="outputs_v2/figures/vlm_comparison_rafdb.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--pad-inches",
        type=float,
        default=0.0,
        help="Whitespace padding around the saved PDF. Use 0 for a tight crop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    models = ["InstructBLIP\n7B", "LLaVA-1.5\n7B", "Plain ViT", "SFRA-RAG"]
    accuracies = [23.96, 60.56, 87.38, 87.33]
    errors = [0.0, 0.0, 0.10, 0.42]
    colors = ["#9aa0a6", "#7f8790", "#355c9a", "#c43b32"]

    fig, ax = plt.subplots(figsize=(5.2, 2.75))
    bars = ax.bar(
        models,
        accuracies,
        yerr=errors,
        capsize=3,
        color=colors,
        edgecolor="#222222",
        linewidth=0.7,
    )

    for bar, value in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.8,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, 103)
    ax.set_ylabel("Top-1 Accuracy on RAF-DB (%)", fontsize=9, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=0.15)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=args.pad_inches)
    plt.close(fig)
    print(f"Saved RAF-DB comparison figure: {output}")


if __name__ == "__main__":
    main()
