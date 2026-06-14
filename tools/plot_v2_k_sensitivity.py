#!/usr/bin/env python3
"""Plot all-seed v2 retrieval-depth sensitivity from saved CSV outputs."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt


GROUPS = {
    "plain_vit": ("Plain ViT", "#1f77b4", "o"),
    "vit_gem": ("ViT+GeM", "#ff7f0e", "s"),
    "sfra_v2": ("SFRA-RAG", "#2ca02c", "^"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v2 k-sensitivity across all seeds.")
    parser.add_argument(
        "--rag-root",
        default="outputs_v2/metrics/rag_fusion",
        help="Directory containing per-run *_k_sensitivity.csv files.",
    )
    parser.add_argument(
        "--output",
        default="outputs_v2/figures/k_sensitivity_v2.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--pad-inches",
        type=float,
        default=0.0,
        help="Extra padding passed to savefig. Use 0 for no white border.",
    )
    return parser.parse_args()


def group_for_run(run_name: str) -> str:
    if run_name.startswith("plain_vit"):
        return "plain_vit"
    if run_name.startswith("vit_gem"):
        return "vit_gem"
    if run_name.startswith("sfra_v2"):
        return "sfra_v2"
    raise ValueError(f"Unknown run group for {run_name}")


def load_rows(rag_root: Path) -> dict[str, dict[int, list[dict[str, float]]]]:
    grouped: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for csv_path in sorted(rag_root.glob("*/*_k_sensitivity.csv")):
        run_name = csv_path.parent.name
        group = group_for_run(run_name)
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                k = int(row["k"])
                grouped[group][k].append(
                    {
                        "rag": float(row["rag_fused_accuracy"]) * 100.0,
                        "retrieval": float(row["retrieval_accuracy"]) * 100.0,
                        "consistency": float(row["avg_retrieval_consistency"]) * 100.0,
                    }
                )
    if not grouped:
        raise FileNotFoundError(f"No *_k_sensitivity.csv files found under {rag_root}")
    return grouped


def means_and_stds(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def plot(grouped: dict[str, dict[int, list[dict[str, float]]]], output: Path, pad_inches: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)

    for group_key in ["plain_vit", "vit_gem", "sfra_v2"]:
        label, color, marker = GROUPS[group_key]
        by_k = grouped[group_key]
        ks = sorted(by_k)

        rag_means, rag_stds = [], []
        consistency_means, consistency_stds = [], []
        for k in ks:
            rag_mean, rag_std = means_and_stds([item["rag"] for item in by_k[k]])
            consistency_mean, consistency_std = means_and_stds([item["consistency"] for item in by_k[k]])
            rag_means.append(rag_mean)
            rag_stds.append(rag_std)
            consistency_means.append(consistency_mean)
            consistency_stds.append(consistency_std)

        axes[0].errorbar(
            ks,
            rag_means,
            yerr=rag_stds,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4,
            capsize=2,
            label=label,
        )
        axes[1].errorbar(
            ks,
            consistency_means,
            yerr=consistency_stds,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4,
            capsize=2,
            label=label,
        )

    axes[0].set_ylabel("RAG-Fused Accuracy (%)", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Retrieval Consistency (%)", fontsize=9, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Retrieval Depth (k)", fontsize=9, fontweight="bold")
        ax.set_xticks(range(1, 10))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax.tick_params(axis="both", labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    axes[0].set_ylim(86.2, 87.8)
    axes[1].set_ylim(50, 102)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout(pad=0.2, w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"Saved k-sensitivity PDF: {output}")


def main() -> None:
    args = parse_args()
    grouped = load_rows(Path(args.rag_root))
    plot(grouped, Path(args.output), args.pad_inches)


if __name__ == "__main__":
    main()
