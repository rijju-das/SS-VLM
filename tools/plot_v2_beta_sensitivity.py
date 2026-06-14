#!/usr/bin/env python3
"""Plot learned-AU beta sensitivity from saved v2 RAG prediction CSVs.

The prediction CSVs already contain retrieval, AU, and fused distributions for
the run-time beta. This script reconstructs the classifier distribution and
then replays fusion for a range of beta values without rerunning inference.
"""

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
    parser = argparse.ArgumentParser(description="Plot beta sensitivity for learned-AU v2 RAG fusion.")
    parser.add_argument(
        "--rag-root",
        default="outputs_v2/metrics/rag_fusion_learned_au/beta005",
        help="Directory containing per-run *_rag_predictions.csv files.",
    )
    parser.add_argument(
        "--output",
        default="outputs_v2/figures/beta_sensitivity_v2_learned_au.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Retrieval fusion alpha used when the saved RAG outputs were created.",
    )
    parser.add_argument(
        "--source-beta",
        type=float,
        default=0.05,
        help="AU beta used in the saved fused_distribution column.",
    )
    parser.add_argument(
        "--selected-beta",
        type=float,
        default=0.05,
        help="Beta value to highlight as the selected operating point.",
    )
    parser.add_argument(
        "--betas",
        default="0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12,0.15,0.20,0.25,0.30",
        help="Comma-separated beta values to evaluate.",
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


def parse_distribution(text: str) -> list[float]:
    return [float(item) for item in text.split(";")]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def reconstruct_classifier_distribution(
    retrieval_dist: list[float],
    au_dist: list[float],
    fused_dist: list[float],
    alpha: float,
    source_beta: float,
) -> list[float]:
    """Invert saved fusion to recover the classifier probability distribution."""
    base_dist = [(fused_dist[i] - source_beta * au_dist[i]) / (1.0 - source_beta) for i in range(7)]
    classifier_dist = [(base_dist[i] - alpha * retrieval_dist[i]) / (1.0 - alpha) for i in range(7)]
    classifier_dist = [max(0.0, value) for value in classifier_dist]
    total = sum(classifier_dist)
    if total <= 0:
        return [1.0 / 7.0] * 7
    return [value / total for value in classifier_dist]


def evaluate_run(csv_path: Path, betas: list[float], alpha: float, source_beta: float) -> dict[float, float]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No prediction rows found in {csv_path}")

    correct_by_beta = dict.fromkeys(betas, 0)
    for row in rows:
        gt_label = int(row["gt_label"])
        retrieval_dist = parse_distribution(row["retrieval_distribution"])
        au_dist = parse_distribution(row["au_distribution"])
        fused_dist = parse_distribution(row["fused_distribution"])
        classifier_dist = reconstruct_classifier_distribution(
            retrieval_dist=retrieval_dist,
            au_dist=au_dist,
            fused_dist=fused_dist,
            alpha=alpha,
            source_beta=source_beta,
        )
        classifier_retrieval_dist = [
            (1.0 - alpha) * classifier_dist[i] + alpha * retrieval_dist[i] for i in range(7)
        ]

        for beta in betas:
            replayed_dist = [
                (1.0 - beta) * classifier_retrieval_dist[i] + beta * au_dist[i] for i in range(7)
            ]
            correct_by_beta[beta] += int(argmax(replayed_dist) == gt_label)

    return {beta: correct / len(rows) * 100.0 for beta, correct in correct_by_beta.items()}


def load_grouped_results(
    rag_root: Path, betas: list[float], alpha: float, source_beta: float
) -> dict[str, dict[float, list[float]]]:
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    prediction_files = sorted(rag_root.glob("*/*_rag_predictions.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No *_rag_predictions.csv files found under {rag_root}")

    for csv_path in prediction_files:
        run_name = csv_path.parent.name
        group = group_for_run(run_name)
        scores = evaluate_run(csv_path, betas, alpha, source_beta)
        for beta, accuracy in scores.items():
            grouped[group][beta].append(accuracy)
    return grouped


def mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def plot(
    grouped: dict[str, dict[float, list[float]]],
    output: Path,
    selected_beta: float,
    pad_inches: float,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))

    all_means: list[float] = []
    for group_key in ["plain_vit", "vit_gem", "sfra_v2"]:
        label, color, marker = GROUPS[group_key]
        by_beta = grouped[group_key]
        betas = sorted(by_beta)
        means, stds = [], []
        for beta in betas:
            beta_mean, beta_std = mean_std(by_beta[beta])
            means.append(beta_mean)
            stds.append(beta_std)
            all_means.append(beta_mean)

        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.plot(betas, means, color=color, marker=marker, linewidth=1.9, markersize=4, label=label)
        ax.fill_between(betas, lower, upper, color=color, alpha=0.12, linewidth=0)

    ax.axvline(selected_beta, color="#333333", linestyle="--", linewidth=1.0)
    ax.text(
        selected_beta + 0.006,
        max(all_means) - 0.03,
        r"selected $\beta=0.05$",
        fontsize=8,
        color="#333333",
        va="top",
    )

    ax.set_xlabel(r"AU Fusion Weight ($\beta$)", fontsize=9, fontweight="bold")
    ax.set_ylabel("RAG-Fused Accuracy (%)", fontsize=9, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    y_min = min(all_means) - 0.15
    y_max = max(all_means) + 0.15
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.005, 0.305)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.tight_layout(pad=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"Saved beta-sensitivity PDF: {output}")


def main() -> None:
    args = parse_args()
    betas = [float(item.strip()) for item in args.betas.split(",") if item.strip()]
    grouped = load_grouped_results(Path(args.rag_root), betas, args.alpha, args.source_beta)
    plot(grouped, Path(args.output), args.selected_beta, args.pad_inches)


if __name__ == "__main__":
    main()
