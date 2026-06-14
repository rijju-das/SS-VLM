#!/usr/bin/env python3
"""Plot zero-shot VLM confusion matrices from saved RAF-DB metrics JSON files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = {
    "instructblip": {
        "metrics": "outputs_vlm/zero_shot/rafdb/instructblip_fixed/metrics.json",
        "paper_output": "../2026_IEEE_SSVLM_Riju/Figure/instructblip_rafdb.pdf",
        "repo_output": "outputs_vlm/zero_shot/rafdb/instructblip_fixed/confusion_matrix.pdf",
    },
    "llava15": {
        "metrics": "outputs_vlm/zero_shot/rafdb/llava15_fixed/metrics.json",
        "paper_output": "../2026_IEEE_SSVLM_Riju/Figure/llava_rafdb.pdf",
        "repo_output": "outputs_vlm/zero_shot/rafdb/llava15_fixed/confusion_matrix.pdf",
    },
}

DEFAULT_COMBINED_OUTPUTS = [
    "../2026_IEEE_SSVLM_Riju/Figure/vlm_confusion_matrices_rafdb.pdf",
    "outputs_vlm/zero_shot/rafdb/vlm_confusion_matrices_rafdb.pdf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create tight-cropped RAF-DB zero-shot confusion matrix PDFs."
    )
    parser.add_argument(
        "--run",
        choices=sorted(DEFAULT_RUNS),
        default=None,
        help="Which fixed zero-shot run to plot.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Plot InstructBLIP and LLaVA-1.5 side by side with one shared colorbar.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Optional override for the metrics JSON path.",
    )
    parser.add_argument(
        "--output",
        action="append",
        default=None,
        help="Output PDF path. Repeat to save paper and repo copies.",
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Plot raw counts instead of row-normalized percentages.",
    )
    parser.add_argument(
        "--pad-inches",
        type=float,
        default=0.0,
        help="Whitespace padding around the saved PDF. Use 0 for a tight crop.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> tuple[list[str], np.ndarray, float]:
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    labels = metrics["labels"]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    accuracy = float(metrics["accuracy"]) * 100.0
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(
            f"Confusion matrix shape {matrix.shape} does not match {len(labels)} labels"
        )
    return labels, matrix, accuracy


def plot_matrix(
    labels: list[str],
    counts: np.ndarray,
    accuracy: float,
    output: Path,
    normalize: bool,
    pad_inches: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        row_totals = counts.sum(axis=1, keepdims=True)
        values = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)
        display_values = values * 100.0
        colorbar_label = "Row-normalized (%)"
        vmax = 100.0
    else:
        display_values = counts
        colorbar_label = "Count"
        vmax = max(float(display_values.max()), 1.0)

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(display_values, cmap="Blues", vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted label", fontsize=8, fontweight="bold")
    ax.set_ylabel("True label", fontsize=8, fontweight="bold")

    threshold = vmax * 0.5
    for row in range(display_values.shape[0]):
        for col in range(display_values.shape[1]):
            value = display_values[row, col]
            text = f"{value:.0f}%" if normalize else f"{int(value)}"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if value > threshold else "#1f2933",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(colorbar_label, fontsize=7)

    fig.tight_layout(pad=0.05)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"Saved confusion matrix: {output}")


def normalized_or_count_matrix(counts: np.ndarray, normalize: bool) -> tuple[np.ndarray, str, float]:
    if normalize:
        row_totals = counts.sum(axis=1, keepdims=True)
        values = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)
        return values * 100.0, "Row-normalized (%)", 100.0
    values = counts
    return values, "Count", max(float(values.max()), 1.0)


def draw_matrix_panel(
    ax: plt.Axes,
    labels: list[str],
    display_values: np.ndarray,
    vmax: float,
    show_ylabel: bool = True,
) -> None:
    im = ax.imshow(display_values, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted label", fontsize=8, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("True label", fontsize=8, fontweight="bold")

    threshold = vmax * 0.5
    is_percent = vmax == 100.0
    for row in range(display_values.shape[0]):
        for col in range(display_values.shape[1]):
            value = display_values[row, col]
            text = f"{value:.0f}%" if is_percent else f"{int(value)}"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=6.1,
                color="white" if value > threshold else "#1f2933",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def plot_combined_matrices(outputs: list[Path], normalize: bool, pad_inches: float) -> None:
    run_order = [
        "instructblip",
        "llava15",
    ]
    loaded = []
    vmax = 100.0 if normalize else 1.0
    colorbar_label = "Row-normalized (%)" if normalize else "Count"
    for run_key in run_order:
        labels, counts, _accuracy = load_metrics(Path(DEFAULT_RUNS[run_key]["metrics"]))
        display_values, colorbar_label, local_vmax = normalized_or_count_matrix(counts, normalize)
        vmax = max(vmax, local_vmax)
        loaded.append((labels, display_values))

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45), constrained_layout=True)
        image = None
        for index, (ax, (labels, display_values)) in enumerate(zip(axes, loaded)):
            image = draw_matrix_panel(
                ax,
                labels,
                display_values,
                vmax,
                show_ylabel=index == 0,
            )

        cbar = fig.colorbar(image, ax=axes, fraction=0.032, pad=0.018)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(colorbar_label, fontsize=7)

        fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=pad_inches)
        plt.close(fig)
        print(f"Saved combined confusion matrices: {output}")


def main() -> None:
    args = parse_args()
    if args.combined:
        outputs = [Path(p) for p in (args.output or DEFAULT_COMBINED_OUTPUTS)]
        plot_combined_matrices(
            outputs=outputs,
            normalize=not args.counts,
            pad_inches=args.pad_inches,
        )
        return

    if args.run is None:
        raise SystemExit("Specify --run for a single plot or --combined for the shared-colorbar figure.")

    defaults = DEFAULT_RUNS[args.run]
    metrics_path = Path(args.metrics or defaults["metrics"])
    outputs = [Path(p) for p in (args.output or [defaults["paper_output"], defaults["repo_output"]])]

    labels, counts, accuracy = load_metrics(metrics_path)
    for output in outputs:
        plot_matrix(
            labels=labels,
            counts=counts,
            accuracy=accuracy,
            output=output,
            normalize=not args.counts,
            pad_inches=args.pad_inches,
        )


if __name__ == "__main__":
    main()
