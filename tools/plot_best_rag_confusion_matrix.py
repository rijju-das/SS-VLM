#!/usr/bin/env python3
"""Plot the confusion matrix for the best RAG-fused v2 run.

The script scans v2 RAG summary JSON files, selects the run with the highest
RAG-fused accuracy, and saves a tightly cropped PDF confusion matrix.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Keep matplotlib/font caches inside writable temp locations on shared servers.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np


RAFDB_SHORT_LABELS = ["Sur", "Fea", "Dis", "Hap", "Sad", "Ang", "Neu"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a tight PDF confusion matrix for the best v2 RAG-fused run."
    )
    parser.add_argument(
        "--rag-root",
        default="outputs_v2/metrics/rag_fusion",
        help="Directory containing per-run *_rag_summary.json files.",
    )
    parser.add_argument(
        "--output",
        default="outputs_v2/figures/best_rag_confusion_matrix.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--matrix-key",
        choices=["confusion_matrix", "classifier_confusion_matrix"],
        default="confusion_matrix",
        help="Use RAG-fused confusion matrix or classifier-only confusion matrix.",
    )
    parser.add_argument(
        "--normalize",
        choices=["row", "none"],
        default="row",
        help="Row-normalize the matrix by true class support or plot raw counts.",
    )
    parser.add_argument(
        "--title",
        action="store_true",
        help="Add a compact title with run name and accuracy.",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Disable the colorbar for an even tighter figure.",
    )
    parser.add_argument(
        "--pad-inches",
        type=float,
        default=0.0,
        help="Extra padding passed to savefig. Use 0 for no white border.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_best_summary(rag_root: Path) -> tuple[str, Path, dict]:
    summaries = []
    for summary_path in sorted(rag_root.glob("*/*_rag_summary.json")):
        summary = load_json(summary_path)
        accuracy = summary.get("accuracy")
        if accuracy is None:
            continue
        run_name = summary_path.name.removesuffix("_rag_summary.json")
        summaries.append((float(accuracy), run_name, summary_path, summary))

    if not summaries:
        raise FileNotFoundError(f"No *_rag_summary.json files found under {rag_root}")

    accuracy, run_name, summary_path, summary = max(summaries, key=lambda item: item[0])
    print(f"Best RAG-fused run: {run_name} ({accuracy * 100:.2f}%)")
    print(f"Summary JSON: {summary_path}")
    return run_name, summary_path, summary


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )


def plot_confusion_matrix(
    matrix: np.ndarray,
    run_name: str,
    accuracy: float,
    output_path: Path,
    normalize: str,
    show_title: bool,
    show_colorbar: bool,
    pad_inches: float,
) -> None:
    values = row_normalize(matrix) if normalize == "row" else matrix.astype(float)
    vmax = 1.0 if normalize == "row" else float(values.max())
    annot_format = ".2f" if normalize == "row" else ".0f"

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=vmax, interpolation="nearest")

    ax.set_xticks(np.arange(len(RAFDB_SHORT_LABELS)))
    ax.set_yticks(np.arange(len(RAFDB_SHORT_LABELS)))
    ax.set_xticklabels(RAFDB_SHORT_LABELS, fontsize=10)
    ax.set_yticklabels(RAFDB_SHORT_LABELS, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=4)
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold", labelpad=4)
    ax.tick_params(axis="both", which="both", length=0)

    if show_title:
        ax.set_title(f"{run_name} RAG-fused ({accuracy * 100:.2f}%)", fontsize=11, pad=6)

    threshold = vmax * 0.55
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if values[i, j] >= threshold else "#222222"
            ax.text(
                j,
                i,
                format(values[i, j], annot_format),
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_colorbar:
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        colorbar.ax.tick_params(labelsize=9, length=0)
        colorbar.outline.set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"Saved tight PDF: {output_path}")


def main() -> None:
    args = parse_args()
    rag_root = Path(args.rag_root)
    output_path = Path(args.output)

    run_name, _summary_path, summary = find_best_summary(rag_root)
    matrix = np.asarray(summary[args.matrix_key], dtype=float)
    if matrix.shape != (7, 7):
        raise ValueError(f"Expected a 7x7 RAF-DB matrix, got {matrix.shape}")

    plot_confusion_matrix(
        matrix=matrix,
        run_name=run_name,
        accuracy=float(summary["accuracy"]),
        output_path=output_path,
        normalize=args.normalize,
        show_title=args.title,
        show_colorbar=not args.no_colorbar,
        pad_inches=args.pad_inches,
    )


if __name__ == "__main__":
    main()
