#!/usr/bin/env python3
"""Generate candidate AU-calibration and RAG-analysis figures for v2 results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt


EMOTIONS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
GROUPS = {
    "plain_vit_ce": ("Plain ViT", "#4C78A8"),
    "vit_gem_ce": ("ViT+GeM", "#F58518"),
    "sfra_v2_l005": ("SFRA-RAG", "#54A24B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create candidate AU/RAG figures from saved v2 outputs.")
    parser.add_argument("--repo-root", default=".", help="Spectral-Symbolic-VLM repo root.")
    parser.add_argument(
        "--au-calibrator-dir",
        default="outputs_v2/au_calibrator/au_both_rf_seed42",
        help="Learned AU prior output directory.",
    )
    parser.add_argument(
        "--au-mapping-dir",
        default="outputs_v2/au_mapping/learned_au_mapping_intensity_elasticnet",
        help="Interpretable AU mapping output directory.",
    )
    parser.add_argument(
        "--manual-rag-root",
        default="outputs_v2/metrics/rag_fusion",
        help="Manual-prior RAG root, used only to read manual AU prior accuracy.",
    )
    parser.add_argument(
        "--learned-rag-root",
        default="outputs_v2/metrics/rag_fusion_learned_au/beta005",
        help="Learned-AU RAG root.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_v2/figures/au_rag_candidates",
        help="Directory where candidate PDFs will be saved.",
    )
    parser.add_argument("--pad-inches", type=float, default=0.0)
    return parser.parse_args()


def save_pdf(fig: plt.Figure, path: Path, pad_inches: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"Saved {path}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def variant_for_run(run_name: str) -> str:
    for key in GROUPS:
        if run_name.startswith(key):
            return key
    raise ValueError(f"Unknown run variant: {run_name}")


def mean_std(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def diverging_heatmap_text_color(value: float, vmax: float) -> str:
    """Keep coefficient labels readable on saturated red/blue heatmap cells."""
    if vmax <= 0:
        return "#111111"
    return "white" if abs(value) / vmax >= 0.52 else "#111111"


def plot_au_mapping_heatmap(mapping_dir: Path, output_dir: Path, pad_inches: float) -> None:
    stability_path = mapping_dir / "learned_au_mapping_stability.csv"
    values: dict[str, dict[str, float]] = defaultdict(dict)
    au_order: list[str] = []
    with stability_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            emotion = row["emotion"]
            au = row["au"]
            if au not in au_order:
                au_order.append(au)
            values[emotion][au] = float(row["mean_coefficient"])

    matrix = [[values[emotion].get(au, 0.0) for au in au_order] for emotion in EMOTIONS]
    vmax = max(abs(item) for row in matrix for item in row)

    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(au_order)))
    ax.set_xticklabels(au_order, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels(EMOTIONS, fontsize=8)
    ax.set_xlabel("OpenFace AU Intensity Feature", fontsize=9, fontweight="bold")
    ax.set_ylabel("Emotion Class", fontsize=9, fontweight="bold")
    ax.tick_params(length=0)

    for y, row in enumerate(matrix):
        ranked = sorted(range(len(row)), key=lambda idx: abs(row[idx]), reverse=True)[:3]
        for x in ranked:
            ax.text(
                x,
                y,
                f"{row[x]:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=diverging_heatmap_text_color(row[x], vmax),
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    cbar.set_label("Mean Standardized Coefficient", fontsize=8, fontweight="bold")
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.25)
    save_pdf(fig, output_dir / "au_mapping_heatmap_learned.pdf", pad_inches)


def load_au_test_predictions(predictions_path: Path) -> list[dict[str, str]]:
    with predictions_path.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("subset") == "test"]
    if not rows:
        raise ValueError(f"No test rows found in {predictions_path}")
    return rows


def plot_au_reliability(au_dir: Path, output_dir: Path, pad_inches: float) -> None:
    rows = load_au_test_predictions(au_dir / "predictions.csv")
    bins = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]
    bin_centres, accuracies, confidences, counts = [], [], [], []
    ece = 0.0
    for low, high in bins:
        selected = [
            row
            for row in rows
            if low <= float(row["confidence"]) < high or (high == 1.0 and float(row["confidence"]) <= high)
        ]
        if selected:
            acc = mean(float(row["correct"]) for row in selected)
            conf = mean(float(row["confidence"]) for row in selected)
        else:
            acc = 0.0
            conf = (low + high) / 2.0
        ece += len(selected) / len(rows) * abs(acc - conf)
        bin_centres.append((low + high) / 2.0)
        accuracies.append(acc * 100.0)
        confidences.append(conf * 100.0)
        counts.append(len(selected))

    fig, ax = plt.subplots(figsize=(3.45, 3.15))
    ax.plot([0, 100], [0, 100], color="#777777", linestyle="--", linewidth=1.0, label="Perfect calibration")
    ax.bar(
        [x * 100.0 for x in bin_centres],
        accuracies,
        width=8.0,
        color="#A0CBE8",
        edgecolor="#4C78A8",
        linewidth=0.8,
        label="Empirical accuracy",
    )
    ax.plot([x * 100.0 for x in bin_centres], confidences, color="#E45756", marker="o", markersize=3.5, label="Mean confidence")
    ax.text(5, 91, f"ECE = {ece * 100:.2f}%", fontsize=8, fontweight="bold")
    ax.set_xlabel("AU Prior Confidence (%)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Empirical Accuracy (%)", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(pad=0.25)
    save_pdf(fig, output_dir / "au_prior_reliability_learned.pdf", pad_inches)


def plot_au_confusion_matrix(au_dir: Path, output_dir: Path, pad_inches: float) -> None:
    matrix_path = au_dir / "test_confusion_matrix.csv"
    matrix: list[list[int]] = []
    row_labels: list[str] = []
    with matrix_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)[1:]
        for row in reader:
            row_labels.append(row[0])
            matrix.append([int(item) for item in row[1:]])

    norm = []
    for row in matrix:
        total = sum(row)
        norm.append([value / total * 100.0 if total else 0.0 for value in row])

    fig, ax = plt.subplots(figsize=(4.4, 3.65))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=max(max(row) for row in norm))
    ax.set_xticks(range(len(header)))
    ax.set_xticklabels(header, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("Predicted Emotion from Learned AU Prior", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Ground Truth Emotion", fontsize=8.5, fontweight="bold")
    ax.tick_params(length=0)

    for y, row in enumerate(norm):
        for x, value in enumerate(row):
            if value >= 10.0 or x == y:
                color = "white" if value > 45 else "#111111"
                ax.text(x, y, f"{value:.0f}", ha="center", va="center", fontsize=5.8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("Row-normalized %", fontsize=8, fontweight="bold")
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.25)
    save_pdf(fig, output_dir / "au_prior_confusion_matrix_learned.pdf", pad_inches)


def plot_au_calibration_summary(mapping_dir: Path, au_dir: Path, output_dir: Path, pad_inches: float) -> None:
    stability_path = mapping_dir / "learned_au_mapping_stability.csv"
    values: dict[str, dict[str, float]] = defaultdict(dict)
    au_order: list[str] = []
    with stability_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            emotion = row["emotion"]
            au = row["au"]
            if au not in au_order:
                au_order.append(au)
            values[emotion][au] = float(row["mean_coefficient"])

    heatmap = [[values[emotion].get(au, 0.0) for au in au_order] for emotion in EMOTIONS]
    vmax = max(abs(item) for row in heatmap for item in row)

    matrix_path = au_dir / "test_confusion_matrix.csv"
    matrix: list[list[int]] = []
    with matrix_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)[1:]
        for row in reader:
            matrix.append([int(item) for item in row[1:]])
    confusion = []
    for row in matrix:
        total = sum(row)
        confusion.append([value / total * 100.0 if total else 0.0 for value in row])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 3.15),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.24},
        constrained_layout=True,
    )

    ax = axes[0]
    im0 = ax.imshow(heatmap, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(au_order)))
    ax.set_xticklabels(au_order, rotation=45, ha="right", fontsize=6.4)
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels(EMOTIONS, fontsize=7.2)
    ax.set_xlabel("AU Intensity Feature", fontsize=8, fontweight="bold")
    ax.set_ylabel("Emotion", fontsize=8, fontweight="bold")
    ax.set_title("(a) Learned AU-emotion coefficients", fontsize=8.3, fontweight="bold", pad=4)
    ax.tick_params(length=0)
    for y, row in enumerate(heatmap):
        ranked = sorted(range(len(row)), key=lambda idx: abs(row[idx]), reverse=True)[:2]
        for x in ranked:
            ax.text(
                x,
                y,
                f"{row[x]:.2f}",
                ha="center",
                va="center",
                fontsize=5.0,
                color=diverging_heatmap_text_color(row[x], vmax),
            )
    cbar0 = fig.colorbar(im0, ax=ax, fraction=0.038, pad=0.012)
    cbar0.set_label("Coeff.", fontsize=6.8, fontweight="bold")
    cbar0.ax.tick_params(labelsize=6.2)

    ax = axes[1]
    im1 = ax.imshow(confusion, cmap="Blues", vmin=0, vmax=max(max(row) for row in confusion))
    ax.set_xticks(range(len(header)))
    ax.set_xticklabels(header, rotation=45, ha="right", fontsize=6.4)
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels(EMOTIONS, fontsize=7.2)
    ax.set_xlabel("AU-prior prediction", fontsize=8, fontweight="bold")
    ax.set_ylabel("Ground truth", fontsize=8, fontweight="bold")
    ax.set_title("(b) Learned AU-prior confusion", fontsize=8.3, fontweight="bold", pad=4)
    ax.tick_params(length=0)
    for y, row in enumerate(confusion):
        for x, value in enumerate(row):
            if value >= 12.0 or x == y:
                color = "white" if value > 45 else "#111111"
                ax.text(x, y, f"{value:.0f}", ha="center", va="center", fontsize=5.0, color=color)
    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.012)
    cbar1.set_label("Row %", fontsize=6.8, fontweight="bold")
    cbar1.ax.tick_params(labelsize=6.2)

    save_pdf(fig, output_dir / "au_calibration_heatmap_confusion_learned.pdf", pad_inches)


def get_manual_au_accuracy(manual_rag_root: Path) -> float | None:
    values = []
    for summary_path in sorted(manual_rag_root.glob("*/*_rag_summary.json")):
        summary = read_json(summary_path)
        value = summary.get("au_accuracy_on_available")
        if value is not None:
            values.append(float(value) * 100.0)
    return mean(values) if values else None


def plot_au_prior_upgrade(
    au_dir: Path,
    mapping_dir: Path,
    manual_rag_root: Path,
    output_dir: Path,
    pad_inches: float,
) -> None:
    manual_acc = get_manual_au_accuracy(manual_rag_root)
    mapping_summary = read_json(mapping_dir / "summary.json")
    au_metrics = read_json(au_dir / "metrics.json")

    labels = []
    values = []
    colors = []
    if manual_acc is not None:
        labels.append("Manual\nFACS prior")
        values.append(manual_acc)
        colors.append("#BAB0AC")
    labels.append("Sparse learned\nAU mapping")
    values.append(float(mapping_summary["mean_test_accuracy"]) * 100.0)
    colors.append("#B279A2")
    labels.append("Learned AU\nRandomForest prior")
    values.append(float(au_metrics["test"]["accuracy"]) * 100.0)
    colors.append("#59A14F")

    fig, ax = plt.subplots(figsize=(3.8, 2.75))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="#333333", linewidth=0.7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AU-only Emotion Accuracy (%)", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(values) + 10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(pad=0.25)
    save_pdf(fig, output_dir / "au_prior_upgrade_comparison.pdf", pad_inches)


def load_rag_prediction_rows(rag_root: Path) -> dict[str, list[dict[str, str]]]:
    runs: dict[str, list[dict[str, str]]] = {}
    for prediction_path in sorted(rag_root.glob("*/*_rag_predictions.csv")):
        with prediction_path.open("r", encoding="utf-8", newline="") as f:
            runs[prediction_path.parent.name] = list(csv.DictReader(f))
    if not runs:
        raise FileNotFoundError(f"No *_rag_predictions.csv files found under {rag_root}")
    return runs


def rag_run_stats(rows: list[dict[str, str]]) -> dict[str, float]:
    fixes = harms = 0
    classifier_correct = retrieval_correct = rag_correct = 0
    for row in rows:
        gt = row["gt_label"]
        classifier_ok = row["classifier_pred_label"] == gt
        retrieval_ok = row["retrieval_pred_label"] == gt
        rag_ok = row["pred_label"] == gt
        classifier_correct += int(classifier_ok)
        retrieval_correct += int(retrieval_ok)
        rag_correct += int(rag_ok)
        fixes += int((not classifier_ok) and rag_ok)
        harms += int(classifier_ok and (not rag_ok))
    n = len(rows)
    return {
        "fixes": float(fixes),
        "harms": float(harms),
        "net": float(fixes - harms),
        "classifier_accuracy": classifier_correct / n * 100.0,
        "retrieval_accuracy": retrieval_correct / n * 100.0,
        "rag_accuracy": rag_correct / n * 100.0,
    }


def grouped_rag_stats(rag_root: Path) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for run_name, rows in load_rag_prediction_rows(rag_root).items():
        grouped[variant_for_run(run_name)].append(rag_run_stats(rows))
    return grouped


def plot_rag_correction_flow(rag_root: Path, output_dir: Path, pad_inches: float) -> None:
    grouped = grouped_rag_stats(rag_root)
    variants = ["plain_vit_ce", "vit_gem_ce", "sfra_v2_l005"]
    metrics = [("fixes", "Fixes", "#59A14F"), ("harms", "Harms", "#E15759"), ("net", "Net", "#4C78A8")]
    x_positions = list(range(len(variants)))
    width = 0.22

    fig, ax = plt.subplots(figsize=(4.75, 2.9))
    for offset_idx, (key, label, color) in enumerate(metrics):
        xs = [x + (offset_idx - 1) * width for x in x_positions]
        means, stds = [], []
        for variant in variants:
            metric_values = [item[key] for item in grouped[variant]]
            metric_mean, metric_std = mean_std(metric_values)
            means.append(metric_mean)
            stds.append(metric_std)
        bars = ax.bar(xs, means, width=width, yerr=stds, capsize=2, color=color, edgecolor="#333333", linewidth=0.6, label=label)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + (1.2 if value >= 0 else -2.5), f"{value:.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([GROUPS[item][0] for item in variants], fontsize=8)
    ax.set_ylabel("Images (mean over seeds)", fontsize=9, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout(pad=0.25)
    save_pdf(fig, output_dir / "rag_fix_harm_net_learned_au.pdf", pad_inches)


def plot_rag_component_accuracy(rag_root: Path, output_dir: Path, pad_inches: float) -> None:
    grouped = grouped_rag_stats(rag_root)
    variants = ["plain_vit_ce", "vit_gem_ce", "sfra_v2_l005"]
    metrics = [
        ("classifier_accuracy", "Classifier", "#9C755F"),
        ("retrieval_accuracy", "Retrieval", "#76B7B2"),
        ("rag_accuracy", "RAG fused", "#4C78A8"),
    ]
    width = 0.23
    x_positions = list(range(len(variants)))

    fig, ax = plt.subplots(figsize=(4.8, 2.85))
    for offset_idx, (key, label, color) in enumerate(metrics):
        xs = [x + (offset_idx - 1) * width for x in x_positions]
        means, stds = [], []
        for variant in variants:
            vals = [item[key] for item in grouped[variant]]
            metric_mean, metric_std = mean_std(vals)
            means.append(metric_mean)
            stds.append(metric_std)
        bars = ax.bar(xs, means, width=width, yerr=stds, capsize=2, color=color, edgecolor="#333333", linewidth=0.6, label=label)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.2f}", ha="center", fontsize=6.5, rotation=90)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([GROUPS[item][0] for item in variants], fontsize=8)
    ax.set_ylabel("RAF-DB Test Accuracy (%)", fontsize=9, fontweight="bold")
    ax.set_ylim(86.5, 87.7)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout(pad=0.25)
    save_pdf(fig, output_dir / "rag_component_accuracy_learned_au.pdf", pad_inches)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = repo_root / args.output_dir
    au_dir = repo_root / args.au_calibrator_dir
    mapping_dir = repo_root / args.au_mapping_dir
    manual_rag_root = repo_root / args.manual_rag_root
    learned_rag_root = repo_root / args.learned_rag_root

    plot_au_mapping_heatmap(mapping_dir, output_dir, args.pad_inches)
    plot_au_reliability(au_dir, output_dir, args.pad_inches)
    plot_au_confusion_matrix(au_dir, output_dir, args.pad_inches)
    plot_au_calibration_summary(mapping_dir, au_dir, output_dir, args.pad_inches)
    plot_au_prior_upgrade(au_dir, mapping_dir, manual_rag_root, output_dir, args.pad_inches)
    plot_rag_correction_flow(learned_rag_root, output_dir, args.pad_inches)
    plot_rag_component_accuracy(learned_rag_root, output_dir, args.pad_inches)


if __name__ == "__main__":
    main()
