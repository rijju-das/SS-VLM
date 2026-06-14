#!/usr/bin/env python3
"""Create Figure 9: same-sample learned-AU RAG model comparison.

The figure is generated from saved learned-AU RAG outputs. It compares Plain
ViT-RAG, ViT+GeM-RAG, and SFRA-RAG on the same held-out RAF-DB test image,
instead of using a hand-written or synthetic qualitative example.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


RUNS = [
    ("plain_vit_ce_seed42", "Plain ViT-RAG", "#FDEEEE", "#B94A48"),
    ("vit_gem_ce_seed42", "ViT+GeM-RAG", "#F2F8EF", "#4E8C45"),
    ("sfra_v2_l005_seed42", "SFRA-RAG", "#ECF8F8", "#2A8C8C"),
]

EMOTION_ABBR = {
    "Surprise": "Sur",
    "Fear": "Fea",
    "Disgust": "Dis",
    "Happiness": "Hap",
    "Sadness": "Sad",
    "Anger": "Ang",
    "Neutral": "Neu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot corrected same-sample Figure 9.")
    parser.add_argument(
        "--root-dir",
        default="outputs_v2/metrics/rag_fusion_learned_au/beta005",
        help="Directory containing learned-AU RAG run subfolders.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=1279,
        help="RAF-DB test sample index to compare across models.",
    )
    parser.add_argument(
        "--output",
        action="append",
        default=[],
        help="Output PDF path. Repeat to save in multiple locations.",
    )
    parser.add_argument("--pad-inches", type=float, default=0.0)
    return parser.parse_args()


def load_prediction(run_dir: Path, run_name: str, sample_index: int) -> dict[str, str]:
    path = run_dir / run_name / f"{run_name}_rag_predictions.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["sample_index"]) == sample_index:
                return row
    raise ValueError(f"sample_index={sample_index} not found in {path}")


def add_box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    title_size: float = 9.2,
    body_size: float = 7.9,
    title_color: str = "#111111",
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.017,
        y + h - 0.043,
        title,
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        va="top",
    )
    ax.text(
        x + 0.017,
        y + h - 0.090,
        body,
        fontsize=body_size,
        color="#111111",
        va="top",
        linespacing=1.16,
    )


def topk_summary(row: dict[str, str]) -> str:
    emotions = [item for item in row["top_k_emotions"].split(";") if item]
    counts = Counter(emotions)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{EMOTION_ABBR.get(label, label[:3])} x{count}" for label, count in ordered)


def distribution_value(row: dict[str, str], label_index: int) -> float:
    values = [float(item) for item in row["fused_distribution"].split(";")]
    return values[label_index]


def status(row: dict[str, str]) -> tuple[str, str]:
    if row["correct"] == "1":
        return "correct", "#1B7F3A"
    return "wrong", "#B00020"


def build_card_body(row: dict[str, str]) -> str:
    correctness, _ = status(row)
    return "\n".join(
        [
            f"Final: {row['pred_emotion']} ({correctness})",
            f"Cls: {row['classifier_pred_emotion']} ({float(row['classifier_confidence']):.3f})",
            f"Ret: {row['retrieval_pred_emotion']} ({float(row['retrieval_confidence']):.3f})",
            f"AU: {row['au_pred_emotion']} ({float(row['au_confidence']):.3f})",
            f"Top-k: {topk_summary(row)}",
        ]
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    root_dir = Path(args.root_dir)
    if not root_dir.is_absolute():
        root_dir = repo_root / root_dir

    rows = {
        run_name: load_prediction(root_dir, run_name, args.sample_index)
        for run_name, _, _, _ in RUNS
    }
    reference = next(iter(rows.values()))
    image_name = Path(reference["image_path"]).name
    gt = reference["gt_emotion"]
    au_source = reference["au_source"]
    au_label = reference["au_pred_emotion"]
    au_conf = float(reference["au_confidence"])

    outputs = [Path(item) for item in args.output]
    if not outputs:
        outputs = [repo_root / "outputs_v2/figures/figure9_learned_au_grounded_report.pdf"]

    fig, ax = plt.subplots(figsize=(7.25, 4.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    header = f"RAF-DB test sample {args.sample_index} ({image_name}); ground truth = {gt}"
    add_box(
        ax,
        (0.02, 0.84),
        (0.96, 0.13),
        "Same-Sample Learned-AU RAG Comparison",
        header,
        facecolor="#EAF2FC",
        edgecolor="#2F5E8C",
        title_size=9.6,
        body_size=8.3,
    )

    shared_body = (
        f"Shared AU prior: {au_label} ({au_conf:.3f}); same fusion rule; "
        "only visual embedding changes."
    )
    add_box(
        ax,
        (0.02, 0.68),
        (0.96, 0.12),
        "Controlled Comparison Setup",
        shared_body,
        facecolor="#F8F8F8",
        edgecolor="#777777",
        title_size=8.8,
        body_size=7.7,
    )

    x_positions = [0.02, 0.345, 0.67]
    for x, (run_name, display_name, fill, edge) in zip(x_positions, RUNS):
        row = rows[run_name]
        correctness, title_color = status(row)
        title = display_name
        add_box(
            ax,
            (x, 0.30),
            (0.305, 0.34),
            title,
            build_card_body(row),
            facecolor=fill,
            edgecolor=edge,
            title_size=8.5,
            body_size=7.35,
            title_color=title_color,
        )

    takeaway = "\n".join(
        [
            "Plain ViT-RAG and ViT+GeM-RAG remain Neutral despite AU evidence for Happiness.",
            "Only SFRA-RAG aligns classifier, retrieval, and AU evidence with the correct label.",
            "Thus, report grounding depends on the visual embedding that populates prototype retrieval.",
        ]
    )
    add_box(
        ax,
        (0.02, 0.05),
        (0.96, 0.20),
        "Grounding and Hallucination-Relevance",
        takeaway,
        facecolor="#FFF7EA",
        edgecolor="#B77729",
        title_size=8.9,
        body_size=7.8,
    )

    for output in outputs:
        if not output.is_absolute():
            output = repo_root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=args.pad_inches)
        print(f"Saved {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
