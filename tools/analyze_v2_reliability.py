#!/usr/bin/env python3
"""Summarize v2 per-class, correction-flow, and calibration reliability metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


GROUPS = {
    "Plain ViT-RAG": ["plain_vit_ce_seed42", "plain_vit_ce_seed123", "plain_vit_ce_seed2026"],
    "ViT+GeM-RAG": ["vit_gem_ce_seed42", "vit_gem_ce_seed123", "vit_gem_ce_seed2026"],
    "SFRA-RAG": ["sfra_v2_l005_seed42", "sfra_v2_l005_seed123", "sfra_v2_l005_seed2026"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SS-VLM v2 reliability outputs.")
    parser.add_argument("--rag-root", default="outputs_v2/metrics/rag_fusion_learned_au/beta005")
    parser.add_argument("--best-run", default="sfra_v2_l005_seed42")
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs_v2/metrics/reliability_analysis_learned_au")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float) -> float:
    return float(value) * 100.0


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) <= 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def expected_calibration_error(rows: list[dict[str, str]], correct_col: str, confidence_col: str, bins: int) -> float:
    confidences = np.array([float(row[confidence_col]) for row in rows], dtype=float)
    correctness = np.array([int(row[correct_col]) for row in rows], dtype=float)
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
        bin_weight = float(mask.mean())
        bin_accuracy = float(correctness[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
    return ece


def correction_flow(rows: list[dict[str, str]]) -> dict[str, int]:
    fixes = harms = both_correct = both_wrong = 0
    for row in rows:
        classifier_correct = bool(int(row["classifier_correct"]))
        rag_correct = bool(int(row["correct"]))
        fixes += int((not classifier_correct) and rag_correct)
        harms += int(classifier_correct and (not rag_correct))
        both_correct += int(classifier_correct and rag_correct)
        both_wrong += int((not classifier_correct) and (not rag_correct))
    return {
        "fixes": fixes,
        "harms": harms,
        "net": fixes - harms,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rag_root = Path(args.rag_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows = []
    group_rows = []
    calibration_rows = []

    for group_name, runs in GROUPS.items():
        group_fixes = []
        group_harms = []
        group_net = []
        for run in runs:
            run_dir = rag_root / run
            predictions = read_csv(run_dir / f"{run}_rag_predictions.csv")
            summary = read_json(run_dir / f"{run}_rag_summary.json")
            flow = correction_flow(predictions)
            rag_ece = expected_calibration_error(predictions, "correct", "confidence", args.ece_bins)
            classifier_ece = expected_calibration_error(
                predictions, "classifier_correct", "classifier_confidence", args.ece_bins
            )
            retrieval_ece = expected_calibration_error(
                predictions, "retrieval_correct", "retrieval_confidence", args.ece_bins
            )

            per_run_rows.append(
                {
                    "group": group_name,
                    "run": run,
                    "classifier_accuracy": pct(summary["classifier_accuracy"]),
                    "rag_accuracy": pct(summary["accuracy"]),
                    "fixes": flow["fixes"],
                    "harms": flow["harms"],
                    "net": flow["net"],
                    "rag_ece": pct(rag_ece),
                    "classifier_ece": pct(classifier_ece),
                    "retrieval_ece": pct(retrieval_ece),
                }
            )
            calibration_rows.append(per_run_rows[-1])
            group_fixes.append(flow["fixes"])
            group_harms.append(flow["harms"])
            group_net.append(flow["net"])

        fixes_mean, fixes_std = mean_std(group_fixes)
        harms_mean, harms_std = mean_std(group_harms)
        net_mean, net_std = mean_std(group_net)
        group_rows.append(
            {
                "group": group_name,
                "fixes_mean": fixes_mean,
                "fixes_std": fixes_std,
                "harms_mean": harms_mean,
                "harms_std": harms_std,
                "net_mean": net_mean,
                "net_std": net_std,
            }
        )

    best_summary = read_json(rag_root / args.best_run / f"{args.best_run}_rag_summary.json")
    per_class_rows = []
    for class_name, metrics in best_summary["classification_report"].items():
        if not isinstance(metrics, dict) or class_name in {"macro avg", "weighted avg"}:
            continue
        per_class_rows.append(
            {
                "class": class_name,
                "precision": pct(metrics["precision"]),
                "recall": pct(metrics["recall"]),
                "f1": pct(metrics["f1-score"]),
                "support": int(metrics["support"]),
            }
        )

    best_macro = best_summary["classification_report"]["macro avg"]
    best_weighted = best_summary["classification_report"]["weighted avg"]
    best_predictions = read_csv(rag_root / args.best_run / f"{args.best_run}_rag_predictions.csv")
    best_calibration = {
        "run": args.best_run,
        "ece_bins": args.ece_bins,
        "rag_ece": pct(expected_calibration_error(best_predictions, "correct", "confidence", args.ece_bins)),
        "classifier_ece": pct(
            expected_calibration_error(best_predictions, "classifier_correct", "classifier_confidence", args.ece_bins)
        ),
        "retrieval_ece": pct(
            expected_calibration_error(best_predictions, "retrieval_correct", "retrieval_confidence", args.ece_bins)
        ),
        "macro_f1": pct(best_macro["f1-score"]),
        "weighted_f1": pct(best_weighted["f1-score"]),
    }

    write_csv(
        output_dir / "correction_flow_by_run.csv",
        per_run_rows,
        [
            "group",
            "run",
            "classifier_accuracy",
            "rag_accuracy",
            "fixes",
            "harms",
            "net",
            "rag_ece",
            "classifier_ece",
            "retrieval_ece",
        ],
    )
    write_csv(
        output_dir / "correction_flow_by_group.csv",
        group_rows,
        ["group", "fixes_mean", "fixes_std", "harms_mean", "harms_std", "net_mean", "net_std"],
    )
    write_csv(
        output_dir / "best_sfra_per_class.csv",
        per_class_rows,
        ["class", "precision", "recall", "f1", "support"],
    )
    write_csv(
        output_dir / "calibration_ece_by_run.csv",
        calibration_rows,
        [
            "group",
            "run",
            "classifier_accuracy",
            "rag_accuracy",
            "fixes",
            "harms",
            "net",
            "rag_ece",
            "classifier_ece",
            "retrieval_ece",
        ],
    )

    payload = {
        "best_run": args.best_run,
        "best_run_summary": best_calibration,
        "per_class": per_class_rows,
        "correction_flow_by_group": group_rows,
        "correction_flow_by_run": per_run_rows,
    }
    with (output_dir / "v2_reliability_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Best run: {args.best_run}")
    print(
        "Best SFRA-RAG: "
        f"macro F1={best_calibration['macro_f1']:.2f}%, "
        f"weighted F1={best_calibration['weighted_f1']:.2f}%, "
        f"RAG ECE={best_calibration['rag_ece']:.2f}%, "
        f"classifier ECE={best_calibration['classifier_ece']:.2f}%, "
        f"retrieval ECE={best_calibration['retrieval_ece']:.2f}%"
    )
    for row in group_rows:
        print(
            f"{row['group']}: fixes={row['fixes_mean']:.1f}, "
            f"harms={row['harms_mean']:.1f}, net={row['net_mean']:+.1f}"
        )
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
