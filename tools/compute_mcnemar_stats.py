#!/usr/bin/env python3
"""Compute exact paired McNemar statistics for v2 RAG predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute exact McNemar tests for SS-VLM v2.")
    parser.add_argument(
        "--sfra-predictions",
        default="outputs_v2/metrics/rag_fusion/sfra_v2_l005_seed42/sfra_v2_l005_seed42_rag_predictions.csv",
        help="Best SFRA-RAG prediction CSV.",
    )
    parser.add_argument(
        "--plain-predictions",
        default="outputs_v2/metrics/rag_fusion/plain_vit_ce_seed42/plain_vit_ce_seed42_rag_predictions.csv",
        help="Same-seed Plain ViT-RAG prediction CSV.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs_v2/metrics/statistical_tests/mcnemar_sfra_seed42.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs_v2/metrics/statistical_tests/mcnemar_sfra_seed42.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def read_predictions(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {int(row["sample_index"]): row for row in csv.DictReader(f)}


def exact_mcnemar_pvalue(losses: int, wins: int) -> float:
    """Two-sided exact McNemar p-value via a Binomial(n, 0.5) tail."""
    total_discordant = int(losses) + int(wins)
    if total_discordant == 0:
        return 1.0
    smaller = min(int(losses), int(wins))
    tail = sum(math.comb(total_discordant, k) for k in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**total_discordant))


def bool_int(value: Any) -> bool:
    return bool(int(str(value)))


def compare_sfra_rag_vs_classifier(sfra_rows: dict[int, dict[str, str]]) -> dict[str, Any]:
    fixes = harms = both_correct = both_wrong = 0
    for row in sfra_rows.values():
        classifier_correct = bool_int(row["classifier_correct"])
        rag_correct = bool_int(row["correct"])
        fixes += int((not classifier_correct) and rag_correct)
        harms += int(classifier_correct and (not rag_correct))
        both_correct += int(classifier_correct and rag_correct)
        both_wrong += int((not classifier_correct) and (not rag_correct))
    return {
        "comparison": "SFRA-RAG seed42 vs SFRA classifier seed42",
        "wins": fixes,
        "losses": harms,
        "net_wins": fixes - harms,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "discordant_pairs": fixes + harms,
        "exact_mcnemar_p": exact_mcnemar_pvalue(harms, fixes),
    }


def compare_sfra_rag_vs_plain_rag(
    sfra_rows: dict[int, dict[str, str]],
    plain_rows: dict[int, dict[str, str]],
) -> dict[str, Any]:
    missing = sorted(set(sfra_rows) ^ set(plain_rows))
    if missing:
        raise ValueError(f"Prediction files do not have matching sample indices; first mismatch: {missing[:5]}")

    sfra_wins = plain_wins = both_correct = both_wrong = 0
    for sample_index in sorted(sfra_rows):
        sfra_correct = bool_int(sfra_rows[sample_index]["correct"])
        plain_correct = bool_int(plain_rows[sample_index]["correct"])
        sfra_wins += int(sfra_correct and (not plain_correct))
        plain_wins += int(plain_correct and (not sfra_correct))
        both_correct += int(sfra_correct and plain_correct)
        both_wrong += int((not sfra_correct) and (not plain_correct))
    return {
        "comparison": "SFRA-RAG seed42 vs Plain ViT-RAG seed42",
        "wins": sfra_wins,
        "losses": plain_wins,
        "net_wins": sfra_wins - plain_wins,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "discordant_pairs": sfra_wins + plain_wins,
        "exact_mcnemar_p": exact_mcnemar_pvalue(plain_wins, sfra_wins),
    }


def write_json(rows: list[dict[str, Any]], path: Path, inputs: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "test": "two-sided exact McNemar",
        "inputs": inputs,
        "results": rows,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "comparison",
        "wins",
        "losses",
        "net_wins",
        "both_correct",
        "both_wrong",
        "discordant_pairs",
        "exact_mcnemar_p",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    sfra_path = Path(args.sfra_predictions)
    plain_path = Path(args.plain_predictions)
    sfra_rows = read_predictions(sfra_path)
    plain_rows = read_predictions(plain_path)
    results = [
        compare_sfra_rag_vs_classifier(sfra_rows),
        compare_sfra_rag_vs_plain_rag(sfra_rows, plain_rows),
    ]

    write_json(
        results,
        Path(args.output_json),
        {
            "sfra_predictions": str(sfra_path),
            "plain_predictions": str(plain_path),
        },
    )
    write_csv(results, Path(args.output_csv))

    for result in results:
        print(
            f"{result['comparison']}: wins={result['wins']} losses={result['losses']} "
            f"net={result['net_wins']} p={result['exact_mcnemar_p']:.6f}"
        )
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
