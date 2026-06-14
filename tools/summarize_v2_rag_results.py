#!/usr/bin/env python3
"""Summarize v2 RAG and hallucination outputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("outputs_v2/metrics/rag_fusion")


def pct(value):
    if value is None:
        return ""
    return f"{float(value) * 100:.2f}"


def main() -> None:
    rows = []
    for summary_path in sorted(ROOT.glob("*/*_rag_summary.json")):
        run_dir = summary_path.parent
        run_name = summary_path.name.removesuffix("_rag_summary.json")
        hallucination_path = run_dir / f"{run_name}_hallucination_metrics.json"
        summary = json.loads(summary_path.read_text())
        hallucination = json.loads(hallucination_path.read_text()) if hallucination_path.exists() else {}
        rows.append(
            {
                "run": run_name,
                "classifier": summary.get("classifier_accuracy"),
                "retrieval": summary.get("retrieval_accuracy"),
                "rag": summary.get("accuracy"),
                "rag_delta": (
                    summary.get("accuracy") - summary.get("classifier_accuracy")
                    if summary.get("accuracy") is not None
                    and summary.get("classifier_accuracy") is not None
                    else None
                ),
                "macro_f1": summary.get("classification_report", {}).get("macro avg", {}).get("f1-score"),
                "faithfulness": hallucination.get("evidence_faithfulness_rate"),
                "malformed_au": hallucination.get("malformed_au_mention_rate"),
                "clinical_unsupported": hallucination.get("unsupported_clinical_claim_rate"),
                "contradiction": hallucination.get("contradiction_rate"),
                "llm_judge": hallucination.get("llm_judge_faithfulness_mean"),
                "reports": hallucination.get("total_reports"),
            }
        )

    if not rows:
        print("No v2 RAG summaries found.")
        return

    print(
        "\t".join(
            [
                "run",
                "classifier",
                "retrieval",
                "rag",
                "rag_delta",
                "macro_f1",
                "faithfulness",
                "malformed_au",
                "clinical_unsupported",
                "contradiction",
                "llm_judge",
                "reports",
            ]
        )
    )
    for row in rows:
        print(
            "\t".join(
                [
                    row["run"],
                    pct(row["classifier"]),
                    pct(row["retrieval"]),
                    pct(row["rag"]),
                    pct(row["rag_delta"]),
                    pct(row["macro_f1"]),
                    pct(row["faithfulness"]),
                    pct(row["malformed_au"]),
                    pct(row["clinical_unsupported"]),
                    pct(row["contradiction"]),
                    pct(row["llm_judge"]),
                    str(row["reports"] or ""),
                ]
            )
        )


if __name__ == "__main__":
    main()
