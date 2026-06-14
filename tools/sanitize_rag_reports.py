#!/usr/bin/env python3
"""Replace unsafe RAG reports with deterministic evidence-only fallbacks.

This is useful when older saved reports were generated before the strict AU
guardrail was added to the pipeline. It does not change predictions; it only
replaces report text that mentions invalid or unsupported AU identifiers.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evaluate_hallucination_metrics import extract_aus, extract_malformed_au_mentions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize saved SS-VLM RAG report JSON files.")
    parser.add_argument(
        "--root",
        default="outputs_v2/metrics/rag_fusion",
        help="Root containing run subfolders with *_rag_sample_reports.json files.",
    )
    parser.add_argument(
        "--reports-json",
        action="append",
        default=[],
        help="Specific report JSON to sanitize; repeat to pass multiple files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report changes.")
    return parser.parse_args()


def report_files(args: argparse.Namespace) -> list[Path]:
    explicit = [Path(path).expanduser() for path in args.reports_json]
    if explicit:
        return explicit
    return sorted(Path(args.root).expanduser().glob("*/*_rag_sample_reports.json"))


def prototype_summary(evidence: str) -> str:
    labels = []
    for line in evidence.splitlines():
        match = re.match(r"\d+\.\s+([A-Za-z]+)\s+prototype", line.strip())
        if match:
            labels.append(match.group(1))
    return ", ".join(labels[:9]) if labels else "not available"


def au_summary(evidence: str) -> str:
    active_aus = sorted(set(re.findall(r"\bAU\d{2}=\d+(?:\.\d+)?", evidence)))
    if active_aus:
        return "Explicit active AU evidence: " + ", ".join(active_aus[:8]) + "."
    if "no AU mean above" in evidence:
        return "The retrieved evidence reports no AU mean above the active threshold for the listed prototypes."
    if "FACS-informed class prior" in evidence:
        return "The retrieved evidence provides FACS-informed class priors rather than image-specific AU intensities."
    return "No explicit AU identifier should be inferred beyond the retrieved evidence."


def safe_report(item: dict[str, Any]) -> str:
    pred = str(item.get("pred") or "the predicted class")
    classifier_pred = str(item.get("classifier_pred") or pred)
    retrieval_pred = str(item.get("retrieval_pred") or pred)
    evidence = str(item.get("retrieved_evidence") or "")
    confidence = item.get("confidence")
    try:
        confidence_text = f"{float(confidence):.2f}"
    except (TypeError, ValueError):
        confidence_text = "not recorded"

    support_text = "limited or mixed" if classifier_pred != retrieval_pred else "consistent"
    return (
        "Observation: "
        f"The final fused expression is {pred} with confidence {confidence_text}. "
        f"The classifier prediction is {classifier_pred}, and the retrieval-majority prediction is {retrieval_pred}.\n\n"
        "Evidence:\n"
        f"1. Retrieved prototype labels in rank order: {prototype_summary(evidence)}.\n"
        f"2. {au_summary(evidence)}\n\n"
        "Conclusion: "
        f"The expression label is reported as {pred} with {support_text} retrieval evidence. "
        "No additional AU identifiers, clinical conditions, demographics, or non-facial attributes are inferred."
    )


def has_au_hallucination(item: dict[str, Any]) -> bool:
    report = str(item.get("report") or "")
    evidence = str(item.get("retrieved_evidence") or "")
    if extract_malformed_au_mentions(report):
        return True
    report_aus = extract_aus(report)
    evidence_aus = extract_aus(evidence)
    return bool(report_aus and not report_aus.issubset(evidence_aus))


def sanitize_file(path: Path, dry_run: bool) -> int:
    with path.open("r", encoding="utf-8") as f:
        reports = json.load(f)
    if not isinstance(reports, list):
        raise ValueError(f"Expected a list in {path}")

    changed = 0
    for item in reports:
        if isinstance(item, dict) and has_au_hallucination(item):
            item["report"] = safe_report(item)
            changed += 1

    if changed and not dry_run:
        with path.open("w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
    return changed


def main() -> None:
    args = parse_args()
    total_changed = 0
    files = report_files(args)
    for path in files:
        changed = sanitize_file(path, args.dry_run)
        total_changed += changed
        if changed:
            print(f"{path}: replaced {changed} report(s)")
    print(f"Total replaced reports: {total_changed}")


if __name__ == "__main__":
    main()
