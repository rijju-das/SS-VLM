#!/usr/bin/env python3
"""Evaluate hallucination/faithfulness metrics for SS-VLM RAG reports.

This script is intentionally post-hoc: it reads the saved RAG reports and
prediction CSVs, then writes separate hallucination metric artifacts.

Implemented metrics:
1. Evidence Faithfulness Rate
2. Unsupported Clinical Claim Rate
3. AU Consistency Score
4. Prediction-Evidence Agreement
5. Contradiction Rate
6. Optional LLM-as-Judge Faithfulness
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


EMOTION_PATTERNS = {
    "Surprise": [r"\bsurprise\b", r"\bsurprised\b", r"\bsurprising\b"],
    "Fear": [r"\bfear\b", r"\bfearful\b"],
    "Disgust": [r"\bdisgust\b", r"\bdisgusted\b"],
    "Happiness": [r"\bhappiness\b", r"\bhappy\b"],
    "Sadness": [r"\bsadness\b", r"\bsad\b"],
    "Anger": [r"\banger\b", r"\bangry\b"],
    "Neutral": [r"\bneutral\b"],
}

CLINICAL_TERMS = [
    "adhd",
    "anxiety",
    "autism",
    "bipolar",
    "clinical disorder",
    "clinical symptom",
    "cognitive impairment",
    "depression",
    "diagnosis",
    "diagnostic",
    "disorder",
    "mental health",
    "patient",
    "psychosis",
    "ptsd",
    "schizophrenia",
    "suicidal",
    "therapy",
    "trauma",
    "treatment",
]

CLAIM_KEYWORDS = [
    "au",
    "classifier",
    "confidence",
    "evidence",
    "expression",
    "fused",
    "intensit",
    "label",
    "openface",
    "prediction",
    "prototype",
    "retrieval",
]

VALID_AU_NUMBERS = {
    1,
    2,
    4,
    5,
    6,
    7,
    9,
    10,
    12,
    14,
    15,
    17,
    20,
    23,
    25,
    26,
    28,
    45,
}

AU_MENTION_RE = re.compile(r"\bAU\s*[_-]?\s*0?(\d{1,3})(?:_[rc])?\b", flags=re.I)

NEGATION_TERMS = [
    "absence",
    "do not exceed",
    "does not exceed",
    "lack",
    "no au",
    "no significant",
    "not above",
    "without",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute faithfulness/hallucination metrics for SS-VLM RAG reports."
    )
    parser.add_argument("--reports-json", required=True, help="Path to *_rag_sample_reports.json")
    parser.add_argument("--predictions-csv", default="", help="Optional *_rag_predictions.csv")
    parser.add_argument("--output-json", required=True, help="Summary output JSON")
    parser.add_argument("--output-csv", required=True, help="Per-report output CSV")
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Run optional LLM-as-judge faithfulness scoring.",
    )
    parser.add_argument(
        "--judge-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Local/transformers model name for --llm-judge.",
    )
    parser.add_argument(
        "--judge-max-reports",
        type=int,
        default=0,
        help="Limit judged reports for smoke tests; 0 means all reports.",
    )
    return parser.parse_args()


def load_reports(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return [item for item in data if isinstance(item, dict)]


def load_predictions(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {str(row.get("sample_index", "")): row for row in reader}


def clean_markdown(text: str) -> str:
    text = re.sub(r"\*\*", "", text or "")
    text = re.sub(r"#+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_claims(report: str) -> list[str]:
    text = clean_markdown(report)
    text = re.sub(r"\b(Observation|Evidence|Conclusion)\s*:", ". ", text, flags=re.I)
    raw_parts = re.split(r"(?<=[.!?])\s+|\s+\d+\.\s+", text)
    claims = []
    for part in raw_parts:
        claim = part.strip(" -:\t\r\n")
        if not claim:
            continue
        lower = claim.lower()
        has_emotion = bool(extract_emotions(claim))
        has_au = bool(extract_aus(claim))
        has_keyword = any(keyword in lower for keyword in CLAIM_KEYWORDS)
        has_clinical = any(term in lower for term in CLINICAL_TERMS)
        if has_emotion or has_au or has_keyword or has_clinical:
            claims.append(claim)
    return claims


def extract_aus(text: str) -> set[str]:
    aus = set()
    for match in AU_MENTION_RE.finditer(text or ""):
        number = int(match.group(1))
        if number in VALID_AU_NUMBERS:
            aus.add(f"AU{number:02d}")
    return aus


def extract_malformed_au_mentions(text: str) -> list[str]:
    malformed = []
    for match in re.finditer(r"\bAUs?\s+(\d+)\s+to\s+(\d+)", text or "", flags=re.I):
        start = int(match.group(1))
        end = int(match.group(2))
        if start not in VALID_AU_NUMBERS or end not in VALID_AU_NUMBERS:
            malformed.append(f"AU{start}-AU{end}")
    for match in re.finditer(r"\bAUs?\s+([0-9][0-9,\s]*(?:and\s+\d+)?)", text or "", flags=re.I):
        for number_text in re.findall(r"\d+", match.group(1)):
            number = int(number_text)
            if number not in VALID_AU_NUMBERS:
                malformed.append(f"AU{number_text}")
    for match in AU_MENTION_RE.finditer(text or ""):
        number = int(match.group(1))
        if number not in VALID_AU_NUMBERS:
            malformed.append(f"AU{match.group(1)}")
    return sorted(set(malformed))


def extract_emotions(text: str) -> set[str]:
    found = set()
    lower = (text or "").lower()
    for emotion, patterns in EMOTION_PATTERNS.items():
        if any(re.search(pattern, lower) for pattern in patterns):
            found.add(emotion)
    return found


def extract_reported_final_emotions(report: str) -> set[str]:
    text = clean_markdown(report)
    final_patterns = [
        r"final fused expression(?: reported by the classifier)? is ([A-Za-z]+)",
        r"final fused expression for this dataset is ([A-Za-z]+)",
        r"expression label is ([A-Za-z]+)",
        r"expression is ([A-Za-z]+)",
        r"prediction is ([A-Za-z]+)",
        r"predicted as ([A-Za-z]+)",
    ]
    found = set()
    for pattern in final_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            found.update(extract_emotions(match.group(1)))
    return found


def evidence_context(item: dict[str, Any], pred_row: dict[str, str] | None) -> str:
    parts = [
        str(item.get("retrieved_evidence", "")),
        f"Final fused expression: {item.get('pred', '')}",
        f"Classifier expression: {item.get('classifier_pred', '')}",
        f"Retrieval-majority expression: {item.get('retrieval_pred', '')}",
    ]
    if pred_row:
        parts.extend(
            [
                f"Predicted expression: {pred_row.get('pred_emotion', '')}",
                f"Classifier expression: {pred_row.get('classifier_pred_emotion', '')}",
                f"Retrieval expression: {pred_row.get('retrieval_pred_emotion', '')}",
                f"Top-k emotions: {pred_row.get('top_k_emotions', '')}",
            ]
        )
    return "\n".join(parts)


def has_no_au_above_threshold(text: str) -> bool:
    lower = (text or "").lower()
    return bool(re.search(r"no\s+au\s+mean\s+above\s+\d", lower)) or "no au means above" in lower


def has_negated_au_claim(text: str) -> bool:
    lower = (text or "").lower()
    return "au" in lower and any(term in lower for term in NEGATION_TERMS)


def claim_is_supported(claim: str, evidence: str) -> bool:
    if extract_malformed_au_mentions(claim):
        return False

    claim_emotions = extract_emotions(claim)
    evidence_emotions = extract_emotions(evidence)
    claim_aus = extract_aus(claim)
    evidence_aus = extract_aus(evidence)

    if claim_aus and not claim_aus.issubset(evidence_aus):
        return False

    if has_negated_au_claim(claim) and has_no_au_above_threshold(evidence):
        au_supported = True
    else:
        au_supported = not ("au" in claim.lower() and not claim_aus and not evidence_aus)

    if not au_supported:
        return False

    if claim_emotions and not claim_emotions.issubset(evidence_emotions):
        return False

    clinical_terms = clinical_terms_in_text(claim)
    if clinical_terms and not clinical_terms.issubset(clinical_terms_in_text(evidence)):
        return False

    if claim_emotions or claim_aus or clinical_terms:
        return True

    lower = claim.lower()
    if "no au" in lower or "absence of any au" in lower:
        return has_no_au_above_threshold(evidence)
    if "limited" in lower or "mixed" in lower or "conflict" in lower:
        return bool((evidence or "").strip())
    return any(keyword in evidence.lower() for keyword in re.findall(r"[A-Za-z]{5,}", lower))


def clinical_terms_in_text(text: str) -> set[str]:
    lower = (text or "").lower()
    return {term for term in CLINICAL_TERMS if term in lower}


def unsupported_clinical_claims(claims: list[str], evidence: str) -> list[str]:
    evidence_terms = clinical_terms_in_text(evidence)
    unsupported = []
    for claim in claims:
        claim_terms = clinical_terms_in_text(claim)
        if claim_terms and not claim_terms.issubset(evidence_terms):
            unsupported.append(claim)
    return unsupported


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_item(item: dict[str, Any], pred_row: dict[str, str] | None) -> dict[str, Any]:
    evidence = evidence_context(item, pred_row)
    report = str(item.get("report", ""))
    claims = split_claims(report)
    supported_claims = [claim for claim in claims if claim_is_supported(claim, evidence)]
    malformed_au_mentions = extract_malformed_au_mentions(report)

    report_aus = extract_aus(report)
    evidence_aus = extract_aus(evidence)
    if report_aus:
        supported_aus = report_aus.intersection(evidence_aus)
        au_consistency = len(supported_aus) / len(report_aus)
    else:
        supported_aus = set()
        au_consistency = None

    clinical_unsupported = unsupported_clinical_claims(claims, evidence)
    reported_final = extract_reported_final_emotions(report)
    pred = str(item.get("pred") or (pred_row or {}).get("pred_emotion") or "")
    contradiction = bool(reported_final and pred and pred not in reported_final)

    consistency = parse_float(item.get("consistency"))
    if consistency is None and pred_row:
        consistency = parse_float(pred_row.get("retrieval_consistency"))

    retrieval_pred = str(item.get("retrieval_pred") or (pred_row or {}).get("retrieval_pred_emotion") or "")
    pred_retrieval_match = bool(pred and retrieval_pred and pred == retrieval_pred)

    factual_claim_count = len(claims)
    supported_claim_count = len(supported_claims)
    faithfulness = (
        supported_claim_count / factual_claim_count if factual_claim_count > 0 else None
    )

    return {
        "sample_index": str(item.get("sample_index", "")),
        "image_path": str(item.get("image_path", "")),
        "gt": str(item.get("gt") or (pred_row or {}).get("gt_emotion") or ""),
        "pred": pred,
        "classifier_pred": str(
            item.get("classifier_pred") or (pred_row or {}).get("classifier_pred_emotion") or ""
        ),
        "retrieval_pred": retrieval_pred,
        "retrieval_consistency": consistency,
        "pred_retrieval_match": pred_retrieval_match,
        "factual_claims": factual_claim_count,
        "supported_claims": supported_claim_count,
        "evidence_faithfulness": faithfulness,
        "report_au_mentions": len(report_aus),
        "supported_au_mentions": len(supported_aus),
        "au_consistency": au_consistency,
        "malformed_au_mentions": ";".join(malformed_au_mentions),
        "malformed_au_count": len(malformed_au_mentions),
        "has_malformed_au_mention": bool(malformed_au_mentions),
        "unsupported_clinical_claims": len(clinical_unsupported),
        "has_unsupported_clinical_claim": bool(clinical_unsupported),
        "contradiction": contradiction,
        "reported_final_emotions": ";".join(sorted(reported_final)),
        "unsupported_clinical_claim_text": " | ".join(clinical_unsupported),
    }


def run_llm_judge(
    rows: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    predictions: dict[str, dict[str, str]],
    model_name: str,
    max_reports: int,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    limit = len(rows) if max_reports <= 0 else min(max_reports, len(rows))
    report_by_index = {str(item.get("sample_index", "")): item for item in reports}
    for row in rows[:limit]:
        item = report_by_index.get(str(row["sample_index"]), {})
        pred_row = predictions.get(str(row["sample_index"]), {})
        prompt = build_judge_prompt(item, pred_row)
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": "You are a strict evaluator of evidence faithfulness.",
                },
                {"role": "user", "content": prompt},
            ]
            model_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            model_prompt = f"System: You are a strict evaluator of evidence faithfulness.\nUser: {prompt}\nAssistant:"

        inputs = tokenizer([model_prompt], return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()
        parsed = parse_judge_response(response)
        row["llm_judge_label"] = parsed.get("label", "parse_error")
        row["llm_judge_score"] = parsed.get("score")
        row["llm_judge_reason"] = parsed.get("reason", response[:300])

    for row in rows[limit:]:
        row["llm_judge_label"] = ""
        row["llm_judge_score"] = ""
        row["llm_judge_reason"] = ""


def build_judge_prompt(item: dict[str, Any], pred_row: dict[str, str]) -> str:
    return f"""Evaluate whether the report is faithful to the evidence.

Prediction metadata:
- Final prediction: {item.get("pred") or pred_row.get("pred_emotion", "")}
- Classifier prediction: {item.get("classifier_pred") or pred_row.get("classifier_pred_emotion", "")}
- Retrieval prediction: {item.get("retrieval_pred") or pred_row.get("retrieval_pred_emotion", "")}

Retrieved evidence:
{item.get("retrieved_evidence", "")}

Report:
{item.get("report", "")}

Return only JSON with this schema:
{{"label":"supported|partial|unsupported","score":0.0,"reason":"short reason"}}
"""


def parse_judge_response(response: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", response, flags=re.S)
    if not match:
        return {"label": "parse_error", "score": "", "reason": response[:300]}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"label": "parse_error", "score": "", "reason": response[:300]}
    label = str(data.get("label", "parse_error")).lower()
    if label not in {"supported", "partial", "unsupported"}:
        label = "parse_error"
    score = parse_float(data.get("score"))
    return {
        "label": label,
        "score": "" if score is None else max(0.0, min(1.0, score)),
        "reason": str(data.get("reason", ""))[:500],
    }


def summarize(rows: list[dict[str, Any]], reports_json: Path, predictions_csv: Path | None) -> dict[str, Any]:
    faithfulness_values = [
        row["evidence_faithfulness"]
        for row in rows
        if row["evidence_faithfulness"] is not None
    ]
    au_values = [row["au_consistency"] for row in rows if row["au_consistency"] is not None]
    consistency_values = [
        row["retrieval_consistency"]
        for row in rows
        if row["retrieval_consistency"] is not None
    ]
    judge_scores = [
        parse_float(row.get("llm_judge_score"))
        for row in rows
        if parse_float(row.get("llm_judge_score")) is not None
    ]
    judge_labels = Counter(
        row.get("llm_judge_label", "")
        for row in rows
        if row.get("llm_judge_label")
    )

    total = len(rows)
    return {
        "reports_json": str(reports_json),
        "predictions_csv": str(predictions_csv) if predictions_csv else "",
        "total_reports": total,
        "total_factual_claims": sum(int(row["factual_claims"]) for row in rows),
        "total_supported_claims": sum(int(row["supported_claims"]) for row in rows),
        "evidence_faithfulness_rate": mean(faithfulness_values) if faithfulness_values else None,
        "unsupported_clinical_claim_rate": (
            sum(1 for row in rows if row["has_unsupported_clinical_claim"]) / total
            if total
            else None
        ),
        "unsupported_clinical_claim_count": sum(
            int(row["unsupported_clinical_claims"]) for row in rows
        ),
        "au_consistency_score": mean(au_values) if au_values else None,
        "reports_with_au_mentions": len(au_values),
        "malformed_au_mention_rate": (
            sum(1 for row in rows if row["has_malformed_au_mention"]) / total
            if total
            else None
        ),
        "malformed_au_mention_count": sum(1 for row in rows if row["has_malformed_au_mention"]),
        "malformed_au_report_count": sum(1 for row in rows if row["has_malformed_au_mention"]),
        "malformed_au_mention_total": sum(int(row["malformed_au_count"]) for row in rows),
        "prediction_evidence_agreement_mean": (
            mean(consistency_values) if consistency_values else None
        ),
        "pred_retrieval_match_rate": (
            sum(1 for row in rows if row["pred_retrieval_match"]) / total if total else None
        ),
        "contradiction_rate": (
            sum(1 for row in rows if row["contradiction"]) / total if total else None
        ),
        "contradiction_count": sum(1 for row in rows if row["contradiction"]),
        "llm_judge_label_counts": dict(judge_labels),
        "llm_judge_faithfulness_mean": mean(judge_scores) if judge_scores else None,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "sample_index",
        "image_path",
        "gt",
        "pred",
        "classifier_pred",
        "retrieval_pred",
        "retrieval_consistency",
        "pred_retrieval_match",
        "factual_claims",
        "supported_claims",
        "evidence_faithfulness",
        "report_au_mentions",
        "supported_au_mentions",
        "au_consistency",
        "malformed_au_mentions",
        "malformed_au_count",
        "has_malformed_au_mention",
        "unsupported_clinical_claims",
        "has_unsupported_clinical_claim",
        "contradiction",
        "reported_final_emotions",
        "unsupported_clinical_claim_text",
        "llm_judge_label",
        "llm_judge_score",
        "llm_judge_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def load_existing_judge_rows(path: Path) -> dict[str, dict[str, str]]:
    """Preserve previously computed LLM-judge labels when only rule metrics change."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {
            str(row.get("sample_index", "")): {
                "llm_judge_label": row.get("llm_judge_label", ""),
                "llm_judge_score": row.get("llm_judge_score", ""),
                "llm_judge_reason": row.get("llm_judge_reason", ""),
            }
            for row in reader
        }


def main() -> None:
    args = parse_args()
    reports_json = Path(args.reports_json).expanduser().resolve()
    predictions_csv = Path(args.predictions_csv).expanduser().resolve() if args.predictions_csv else None
    output_json = Path(args.output_json).expanduser()
    output_csv = Path(args.output_csv).expanduser()

    reports = load_reports(reports_json)
    predictions = load_predictions(predictions_csv)
    rows = [
        evaluate_item(item, predictions.get(str(item.get("sample_index", ""))))
        for item in reports
    ]

    for row in rows:
        row.setdefault("llm_judge_label", "")
        row.setdefault("llm_judge_score", "")
        row.setdefault("llm_judge_reason", "")

    if args.llm_judge:
        run_llm_judge(
            rows=rows,
            reports=reports,
            predictions=predictions,
            model_name=args.judge_model,
            max_reports=args.judge_max_reports,
        )
    else:
        existing_judge_rows = load_existing_judge_rows(output_csv)
        for row in rows:
            preserved = existing_judge_rows.get(str(row["sample_index"]))
            if preserved:
                row.update(preserved)

    summary = summarize(rows, reports_json, predictions_csv)
    write_json(summary, output_json)
    write_csv(rows, output_csv)

    print(f"Reports evaluated: {summary['total_reports']}")
    print(f"Evidence Faithfulness Rate: {summary['evidence_faithfulness_rate']}")
    print(f"Unsupported Clinical Claim Rate: {summary['unsupported_clinical_claim_rate']}")
    print(f"AU Consistency Score: {summary['au_consistency_score']}")
    print(f"Prediction-Evidence Agreement: {summary['prediction_evidence_agreement_mean']}")
    print(f"Contradiction Rate: {summary['contradiction_rate']}")
    print(f"Summary JSON: {output_json}")
    print(f"Per-report CSV: {output_csv}")


if __name__ == "__main__":
    main()
