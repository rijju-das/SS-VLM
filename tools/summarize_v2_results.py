#!/usr/bin/env python3
"""Summarize SS-VLM v2 training and RAG result JSON files."""

import argparse
import json
from pathlib import Path


def pct(value):
    if value is None or value == "":
        return ""
    return f"{100 * float(value):.2f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize SS-VLM v2 results")
    parser.add_argument("--metrics-root", default="outputs_v2/metrics")
    args = parser.parse_args()

    root = Path(args.metrics_root)
    if not root.exists():
        raise SystemExit(f"Metrics root not found: {root}")

    train_rows = []
    rag_rows = []
    for path in sorted(root.glob("**/*_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "best_val_acc" in data:
            train_rows.append(
                {
                    "run": path.name.replace("_train_summary.json", ""),
                    "variant": data.get("model_variant", ""),
                    "seed": data.get("seed", ""),
                    "best_val_acc": pct(data.get("best_val_acc")),
                    "best_epoch": data.get("best_epoch", ""),
                    "epochs_completed": data.get("epochs_completed", ""),
                    "lambda": data.get("lambda_cont", ""),
                    "backbone_lr": data.get("backbone_lr", ""),
                    "head_lr": data.get("head_lr", ""),
                    "path": str(path),
                }
            )
        elif "accuracy" in data:
            rag_rows.append(
                {
                    "run": path.name.replace("_rag_summary.json", ""),
                    "variant": data.get("model_variant", ""),
                    "classifier": pct(data.get("classifier_accuracy")),
                    "retrieval": pct(data.get("retrieval_accuracy")),
                    "rag_fused": pct(data.get("accuracy")),
                    "au_acc": pct(data.get("au_accuracy_on_available")),
                    "au_available": data.get("au_available", ""),
                    "k": data.get("top_k_retrieval", ""),
                    "alpha": data.get("fusion_alpha", ""),
                    "au_beta": data.get("au_fusion_beta", ""),
                    "path": str(path),
                }
            )

    print("\nTRAINING SUMMARIES")
    if not train_rows:
        print("No training summaries found.")
    else:
        headers = ["run", "variant", "seed", "best_val_acc", "best_epoch", "epochs_completed", "lambda", "backbone_lr", "head_lr"]
        print("\t".join(headers))
        for row in train_rows:
            print("\t".join(str(row[h]) for h in headers))

    print("\nRAG SUMMARIES")
    if not rag_rows:
        print("No RAG summaries found.")
    else:
        headers = ["run", "variant", "classifier", "retrieval", "rag_fused", "au_acc", "au_available", "k", "alpha", "au_beta"]
        print("\t".join(headers))
        for row in rag_rows:
            print("\t".join(str(row[h]) for h in headers))


if __name__ == "__main__":
    main()
