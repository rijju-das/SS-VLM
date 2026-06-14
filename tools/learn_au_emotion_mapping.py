#!/usr/bin/env python3
"""Learn an interpretable AU-to-emotion support matrix from RAF-DB.

The goal is to replace the hard-coded FACS-style AU mapping with a data-driven,
auditable mapping learned from OpenFace AU values and RAF-DB train labels.

Method:
  1. Use only RAF-DB train rows for fitting and model selection.
  2. Standardize AU features so coefficients are comparable.
  3. Fit sparse multinomial logistic regression across several seeds.
  4. Extract class-wise positive coefficients as learned AU support.
  5. Summarize stability of selected AUs across seeds.
  6. Optionally confirm via class-wise permutation importance.

Output files include coefficient tables, stable AU-support summaries, test
metrics, and optional permutation-importance confirmation tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from train_au_emotion_calibrator import EMOTIONS, feature_columns, read_au_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn interpretable AU-emotion support from RAF-DB.")
    parser.add_argument("--au-csv", default="outputs/rafdb_openface2_aus.csv")
    parser.add_argument("--output-dir", default="outputs_v2/au_mapping")
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--feature-set",
        choices=["intensity", "presence", "both"],
        default="intensity",
        help="Use AU intensity, AU presence, or both. Intensity is recommended for readable AU support.",
    )
    parser.add_argument("--seeds", default="42,123,2026,7,99")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument(
        "--penalty",
        choices=["l1", "elasticnet"],
        default="elasticnet",
        help="Sparse penalty for interpretable coefficients.",
    )
    parser.add_argument("--l1-ratio", type=float, default=0.7, help="Only used for elastic-net logistic regression.")
    parser.add_argument("--c-values", default="0.03,0.05,0.10,0.20,0.50,1.00,2.00")
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "accuracy", "weighted_f1", "nll"],
        default="macro_f1",
        help="Metric for selecting C on the internal train/validation split.",
    )
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument(
        "--permutation-subset",
        choices=["test", "val"],
        default="test",
        help="Subset used for class-wise permutation confirmation.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one seed")
    return items


def parse_float_list(value: str) -> list[float]:
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one C value")
    return items


def display_au(feature_name: str) -> str:
    if feature_name.endswith("_r"):
        return feature_name[:-2]
    if feature_name.endswith("_c"):
        return f"{feature_name[:-2]}_presence"
    return feature_name


def split_train_test(au_csv: Path, cols: list[str]) -> dict:
    x_all, y_all, metas, splits = read_au_csv(au_csv, cols)
    train_mask = splits == "train"
    test_mask = splits == "test"
    return {
        "x_train_full": x_all[train_mask],
        "y_train_full": y_all[train_mask],
        "meta_train_full": [meta for meta, keep in zip(metas, train_mask) if keep],
        "x_test": x_all[test_mask],
        "y_test": y_all[test_mask],
        "meta_test": [meta for meta, keep in zip(metas, test_mask) if keep],
    }


def build_model(args: argparse.Namespace, c_value: float, seed: int) -> LogisticRegression:
    kwargs = {
        "C": c_value,
        "penalty": args.penalty,
        "solver": "saga",
        "max_iter": args.max_iter,
        "multi_class": "multinomial",
        "class_weight": None if args.class_weight == "none" else "balanced",
        "random_state": seed,
        "n_jobs": -1,
    }
    if args.penalty == "elasticnet":
        kwargs["l1_ratio"] = args.l1_ratio
    return LogisticRegression(**kwargs)


def predict_proba_from_model(model: LogisticRegression, scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(scaler.transform(x))


def metrics_from_probs(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "nll": float(log_loss(labels, probs, labels=list(range(len(EMOTIONS))))),
    }


def selection_score(metrics: dict, metric: str) -> float:
    value = float(metrics[metric])
    return value if metric != "nll" else -value


def fit_scaled_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    c_value: float,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    model = build_model(args, c_value=c_value, seed=seed)
    model.fit(x_scaled, y_train)
    return scaler, model


def choose_c_for_seed(data: dict, args: argparse.Namespace, seed: int, c_values: list[float]) -> dict:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(data["x_train_full"], data["y_train_full"]))
    x_inner_train = data["x_train_full"][train_idx]
    y_inner_train = data["y_train_full"][train_idx]
    x_val = data["x_train_full"][val_idx]
    y_val = data["y_train_full"][val_idx]

    candidates = []
    for c_value in c_values:
        scaler, model = fit_scaled_logreg(x_inner_train, y_inner_train, args, c_value, seed)
        val_probs = predict_proba_from_model(model, scaler, x_val)
        val_metrics = metrics_from_probs(y_val, val_probs)
        candidates.append(
            {
                "c": c_value,
                "metrics": val_metrics,
                "score": selection_score(val_metrics, args.selection_metric),
            }
        )
    best = max(candidates, key=lambda item: item["score"])
    return {
        "seed": seed,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "x_val": x_val,
        "y_val": y_val,
        "candidates": candidates,
        "best_c": float(best["c"]),
        "best_val_metrics": best["metrics"],
    }


def coefficient_rows(seed: int, selected_c: float, cols: list[str], coef: np.ndarray) -> list[dict]:
    rows = []
    for class_idx, emotion in enumerate(EMOTIONS):
        for feature_idx, feature_name in enumerate(cols):
            value = float(coef[class_idx, feature_idx])
            rows.append(
                {
                    "seed": seed,
                    "selected_c": selected_c,
                    "emotion": emotion,
                    "class_idx": class_idx,
                    "feature": feature_name,
                    "au": display_au(feature_name),
                    "coefficient": value,
                    "positive": int(value > 0),
                }
            )
    return rows


def top_rows_from_coefficients(
    seed: int,
    selected_c: float,
    cols: list[str],
    coef: np.ndarray,
    top_k: int,
    positive: bool,
) -> list[dict]:
    rows = []
    for class_idx, emotion in enumerate(EMOTIONS):
        class_coef = coef[class_idx]
        if positive:
            order = np.argsort(class_coef)[::-1]
            keep = [idx for idx in order if class_coef[idx] > 0][:top_k]
            direction = "positive"
        else:
            order = np.argsort(class_coef)
            keep = [idx for idx in order if class_coef[idx] < 0][:top_k]
            direction = "negative"
        for rank, feature_idx in enumerate(keep, start=1):
            rows.append(
                {
                    "seed": seed,
                    "selected_c": selected_c,
                    "emotion": emotion,
                    "class_idx": class_idx,
                    "rank": rank,
                    "feature": cols[feature_idx],
                    "au": display_au(cols[feature_idx]),
                    "coefficient": float(class_coef[feature_idx]),
                    "direction": direction,
                }
            )
    return rows


def classwise_permutation_importance(
    model: LogisticRegression,
    scaler: StandardScaler,
    x: np.ndarray,
    y: np.ndarray,
    cols: list[str],
    seed: int,
    repeats: int,
) -> list[dict]:
    if repeats <= 0:
        return []

    rng = np.random.default_rng(seed)
    x_scaled = scaler.transform(x)
    baseline_preds = model.predict(x_scaled)
    baseline_recall = recall_score(
        y,
        baseline_preds,
        labels=list(range(len(EMOTIONS))),
        average=None,
        zero_division=0,
    )

    rows = []
    for feature_idx, feature_name in enumerate(cols):
        drops = []
        for _ in range(repeats):
            x_perm = x_scaled.copy()
            x_perm[:, feature_idx] = rng.permutation(x_perm[:, feature_idx])
            perm_preds = model.predict(x_perm)
            perm_recall = recall_score(
                y,
                perm_preds,
                labels=list(range(len(EMOTIONS))),
                average=None,
                zero_division=0,
            )
            drops.append(baseline_recall - perm_recall)
        drops_np = np.asarray(drops, dtype=float)
        for class_idx, emotion in enumerate(EMOTIONS):
            rows.append(
                {
                    "seed": seed,
                    "emotion": emotion,
                    "class_idx": class_idx,
                    "feature": feature_name,
                    "au": display_au(feature_name),
                    "mean_recall_drop": float(np.mean(drops_np[:, class_idx])),
                    "std_recall_drop": float(np.std(drops_np[:, class_idx])),
                    "baseline_recall": float(baseline_recall[class_idx]),
                    "repeats": repeats,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_stability(
    coefficient_rows_all: list[dict],
    top_positive_rows: list[dict],
    top_negative_rows: list[dict],
    seeds: list[int],
    cols: list[str],
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    coef_lookup: dict[tuple[str, str], list[float]] = {}
    positive_lookup: dict[tuple[str, str], int] = {}
    selected_positive_lookup: dict[tuple[str, str], list[int]] = {}
    selected_negative_lookup: dict[tuple[str, str], list[int]] = {}

    for row in coefficient_rows_all:
        key = (row["emotion"], row["feature"])
        coef_lookup.setdefault(key, []).append(float(row["coefficient"]))
        positive_lookup[key] = positive_lookup.get(key, 0) + int(float(row["coefficient"]) > 0)

    for row in top_positive_rows:
        key = (row["emotion"], row["feature"])
        selected_positive_lookup.setdefault(key, []).append(int(row["rank"]))

    for row in top_negative_rows:
        key = (row["emotion"], row["feature"])
        selected_negative_lookup.setdefault(key, []).append(int(row["rank"]))

    stability_rows = []
    n_seeds = len(seeds)
    for emotion in EMOTIONS:
        for feature_name in cols:
            key = (emotion, feature_name)
            coefs = np.asarray(coef_lookup.get(key, []), dtype=float)
            pos_ranks = selected_positive_lookup.get(key, [])
            neg_ranks = selected_negative_lookup.get(key, [])
            mean_coef = float(np.mean(coefs)) if len(coefs) else 0.0
            stability_rows.append(
                {
                    "emotion": emotion,
                    "feature": feature_name,
                    "au": display_au(feature_name),
                    "mean_coefficient": mean_coef,
                    "std_coefficient": float(np.std(coefs)) if len(coefs) else 0.0,
                    "positive_frequency": positive_lookup.get(key, 0) / n_seeds,
                    "top_positive_frequency": len(pos_ranks) / n_seeds,
                    "mean_top_positive_rank": float(np.mean(pos_ranks)) if pos_ranks else "",
                    "top_negative_frequency": len(neg_ranks) / n_seeds,
                    "mean_top_negative_rank": float(np.mean(neg_ranks)) if neg_ranks else "",
                    "support_score": max(mean_coef, 0.0) * (len(pos_ranks) / n_seeds),
                    "absence_score": abs(min(mean_coef, 0.0)) * (len(neg_ranks) / n_seeds),
                }
            )

    summary_rows = []
    for emotion in EMOTIONS:
        emotion_rows = [row for row in stability_rows if row["emotion"] == emotion]
        positive = sorted(
            [row for row in emotion_rows if float(row["mean_coefficient"]) > 0],
            key=lambda row: (
                float(row["top_positive_frequency"]),
                float(row["support_score"]),
                float(row["mean_coefficient"]),
            ),
            reverse=True,
        )[:top_k]
        negative = sorted(
            [row for row in emotion_rows if float(row["mean_coefficient"]) < 0],
            key=lambda row: (
                float(row["top_negative_frequency"]),
                float(row["absence_score"]),
                abs(float(row["mean_coefficient"])),
            ),
            reverse=True,
        )[:top_k]
        summary_rows.append(
            {
                "emotion": emotion,
                "top_positive_aus": "; ".join(
                    f"{row['au']} ({float(row['mean_coefficient']):+.3f}, {float(row['top_positive_frequency']):.0%})"
                    for row in positive
                ),
                "top_negative_or_absence_cues": "; ".join(
                    f"{row['au']} ({float(row['mean_coefficient']):+.3f}, {float(row['top_negative_frequency']):.0%})"
                    for row in negative
                ),
            }
        )

    return stability_rows, summary_rows


def summarize_permutation(rows: list[dict], top_k: int) -> list[dict]:
    if not rows:
        return []
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row["emotion"], row["feature"])
        grouped.setdefault(key, []).append(float(row["mean_recall_drop"]))

    summary_rows = []
    for (emotion, feature_name), values in grouped.items():
        values_np = np.asarray(values, dtype=float)
        summary_rows.append(
            {
                "emotion": emotion,
                "feature": feature_name,
                "au": display_au(feature_name),
                "mean_recall_drop": float(np.mean(values_np)),
                "std_recall_drop": float(np.std(values_np)),
                "positive_drop_frequency": float(np.mean(values_np > 0)),
            }
        )
    ranked = []
    for emotion in EMOTIONS:
        ranked.extend(
            sorted(
                [row for row in summary_rows if row["emotion"] == emotion],
                key=lambda row: (float(row["mean_recall_drop"]), float(row["positive_drop_frequency"])),
                reverse=True,
            )[:top_k]
        )
    return ranked


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def write_latex_summary(path: Path, summary_rows: list[dict]) -> None:
    lines = [
        r"\begin{tabular}{p{0.15\linewidth}p{0.38\linewidth}p{0.38\linewidth}}",
        r"\toprule",
        r"Emotion & Learned positive AU support & Learned negative / absence cues \\",
        r"\midrule",
    ]
    for row in summary_rows:
        lines.append(
            f"{latex_escape(row['emotion'])} & "
            f"{latex_escape(row['top_positive_aus'])} & "
            f"{latex_escape(row['top_negative_or_absence_cues'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")

    repo_root = Path(__file__).resolve().parents[1]
    au_csv = Path(args.au_csv).expanduser()
    if not au_csv.is_absolute():
        au_csv = repo_root / au_csv
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    run_name = args.run_name or f"learned_au_mapping_{args.feature_set}_{args.penalty}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)

    seeds = parse_int_list(args.seeds)
    c_values = parse_float_list(args.c_values)
    cols = feature_columns(args.feature_set)
    data = split_train_test(au_csv, cols)

    print(f"Learning AU-emotion mapping: {run_name}")
    print(f"  AU CSV: {au_csv}")
    print(f"  Features: {args.feature_set} ({len(cols)} columns)")
    print(f"  RAF-DB train/test: {len(data['y_train_full'])}/{len(data['y_test'])}")
    print(f"  Seeds: {seeds}")
    print(f"  C grid: {c_values}")

    seed_metrics_rows = []
    all_coefficients = []
    all_top_positive = []
    all_top_negative = []
    all_permutation = []
    c_selection = []

    for seed in seeds:
        print(f"Seed {seed}: selecting C...")
        selected = choose_c_for_seed(data, args, seed, c_values)
        selected_c = float(selected["best_c"])
        c_selection.extend(
            {
                "seed": seed,
                "c": item["c"],
                "selection_metric": args.selection_metric,
                "selection_score": item["score"],
                **{f"val_{key}": value for key, value in item["metrics"].items()},
                "selected": int(item["c"] == selected_c),
            }
            for item in selected["candidates"]
        )

        scaler, model = fit_scaled_logreg(
            data["x_train_full"],
            data["y_train_full"],
            args,
            c_value=selected_c,
            seed=seed,
        )
        val_probs = predict_proba_from_model(model, scaler, selected["x_val"])
        test_probs = predict_proba_from_model(model, scaler, data["x_test"])
        val_metrics = metrics_from_probs(selected["y_val"], val_probs)
        test_metrics = metrics_from_probs(data["y_test"], test_probs)

        seed_metrics_rows.append(
            {
                "seed": seed,
                "selected_c": selected_c,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
        print(
            f"  C={selected_c:.3g} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} "
            f"test_macro_f1={test_metrics['macro_f1']:.4f}"
        )

        coef = np.asarray(model.coef_, dtype=float)
        all_coefficients.extend(coefficient_rows(seed, selected_c, cols, coef))
        all_top_positive.extend(top_rows_from_coefficients(seed, selected_c, cols, coef, args.top_k, positive=True))
        all_top_negative.extend(top_rows_from_coefficients(seed, selected_c, cols, coef, args.top_k, positive=False))

        if args.permutation_repeats > 0:
            if args.permutation_subset == "val":
                perm_x = selected["x_val"]
                perm_y = selected["y_val"]
            else:
                perm_x = data["x_test"]
                perm_y = data["y_test"]
            all_permutation.extend(
                classwise_permutation_importance(
                    model,
                    scaler,
                    perm_x,
                    perm_y,
                    cols,
                    seed=seed,
                    repeats=args.permutation_repeats,
                )
            )

        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_set": args.feature_set,
                "feature_columns": cols,
                "emotions": EMOTIONS,
                "selected_c": selected_c,
                "seed": seed,
                "test_metrics": test_metrics,
            },
            run_dir / "models" / f"learned_au_mapping_seed{seed}.joblib",
        )

    stability_rows, summary_rows = summarize_stability(
        all_coefficients,
        all_top_positive,
        all_top_negative,
        seeds,
        cols,
        args.top_k,
    )
    permutation_summary_rows = summarize_permutation(all_permutation, args.top_k)

    write_csv(run_dir / "c_selection.csv", c_selection)
    write_csv(run_dir / "seed_metrics.csv", seed_metrics_rows)
    write_csv(run_dir / "seed_coefficients.csv", all_coefficients)
    write_csv(run_dir / "top_positive_aus_by_seed.csv", all_top_positive)
    write_csv(run_dir / "top_negative_aus_by_seed.csv", all_top_negative)
    write_csv(run_dir / "learned_au_mapping_stability.csv", stability_rows)
    write_csv(run_dir / "learned_au_mapping_summary.csv", summary_rows)
    write_latex_summary(run_dir / "learned_au_mapping_summary.tex", summary_rows)
    if all_permutation:
        write_csv(run_dir / "permutation_importance_by_seed.csv", all_permutation)
        write_csv(run_dir / "permutation_importance_summary.csv", permutation_summary_rows)

    summary_json = {
        "run_name": run_name,
        "au_csv": str(au_csv),
        "feature_set": args.feature_set,
        "feature_columns": cols,
        "seeds": seeds,
        "c_values": c_values,
        "penalty": args.penalty,
        "l1_ratio": args.l1_ratio if args.penalty == "elasticnet" else None,
        "class_weight": args.class_weight,
        "selection_metric": args.selection_metric,
        "top_k": args.top_k,
        "permutation_repeats": args.permutation_repeats,
        "permutation_subset": args.permutation_subset,
        "mean_test_accuracy": float(np.mean([row["test_accuracy"] for row in seed_metrics_rows])),
        "std_test_accuracy": float(np.std([row["test_accuracy"] for row in seed_metrics_rows])),
        "mean_test_macro_f1": float(np.mean([row["test_macro_f1"] for row in seed_metrics_rows])),
        "std_test_macro_f1": float(np.std([row["test_macro_f1"] for row in seed_metrics_rows])),
        "classification_report_last_seed_test": classification_report(
            data["y_test"],
            test_probs.argmax(axis=1),
            target_names=EMOTIONS,
            output_dict=True,
            zero_division=0,
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print("Done.")
    print(f"  Run directory: {run_dir}")
    print(f"  Learned mapping: {run_dir / 'learned_au_mapping_summary.csv'}")
    print(f"  Stability:       {run_dir / 'learned_au_mapping_stability.csv'}")
    if all_permutation:
        print(f"  Permutation:     {run_dir / 'permutation_importance_summary.csv'}")
    print(
        f"  Mean test acc:   {summary_json['mean_test_accuracy']:.4f} "
        f"+/- {summary_json['std_test_accuracy']:.4f}"
    )
    print(
        f"  Mean test mF1:   {summary_json['mean_test_macro_f1']:.4f} "
        f"+/- {summary_json['std_test_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
