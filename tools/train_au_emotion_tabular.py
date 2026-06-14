#!/usr/bin/env python3
"""Train sklearn tabular AU-to-emotion calibrators for RAF-DB.

This is an accuracy-oriented companion to train_au_emotion_calibrator.py. It
uses OpenFace AU values as tabular features and trains classical classifiers
such as RandomForest, HistGradientBoosting, logistic regression, or SVC. These
models are often stronger than a tiny MLP for 17-35 dimensional AU features.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_au_emotion_calibrator import EMOTIONS, expected_calibration_error
from train_au_emotion_calibrator import feature_columns, read_au_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sklearn AU-to-emotion calibrators.")
    parser.add_argument("--au-csv", default="outputs/rafdb_openface2_aus.csv")
    parser.add_argument("--output-dir", default="outputs_v2/au_calibrator")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--feature-set", choices=["intensity", "presence", "both"], default="both")
    parser.add_argument(
        "--model",
        choices=["auto", "random_forest", "hist_gbdt", "gbdt", "extra_trees", "logreg", "svc_rbf"],
        default="auto",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy", "macro_f1", "weighted_f1", "nll"],
        default="accuracy",
        help="Validation metric used when --model auto trains multiple candidates.",
    )
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument(
        "--no-refit-full-train",
        action="store_true",
        help="By default, refit the selected model on all RAF-DB train rows after validation selection.",
    )
    parser.add_argument("--temperature-grid", default="0.70,0.80,0.90,1.00,1.10,1.20,1.35,1.50,1.75,2.00,2.50,3.00")
    return parser.parse_args()


def split_dataset(au_csv: Path, cols: list[str], val_size: float, seed: int):
    x_all, y_all, metas, splits = read_au_csv(au_csv, cols)
    train_mask = splits == "train"
    test_mask = splits == "test"

    x_train_full = x_all[train_mask]
    y_train_full = y_all[train_mask]
    meta_train_full = [meta for meta, keep in zip(metas, train_mask) if keep]
    x_test = x_all[test_mask]
    y_test = y_all[test_mask]
    meta_test = [meta for meta, keep in zip(metas, test_mask) if keep]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(x_train_full, y_train_full))
    return {
        "x_train": x_train_full[train_idx],
        "y_train": y_train_full[train_idx],
        "meta_train": [meta_train_full[i] for i in train_idx],
        "x_val": x_train_full[val_idx],
        "y_val": y_train_full[val_idx],
        "meta_val": [meta_train_full[i] for i in val_idx],
        "x_test": x_test,
        "y_test": y_test,
        "meta_test": meta_test,
    }


def parse_temperature_grid(value: str) -> list[float]:
    temps = [float(item.strip()) for item in value.split(",") if item.strip()]
    temps = [temp for temp in temps if temp > 0]
    if not temps:
        raise ValueError("--temperature-grid must contain at least one positive value")
    return temps


def build_model(name: str, args: argparse.Namespace):
    class_weight = None if args.class_weight == "none" else "balanced"
    if name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            min_samples_leaf=2,
            class_weight=class_weight,
        )
        return make_pipeline(StandardScaler(), clf)
    if name == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            min_samples_leaf=2,
            class_weight=class_weight,
        )
        return make_pipeline(StandardScaler(), clf)
    if name == "hist_gbdt":
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            l2_regularization=0.01,
            random_state=args.seed,
        )
        return make_pipeline(StandardScaler(), clf)
    if name == "gbdt":
        clf = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=3,
            random_state=args.seed,
        )
        return make_pipeline(StandardScaler(), clf)
    if name == "logreg":
        clf = LogisticRegression(
            max_iter=3000,
            C=1.0,
            multi_class="multinomial",
            class_weight=class_weight,
        )
        return make_pipeline(StandardScaler(), clf)
    if name == "svc_rbf":
        clf = SVC(C=3.0, gamma="scale", probability=True, class_weight=class_weight, random_state=args.seed)
        return make_pipeline(StandardScaler(), clf)
    raise ValueError(f"Unknown model: {name}")


def scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    probs = np.clip(probs, 1e-12, 1.0)
    logits = np.log(probs)
    logits = logits / max(float(temperature), 1e-8)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def metrics_from_probs(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "nll": float(log_loss(labels, probs, labels=list(range(len(EMOTIONS))))),
        "ece": expected_calibration_error(labels, probs),
    }


def choose_temperature(labels: np.ndarray, probs: np.ndarray, grid: list[float]) -> tuple[float, dict[str, float]]:
    best_temp = 1.0
    best_nll = float("inf")
    scores = {}
    for temp in grid:
        scaled = scale_probs(probs, temp)
        nll = float(log_loss(labels, scaled, labels=list(range(len(EMOTIONS)))))
        scores[f"{temp:.4f}"] = nll
        if nll < best_nll:
            best_nll = nll
            best_temp = temp
    return best_temp, scores


def score_for_selection(metrics: dict, selection_metric: str) -> float:
    value = float(metrics[selection_metric])
    return -value if selection_metric != "nll" else value


def candidate_names(model_name: str) -> list[str]:
    if model_name != "auto":
        return [model_name]
    return ["random_forest", "hist_gbdt", "gbdt", "extra_trees", "logreg", "svc_rbf"]


def write_predictions(path: Path, subset: str, meta: list[dict], labels: np.ndarray, probs: np.ndarray) -> None:
    preds = probs.argmax(axis=1)
    fieldnames = [
        "subset",
        "row_index",
        "split",
        "image_path",
        "relative_path",
        "gt_label",
        "gt_emotion",
        "pred_label",
        "pred_emotion",
        "confidence",
        "correct",
        *[f"prob_{emotion}" for emotion in EMOTIONS],
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for item, label, pred, prob in zip(meta, labels, preds, probs):
            row = {
                "subset": subset,
                "row_index": item["row_index"],
                "split": item["split"],
                "image_path": item["image_path"],
                "relative_path": item["relative_path"],
                "gt_label": int(label),
                "gt_emotion": EMOTIONS[int(label)],
                "pred_label": int(pred),
                "pred_emotion": EMOTIONS[int(pred)],
                "confidence": f"{float(prob[pred]):.6f}",
                "correct": int(pred == label),
            }
            for emotion, value in zip(EMOTIONS, prob):
                row[f"prob_{emotion}"] = f"{float(value):.6f}"
            writer.writerow(row)


def write_confusion_matrix(path: Path, labels: np.ndarray, probs: np.ndarray) -> None:
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds, labels=list(range(len(EMOTIONS))))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt/pred", *EMOTIONS])
        for emotion, row in zip(EMOTIONS, cm):
            writer.writerow([emotion, *[int(v) for v in row]])


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

    run_name = args.run_name or f"au_{args.feature_set}_{args.model}_seed{args.seed}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cols = feature_columns(args.feature_set)
    data = split_dataset(au_csv, cols, args.val_size, args.seed)
    temps = parse_temperature_grid(args.temperature_grid)

    print(f"Training tabular AU calibrator: {run_name}")
    print(f"  AU CSV: {au_csv}")
    print(f"  Features: {args.feature_set} ({len(cols)} columns)")
    print(f"  Train/val/test: {len(data['y_train'])}/{len(data['y_val'])}/{len(data['y_test'])}")
    print(f"  Candidate model(s): {', '.join(candidate_names(args.model))}")

    candidates = []
    for name in candidate_names(args.model):
        print(f"Fitting {name}...")
        model = build_model(name, args)
        model.fit(data["x_train"], data["y_train"])
        val_probs_raw = model.predict_proba(data["x_val"])
        temperature, temp_scores = choose_temperature(data["y_val"], val_probs_raw, temps)
        val_probs = scale_probs(val_probs_raw, temperature)
        val_metrics = metrics_from_probs(data["y_val"], val_probs)
        candidates.append(
            {
                "name": name,
                "model": model,
                "temperature": temperature,
                "temperature_grid_nll": temp_scores,
                "val_metrics": val_metrics,
                "selection_score": score_for_selection(val_metrics, args.selection_metric),
            }
        )
        print(
            f"  val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_nll={val_metrics['nll']:.4f} temp={temperature:.2f}"
        )

    best = min(candidates, key=lambda item: item["selection_score"])
    model = best["model"]
    temperature = float(best["temperature"])

    if not args.no_refit_full_train:
        print(f"Refitting selected {best['name']} on all RAF-DB train rows...")
        model = build_model(best["name"], args)
        x_full_train = np.concatenate([data["x_train"], data["x_val"]], axis=0)
        y_full_train = np.concatenate([data["y_train"], data["y_val"]], axis=0)
        model.fit(x_full_train, y_full_train)

    train_probs = scale_probs(model.predict_proba(data["x_train"]), temperature)
    val_probs = scale_probs(model.predict_proba(data["x_val"]), temperature)
    test_probs = scale_probs(model.predict_proba(data["x_test"]), temperature)

    metrics = {
        "run_name": run_name,
        "au_csv": str(au_csv),
        "feature_set": args.feature_set,
        "feature_columns": cols,
        "seed": args.seed,
        "model_requested": args.model,
        "selected_model": best["name"],
        "refit_full_train": not args.no_refit_full_train,
        "selection_metric": args.selection_metric,
        "temperature": temperature,
        "class_weight": args.class_weight,
        "train_size": int(len(data["y_train"])),
        "val_size": int(len(data["y_val"])),
        "test_size": int(len(data["y_test"])),
        "candidates": [
            {
                "name": item["name"],
                "temperature": float(item["temperature"]),
                "val_metrics": item["val_metrics"],
                "selection_score": float(item["selection_score"]),
            }
            for item in candidates
        ],
        "train": metrics_from_probs(data["y_train"], train_probs),
        "val": metrics_from_probs(data["y_val"], val_probs),
        "test": metrics_from_probs(data["y_test"], test_probs),
        "classification_report_test": classification_report(
            data["y_test"],
            test_probs.argmax(axis=1),
            target_names=EMOTIONS,
            output_dict=True,
            zero_division=0,
        ),
    }

    joblib.dump(
        {
            "model": model,
            "model_type": best["name"],
            "feature_set": args.feature_set,
            "feature_columns": cols,
            "emotions": EMOTIONS,
            "temperature": temperature,
            "metrics": metrics,
        },
        run_dir / "au_emotion_tabular.joblib",
    )
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "resolved_au_csv": str(au_csv), "run_dir": str(run_dir)}, f, indent=2)
    with (run_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": cols,
                "train_counts": np.bincount(data["y_train"], minlength=len(EMOTIONS)).astype(int).tolist(),
                "val_counts": np.bincount(data["y_val"], minlength=len(EMOTIONS)).astype(int).tolist(),
                "test_counts": np.bincount(data["y_test"], minlength=len(EMOTIONS)).astype(int).tolist(),
                "emotions": EMOTIONS,
            },
            f,
            indent=2,
        )

    predictions_path = run_dir / "predictions.csv"
    if predictions_path.exists():
        predictions_path.unlink()
    write_predictions(predictions_path, "train", data["meta_train"], data["y_train"], train_probs)
    write_predictions(predictions_path, "val", data["meta_val"], data["y_val"], val_probs)
    write_predictions(predictions_path, "test", data["meta_test"], data["y_test"], test_probs)
    write_confusion_matrix(run_dir / "test_confusion_matrix.csv", data["y_test"], test_probs)

    print("Done.")
    print(f"  Selected model: {best['name']}")
    print(f"  Model file:      {run_dir / 'au_emotion_tabular.joblib'}")
    print(f"  Metrics:         {run_dir / 'metrics.json'}")
    print(f"  Test acc:        {metrics['test']['accuracy']:.4f}")
    print(f"  Test mF1:        {metrics['test']['macro_f1']:.4f}")
    print(f"  Test ECE:        {metrics['test']['ece']:.4f}")


if __name__ == "__main__":
    main()
