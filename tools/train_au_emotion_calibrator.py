#!/usr/bin/env python3
"""Train a learned AU-to-emotion calibrator for RAF-DB.

This script replaces the hand-written AU-to-emotion lookup with a lightweight
model trained from OpenFace AU values and RAF-DB emotion labels. It does not
change the SS-VLM/RAG pipeline directly; it saves a checkpoint and prediction
files that can later be used as a learned AU prior.

Expected input CSV: outputs/rafdb_openface2_aus.csv from extract_openface2_aus.py
Expected RAF-DB labels in CSV:
    class_id 1 Surprise, 2 Fear, 3 Disgust, 4 Happiness,
             5 Sadness, 6 Anger, 7 Neutral
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


EMOTIONS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

AU_R_COLUMNS = [
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU07_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU25_r",
    "AU26_r",
    "AU45_r",
]

AU_C_COLUMNS = [
    "AU01_c",
    "AU02_c",
    "AU04_c",
    "AU05_c",
    "AU06_c",
    "AU07_c",
    "AU09_c",
    "AU10_c",
    "AU12_c",
    "AU14_c",
    "AU15_c",
    "AU17_c",
    "AU20_c",
    "AU23_c",
    "AU25_c",
    "AU26_c",
    "AU28_c",
    "AU45_c",
]


class AUEmotionMLP(nn.Module):
    """Small tabular classifier for OpenFace AU features."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, num_classes: int = 7):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    meta_train: list[dict]
    x_val: np.ndarray
    y_val: np.ndarray
    meta_val: list[dict]
    x_test: np.ndarray
    y_test: np.ndarray
    meta_test: list[dict]
    feature_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learned AU-to-emotion model for RAF-DB.")
    parser.add_argument("--au-csv", default="outputs/rafdb_openface2_aus.csv")
    parser.add_argument("--output-dir", default="outputs_v2/au_calibrator")
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--feature-set",
        choices=["intensity", "presence", "both"],
        default="intensity",
        help="Use AUxx_r intensities, AUxx_c presence flags, or both.",
    )
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split from RAF-DB train rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", default="64,32", help="Comma-separated hidden dimensions.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "accuracy", "loss"],
        default="macro_f1",
        help="Validation metric used for best checkpoint selection.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--temperature-grid",
        default="0.50,0.60,0.70,0.80,0.90,1.00,1.10,1.20,1.35,1.50,1.75,2.00,2.50,3.00,4.00,5.00",
        help="Candidate temperatures for post-hoc validation NLL calibration.",
    )
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def parse_hidden_dims(value: str) -> list[int]:
    dims = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not dims:
        raise ValueError("--hidden-dims must contain at least one integer")
    return dims


def parse_temperature_grid(value: str) -> list[float]:
    temps = [float(item.strip()) for item in value.split(",") if item.strip()]
    temps = [temp for temp in temps if temp > 0]
    if not temps:
        raise ValueError("--temperature-grid must contain positive floats")
    return temps


def feature_columns(feature_set: str) -> list[str]:
    if feature_set == "intensity":
        return AU_R_COLUMNS
    if feature_set == "presence":
        return AU_C_COLUMNS
    return [*AU_R_COLUMNS, *AU_C_COLUMNS]


def safe_float(value: str) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def row_is_valid(row: dict, cols: list[str]) -> bool:
    if row.get("error", "").strip():
        return False
    if row.get("openface_success", "").strip().lower() in {"0", "false"}:
        return False
    if row.get("split", "").strip() not in {"train", "test"}:
        return False
    try:
        class_id = int(row.get("class_id", ""))
    except ValueError:
        return False
    if class_id < 1 or class_id > 7:
        return False
    return all(np.isfinite(safe_float(row.get(col, ""))) for col in cols)


def read_au_csv(csv_path: Path, cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    features: list[list[float]] = []
    labels: list[int] = []
    metas: list[dict] = []
    splits: list[str] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [col for col in cols if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing AU columns in {csv_path}: {missing}")

        for row_index, row in enumerate(reader):
            if not row_is_valid(row, cols):
                continue
            class_id = int(row["class_id"])
            label = class_id - 1
            values = [safe_float(row[col]) for col in cols]
            features.append(values)
            labels.append(label)
            splits.append(row["split"].strip())
            metas.append(
                {
                    "row_index": row_index,
                    "split": row["split"].strip(),
                    "class_id": class_id,
                    "class_name": row.get("class_name", EMOTIONS[label]),
                    "image_path": row.get("image_path", ""),
                    "relative_path": row.get("relative_path", ""),
                    "openface_confidence": row.get("openface_confidence", ""),
                }
            )

    if not features:
        raise ValueError(f"No valid AU rows found in {csv_path}")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), metas, np.asarray(splits)


def make_dataset_bundle(csv_path: Path, cols: list[str], val_size: float, seed: int) -> DatasetBundle:
    x_all, y_all, metas, splits = read_au_csv(csv_path, cols)
    train_mask = splits == "train"
    test_mask = splits == "test"

    x_train_full = x_all[train_mask]
    y_train_full = y_all[train_mask]
    meta_train_full = [meta for meta, keep in zip(metas, train_mask) if keep]
    x_test = x_all[test_mask]
    y_test = y_all[test_mask]
    meta_test = [meta for meta, keep in zip(metas, test_mask) if keep]

    if len(x_train_full) == 0 or len(x_test) == 0:
        raise ValueError("Expected both train and test rows in the AU CSV.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_indices, val_indices = next(splitter.split(x_train_full, y_train_full))

    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices]
    meta_train = [meta_train_full[i] for i in train_indices]
    x_val = x_train_full[val_indices]
    y_val = y_train_full[val_indices]
    meta_val = [meta_train_full[i] for i in val_indices]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    bundle = DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        meta_train=meta_train,
        x_val=x_val,
        y_val=y_val,
        meta_val=meta_val,
        x_test=x_test,
        y_test=y_test,
        meta_test=meta_test,
        feature_columns=cols,
    )
    bundle.scaler = scaler  # type: ignore[attr-defined]
    return bundle


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(EMOTIONS)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    logits_np = np.concatenate(logits_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    return total_loss / max(total_items, 1), logits_np, labels_np


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(float(temperature), 1e-8)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if not np.any(mask):
            continue
        ece += np.mean(mask) * abs(float(np.mean(accuracies[mask])) - float(np.mean(confidences[mask])))
    return float(ece)


def metrics_from_logits(labels: np.ndarray, logits: np.ndarray, temperature: float = 1.0) -> dict:
    probs = softmax_np(logits, temperature=temperature)
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


def choose_temperature(labels: np.ndarray, logits: np.ndarray, grid: list[float]) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    best_temp = 1.0
    best_nll = math.inf
    for temp in grid:
        probs = softmax_np(logits, temperature=temp)
        nll = float(log_loss(labels, probs, labels=list(range(len(EMOTIONS)))))
        scores[f"{temp:.4f}"] = nll
        if nll < best_nll:
            best_nll = nll
            best_temp = temp
    return best_temp, scores


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(path: Path, labels: np.ndarray, logits: np.ndarray, temperature: float) -> None:
    preds = softmax_np(logits, temperature=temperature).argmax(axis=1)
    cm = confusion_matrix(labels, preds, labels=list(range(len(EMOTIONS))))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt/pred", *EMOTIONS])
        for emotion, row in zip(EMOTIONS, cm):
            writer.writerow([emotion, *[int(v) for v in row]])


def write_predictions(
    path: Path,
    subset: str,
    meta: list[dict],
    labels: np.ndarray,
    logits: np.ndarray,
    temperature: float,
) -> None:
    probs = softmax_np(logits, temperature=temperature)
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def tensor_logits(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = make_loader(x, np.zeros(len(x), dtype=np.int64), batch_size=batch_size, shuffle=False)
    model.eval()
    logits_all = []
    with torch.no_grad():
        for features, _ in loader:
            logits_all.append(model(features.to(device)).cpu().numpy())
    return np.concatenate(logits_all, axis=0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.deterministic)

    repo_root = Path(__file__).resolve().parents[1]
    au_csv = Path(args.au_csv).expanduser()
    if not au_csv.is_absolute():
        au_csv = repo_root / au_csv
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    run_name = args.run_name or f"au_{args.feature_set}_mlp_seed{args.seed}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cols = feature_columns(args.feature_set)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    temperatures = parse_temperature_grid(args.temperature_grid)
    bundle = make_dataset_bundle(au_csv, cols, args.val_size, args.seed)
    scaler: StandardScaler = bundle.scaler  # type: ignore[attr-defined]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AUEmotionMLP(
        input_dim=len(cols),
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        num_classes=len(EMOTIONS),
    ).to(device)
    weights = None if args.no_class_weights else class_weights(bundle.y_train, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_loader(bundle.x_train, bundle.y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(bundle.x_val, bundle.y_val, args.batch_size, shuffle=False)

    best_state = None
    best_epoch = 0
    best_score = math.inf if args.selection_metric == "loss" else -math.inf
    stale_epochs = 0
    history_rows: list[dict] = []

    print(f"Training AU calibrator: {run_name}")
    print(f"  AU CSV: {au_csv}")
    print(f"  Features: {args.feature_set} ({len(cols)} columns)")
    print(f"  Train/val/test: {len(bundle.y_train)}/{len(bundle.y_val)}/{len(bundle.y_test)}")
    print(f"  Device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_logits, train_labels = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_logits, val_labels = run_epoch(model, val_loader, device, criterion)
        train_metrics = metrics_from_logits(train_labels, train_logits)
        val_metrics = metrics_from_logits(val_labels, val_logits)
        if args.selection_metric == "loss":
            current_score = val_loss
            is_best = current_score < best_score - args.min_delta
        elif args.selection_metric == "accuracy":
            current_score = val_metrics["accuracy"]
            is_best = current_score > best_score + args.min_delta
        else:
            current_score = val_metrics["macro_f1"]
            is_best = current_score > best_score + args.min_delta

        if is_best:
            best_score = current_score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_metrics["accuracy"],
                "val_acc": val_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "train_weighted_f1": train_metrics["weighted_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "is_best": int(is_best),
            }
        )

        if epoch == 1 or epoch % 10 == 0 or is_best:
            print(
                f"Epoch {epoch:03d}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)

    train_logits = tensor_logits(model, bundle.x_train, args.batch_size, device)
    val_logits = tensor_logits(model, bundle.x_val, args.batch_size, device)
    test_logits = tensor_logits(model, bundle.x_test, args.batch_size, device)
    best_temperature, temp_grid_scores = choose_temperature(bundle.y_val, val_logits, temperatures)

    metrics = {
        "run_name": run_name,
        "au_csv": str(au_csv),
        "feature_set": args.feature_set,
        "feature_columns": cols,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_selection_score": float(best_score),
        "best_temperature": best_temperature,
        "train_size": int(len(bundle.y_train)),
        "val_size": int(len(bundle.y_val)),
        "test_size": int(len(bundle.y_test)),
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "class_weights": None if weights is None else weights.detach().cpu().tolist(),
        "temperature_grid_nll": temp_grid_scores,
        "uncalibrated": {
            "train": metrics_from_logits(bundle.y_train, train_logits, temperature=1.0),
            "val": metrics_from_logits(bundle.y_val, val_logits, temperature=1.0),
            "test": metrics_from_logits(bundle.y_test, test_logits, temperature=1.0),
        },
        "temperature_scaled": {
            "train": metrics_from_logits(bundle.y_train, train_logits, temperature=best_temperature),
            "val": metrics_from_logits(bundle.y_val, val_logits, temperature=best_temperature),
            "test": metrics_from_logits(bundle.y_test, test_logits, temperature=best_temperature),
        },
        "classification_report_test": classification_report(
            bundle.y_test,
            softmax_np(test_logits, temperature=best_temperature).argmax(axis=1),
            target_names=EMOTIONS,
            output_dict=True,
            zero_division=0,
        ),
    }

    checkpoint_path = run_dir / "au_emotion_calibrator.pth"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_class": "AUEmotionMLP",
            "input_dim": len(cols),
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "num_classes": len(EMOTIONS),
            "emotions": EMOTIONS,
            "feature_set": args.feature_set,
            "feature_columns": cols,
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
            "temperature": float(best_temperature),
            "metrics": metrics,
        },
        checkpoint_path,
    )

    write_history(run_dir / "history.csv", history_rows)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "resolved_au_csv": str(au_csv), "run_dir": str(run_dir)}, f, indent=2)
    with (run_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": cols,
                "train_counts": np.bincount(bundle.y_train, minlength=len(EMOTIONS)).astype(int).tolist(),
                "val_counts": np.bincount(bundle.y_val, minlength=len(EMOTIONS)).astype(int).tolist(),
                "test_counts": np.bincount(bundle.y_test, minlength=len(EMOTIONS)).astype(int).tolist(),
                "emotions": EMOTIONS,
            },
            f,
            indent=2,
        )

    predictions_path = run_dir / "predictions.csv"
    if predictions_path.exists():
        predictions_path.unlink()
    write_predictions(predictions_path, "train", bundle.meta_train, bundle.y_train, train_logits, best_temperature)
    write_predictions(predictions_path, "val", bundle.meta_val, bundle.y_val, val_logits, best_temperature)
    write_predictions(predictions_path, "test", bundle.meta_test, bundle.y_test, test_logits, best_temperature)
    write_confusion_matrix(run_dir / "test_confusion_matrix.csv", bundle.y_test, test_logits, best_temperature)

    test_metrics = metrics["temperature_scaled"]["test"]
    print("Done.")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Metrics:    {run_dir / 'metrics.json'}")
    print(f"  Test acc:   {test_metrics['accuracy']:.4f}")
    print(f"  Test mF1:   {test_metrics['macro_f1']:.4f}")
    print(f"  Test ECE:   {test_metrics['ece']:.4f}")


if __name__ == "__main__":
    main()
