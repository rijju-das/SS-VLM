#!/usr/bin/env python3
"""Extract OpenFace 3 partial AU estimates for RAF-DB image folders.

OpenFace 3's CLI emits an 8-AU subset in an ``action_units`` field, not the
full OpenFace 2 AU intensity set. Use ``extract_openface2_aus.py`` when the
experiment requires full FACS-style AU evidence.
"""

import argparse
import ast
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


RAFDB_LABELS = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happiness",
    "5": "Sadness",
    "6": "Anger",
    "7": "Neutral",
}

AU_COLUMNS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]

# OpenFace 3.0's multitask model emits 8 unnamed DISFA-trained AU outputs.
# The training code selects eval_au=[0,4,8,10,11,1,6,7] from the common DISFA
# label order [AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26].
OPENFACE3_AU_COLUMNS = [
    "AU01_r", "AU06_r", "AU17_r", "AU25_r",
    "AU26_r", "AU02_r", "AU12_r", "AU15_r",
]

RAW_COLUMNS = [
    "face_id", "face_detection", "landmarks", "emotion",
    "gaze_yaw", "gaze_pitch", "action_units",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract OpenFace 3 partial AU estimates for RAF-DB")
    parser.add_argument("--data-root", required=True, help="Path to RAF-DB root")
    parser.add_argument("--output-csv", default="outputs/rafdb_openface3_aus.csv")
    parser.add_argument("--raw-dir", default="outputs/openface3_raw")
    parser.add_argument("--openface-bin", default="openface", help="OpenFace 3 CLI executable")
    parser.add_argument("--openface-cwd", default="", help="OpenFace 3 checkout containing the weights/ folder")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-visible-devices", default="", help="Optional CUDA_VISIBLE_DEVICES value, e.g. 0")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--max-images", type=int, default=0, help="Optional smoke-test limit")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess rows already present")
    parser.add_argument("--dry-run", action="store_true", help="Only count images and verify paths")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument passed through to `openface detect`; repeat as needed",
    )
    return parser.parse_args()


def resolve_binary(openface_bin):
    path = Path(openface_bin)
    if path.exists():
        return str(path)
    resolved = shutil.which(openface_bin)
    if resolved:
        return resolved
    raise SystemExit("OpenFace 3 CLI was not found. Pass --openface-bin /path/to/openface.")


def find_split_root(data_root, split):
    candidates = [
        data_root / split,
        data_root / "DATASET" / split,
        data_root / "original" / split,
        data_root / "DATASET" / "original" / split,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find split '{split}' under {data_root}")


def iter_rafdb_images(data_root, splits):
    for split in splits:
        split_root = find_split_root(data_root, split)
        for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            class_id = class_dir.name
            class_name = RAFDB_LABELS.get(class_id, class_id)
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield split, class_id, class_name, image_path


def has_usable_row(row):
    if row.get("error", "").strip():
        return False
    if row.get("openface_success", "").strip().lower() in {"0", "false"}:
        return False
    return any(row.get(col, "").strip() for col in OPENFACE3_AU_COLUMNS)


def load_existing(output_csv):
    if not output_csv.exists():
        return set()
    with output_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row.get("relative_path", "") for row in reader if has_usable_row(row)}


def safe_output_dir(raw_root, relative_path):
    stem = relative_path.with_suffix("").as_posix().replace("/", "__")
    return raw_root / stem


def normalize_au_key(key):
    match = re.search(r"AU0?(\d+)", str(key), flags=re.IGNORECASE)
    if not match:
        return None
    return f"AU{int(match.group(1)):02d}_r"


def flatten_action_units(values):
    while (
        isinstance(values, (list, tuple))
        and len(values) == 1
        and isinstance(values[0], (list, tuple))
    ):
        values = values[0]
    return values


def parse_action_units(raw):
    if raw in (None, ""):
        return {}

    parsed = None
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(raw)
            break
        except Exception:
            pass

    if isinstance(parsed, dict):
        out = {}
        for key, value in parsed.items():
            norm_key = normalize_au_key(key)
            if norm_key:
                try:
                    out[norm_key] = float(value)
                except (TypeError, ValueError):
                    pass
        return out

    values = flatten_action_units(parsed) if isinstance(parsed, (list, tuple)) else None
    if values is None:
        values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)

    out = {}
    for col, value in zip(OPENFACE3_AU_COLUMNS, values):
        try:
            out[col] = float(value)
        except (TypeError, ValueError):
            pass
    return out


def parse_numeric_list(raw):
    if raw in (None, ""):
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(raw)
            break
        except Exception:
            parsed = None
    if not isinstance(parsed, (list, tuple)):
        return []
    return parsed


def face_confidence(row):
    detection = parse_numeric_list(row.get("face_detection"))
    try:
        return float(detection[4])
    except (IndexError, TypeError, ValueError):
        return 0.0


def parse_openface3_table(table_path):
    delimiter = "\t" if table_path.suffix.lower() == ".tsv" else ","
    with table_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = [{str(k).strip(): str(v).strip() for k, v in row.items()} for row in reader]

    if not rows:
        raise RuntimeError(f"OpenFace 3 output table has no rows: {table_path}")

    best = max(rows, key=face_confidence)
    result = {col: "" for col in AU_COLUMNS}
    result.update(parse_action_units(best.get("action_units")))
    result["success"] = 1
    result["confidence"] = face_confidence(best)
    for col in RAW_COLUMNS:
        result[col] = best.get(col, "")
    return result


def run_openface3(openface_bin, image_path, output_dir, extra_args, device, cuda_visible_devices, openface_cwd):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        openface_bin,
        "detect",
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        *extra_args,
        str(image_path),
    ]
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=openface_cwd or None,
    )
    output_files = sorted(
        [*output_dir.glob("*.csv"), *output_dir.glob("*.tsv")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if completed.returncode != 0 and not output_files:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    if not output_files:
        raise RuntimeError("OpenFace 3 completed but did not produce a CSV/TSV file")
    return parse_openface3_table(output_files[0])


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    images = list(iter_rafdb_images(data_root, args.splits))

    if args.max_images:
        images = images[: args.max_images]

    if args.dry_run:
        print(f"Found {len(images)} image(s) under {data_root}")
        return

    openface_bin = resolve_binary(args.openface_bin)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = set() if args.no_resume else load_existing(output_csv)
    fieldnames = [
        "split", "class_id", "class_name", "image_path", "relative_path",
        "openface_success", "openface_confidence", *AU_COLUMNS, *RAW_COLUMNS, "error",
    ]
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0

    processed = 0
    skipped = 0
    failed = 0
    with output_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for split, class_id, class_name, image_path in images:
            rel_path = image_path.relative_to(data_root)
            rel_text = rel_path.as_posix()
            if rel_text in existing:
                skipped += 1
                continue

            row = {
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "image_path": str(image_path),
                "relative_path": rel_text,
                "error": "",
            }

            try:
                au_result = run_openface3(
                    openface_bin,
                    image_path,
                    safe_output_dir(raw_dir, rel_path),
                    args.extra_arg,
                    args.device,
                    args.cuda_visible_devices,
                    args.openface_cwd,
                )
                row["openface_success"] = au_result.get("success", "")
                row["openface_confidence"] = au_result.get("confidence", "")
                for col in [*AU_COLUMNS, *RAW_COLUMNS]:
                    row[col] = au_result.get(col, "")
            except Exception as exc:
                failed += 1
                row["openface_success"] = 0
                row["openface_confidence"] = ""
                for col in [*AU_COLUMNS, *RAW_COLUMNS]:
                    row[col] = ""
                row["error"] = str(exc)

            writer.writerow(row)
            f.flush()
            processed += 1
            if processed % 100 == 0:
                print(f"Processed={processed} skipped={skipped} failed={failed}")

    print(f"Done. Processed={processed} skipped={skipped} failed={failed} output={output_csv}")


if __name__ == "__main__":
    main()
