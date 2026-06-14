#!/usr/bin/env python3
"""Extract full OpenFace 2 AU intensities for RAF-DB image folders.

Expected RAF-DB layout:
    RAF-DB/
      train/1/*.jpg
      train/2/*.jpg
      ...
      test/1/*.jpg

Also supported:
    RAF-DB/DATASET/train/1/*.jpg
    RAF-DB/original/train/1/*.jpg
    RAF-DB/DATASET/original/train/1/*.jpg

This script calls classic OpenFace 2 FeatureExtraction per image and writes one
row per image with the full OpenFace AU intensity set. Runs are resumable.
"""

import argparse
import csv
import os
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

AU_R_COLUMNS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]

AU_C_COLUMNS = [
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
    "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
    "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract full OpenFace 2 AU intensities for RAF-DB")
    parser.add_argument("--data-root", required=True, help="Path to RAF-DB root")
    parser.add_argument("--output-csv", default="outputs/rafdb_openface2_aus.csv")
    parser.add_argument("--raw-dir", default="outputs/openface2_raw")
    parser.add_argument("--openface-bin", default="FeatureExtraction", help="Path to OpenFace 2 FeatureExtraction")
    parser.add_argument(
        "--openface-cwd",
        default="",
        help="OpenFace 2 repo root. Auto-detected from --openface-bin when possible.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--max-images", type=int, default=0, help="Optional smoke-test limit")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess rows already present")
    parser.add_argument("--dry-run", action="store_true", help="Only count images and verify paths")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument passed through to FeatureExtraction; repeat as needed",
    )
    return parser.parse_args()


def resolve_binary(openface_bin):
    path = Path(openface_bin)
    if path.exists():
        return str(path)
    resolved = shutil.which(openface_bin)
    if resolved:
        return resolved
    raise SystemExit(
        "OpenFace 2 FeatureExtraction was not found. "
        "Pass --openface-bin /path/to/OpenFace-2/build/bin/FeatureExtraction."
    )


def resolve_openface_cwd(openface_bin, openface_cwd):
    if openface_cwd:
        return openface_cwd

    path = Path(openface_bin)
    if not path.exists():
        return None

    for parent in path.resolve().parents:
        if (parent / "build").exists() and (
            (parent / "model").exists()
            or (parent / "CMakeLists.txt").exists()
            or (parent / "download_models.sh").exists()
        ):
            return str(parent)
    return None


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
    return any(row.get(col, "").strip() for col in AU_R_COLUMNS)


def load_existing(output_csv):
    if not output_csv.exists():
        return set()
    with output_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row.get("relative_path", "") for row in reader if has_usable_row(row)}


def parse_openface2_csv(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [{str(k).strip(): str(v).strip() for k, v in row.items()} for row in reader]

    if not rows:
        raise RuntimeError(f"OpenFace 2 CSV has no rows: {csv_path}")

    result = {}
    for col in ["confidence", "success", *AU_R_COLUMNS, *AU_C_COLUMNS]:
        values = []
        for row in rows:
            raw = row.get(col)
            if raw in (None, ""):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        result[col] = sum(values) / len(values) if values else ""
    return result


def safe_output_dir(raw_root, relative_path):
    stem = relative_path.with_suffix("").as_posix().replace("/", "__")
    return raw_root / stem


def run_openface2(openface_bin, image_path, output_dir, extra_args, openface_cwd):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        openface_bin,
        "-f",
        str(image_path),
        "-out_dir",
        str(output_dir),
        "-aus",
        "-au_static",
        "-q",
        *extra_args,
    ]
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
        cwd=resolve_openface_cwd(openface_bin, openface_cwd),
    )
    csv_files = sorted(output_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if completed.returncode != 0 and not csv_files:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    if not csv_files:
        raise RuntimeError("OpenFace 2 completed but did not produce a CSV file")
    return parse_openface2_csv(csv_files[0])


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
        "openface_success", "openface_confidence", *AU_R_COLUMNS, *AU_C_COLUMNS, "error",
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
                au_result = run_openface2(
                    openface_bin,
                    image_path,
                    safe_output_dir(raw_dir, rel_path),
                    args.extra_arg,
                    args.openface_cwd,
                )
                row["openface_success"] = au_result.get("success", "")
                row["openface_confidence"] = au_result.get("confidence", "")
                for col in [*AU_R_COLUMNS, *AU_C_COLUMNS]:
                    row[col] = au_result.get(col, "")
            except Exception as exc:
                failed += 1
                row["openface_success"] = 0
                row["openface_confidence"] = ""
                for col in [*AU_R_COLUMNS, *AU_C_COLUMNS]:
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
