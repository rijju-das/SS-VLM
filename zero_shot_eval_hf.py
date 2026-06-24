#!/usr/bin/env python3
"""
Zero-shot facial expression evaluation with Hugging Face VLMs.

Supported models:
- InstructBLIP: Salesforce/instructblip-vicuna-7b
- LLaVA-1.5: llava-hf/llava-1.5-7b-hf

Supported datasets:
- RAF-DB (ImageFolder test split)
- FERPlus (ImageFolder test split or CSV-driven split)

Example:
python zero_shot_eval_hf.py \
  --model llava15 \
  --dataset rafdb \
  --data-root /path/to/RAFDB/DATASET \
  --split test \
  --output-dir outputs/llava15_rafdb
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from transformers import (
    AutoProcessor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    LlavaForConditionalGeneration,
)


RAFDB_LABELS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral",
]

FERPLUS_LABELS = [
    "Neutral",
    "Happiness",
    "Surprise",
    "Sadness",
    "Anger",
    "Disgust",
    "Fear",
    "Contempt",
]

SYNONYM_MAP = {
    "happy": "Happiness",
    "happiness": "Happiness",
    "sad": "Sadness",
    "sadness": "Sadness",
    "angry": "Anger",
    "anger": "Anger",
    "neutral": "Neutral",
    "surprised": "Surprise",
    "surprise": "Surprise",
    "fear": "Fear",
    "fearful": "Fear",
    "disgust": "Disgust",
    "disgusted": "Disgust",
    "contempt": "Contempt",
    "contemptuous": "Contempt",
}


@dataclass
class EvalResult:
    accuracy: float
    total: int
    correct: int
    confusion_matrix: List[List[int]]
    labels: List[str]


class FERPlusCSVDataset(Dataset):
    """
    Generic CSV loader for FERPlus-like annotation files.

    Required columns:
    - image column: one of ["image", "img", "path", "filename", "file"]
    - label column: one of ["label", "emotion", "class"]

    Label values can be:
    - integer indices
    - canonical emotion names
    """

    def __init__(self, image_root: Path, csv_path: Path, labels: Sequence[str]):
        self.image_root = image_root
        self.labels = list(labels)
        self.label_to_idx = {l.lower(): i for i, l in enumerate(self.labels)}
        self.samples: List[Tuple[Path, int]] = []

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {csv_path}")

            fields = {c.strip().lower(): c for c in reader.fieldnames}
            image_col = self._pick_field(fields, ["image", "img", "path", "filename", "file"])
            label_col = self._pick_field(fields, ["label", "emotion", "class"])

            if image_col is None or label_col is None:
                raise ValueError(
                    "Could not infer image/label columns from CSV header. "
                    "Expected image column in [image,img,path,filename,file] and "
                    "label column in [label,emotion,class]."
                )

            for row in reader:
                image_rel = str(row[image_col]).strip()
                label_raw = str(row[label_col]).strip()
                if not image_rel:
                    continue

                image_path = self.image_root / image_rel
                if not image_path.exists():
                    # Keep robust behavior for mixed path styles.
                    image_path = self.image_root / Path(image_rel).name
                if not image_path.exists():
                    continue

                label_idx = self._parse_label(label_raw)
                if label_idx is None:
                    continue

                self.samples.append((image_path, label_idx))

        if not self.samples:
            raise ValueError(
                f"No valid FERPlus samples found from csv={csv_path} image_root={image_root}"
            )

    @staticmethod
    def _pick_field(fields: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
        for c in candidates:
            if c in fields:
                return fields[c]
        return None

    def _parse_label(self, raw: str) -> Optional[int]:
        raw = raw.strip()
        if raw == "":
            return None

        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(self.labels):
                return idx
            return None

        key = raw.lower()
        if key in self.label_to_idx:
            return self.label_to_idx[key]

        mapped = SYNONYM_MAP.get(key)
        if mapped and mapped.lower() in self.label_to_idx:
            return self.label_to_idx[mapped.lower()]

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, str]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return image, label, str(path)


class ImageFolderWithPath(Dataset):
    def __init__(self, root: Path, labels: Sequence[str]):
        self.ds = ImageFolder(str(root))
        self.labels = list(labels)

        # If folder names are numeric (RAF-DB style), map by expected order.
        if all(k.isdigit() for k in self.ds.class_to_idx.keys()):
            class_names_sorted = sorted(self.ds.class_to_idx.keys(), key=lambda x: int(x))
            if len(class_names_sorted) != len(self.labels):
                raise ValueError(
                    f"Found {len(class_names_sorted)} class folders in {root}, "
                    f"but expected {len(self.labels)} labels."
                )
            idx_to_label = {self.ds.class_to_idx[k]: self.labels[i] for i, k in enumerate(class_names_sorted)}
            self.target_remap = {v: self.labels.index(idx_to_label[v]) for v in idx_to_label}
        else:
            # If class folders are names, match them to canonical labels.
            name_to_idx = {k.lower(): v for k, v in self.ds.class_to_idx.items()}
            self.target_remap = {}
            for label_idx, label in enumerate(self.labels):
                key = label.lower()
                if key in name_to_idx:
                    self.target_remap[name_to_idx[key]] = label_idx

            if len(self.target_remap) != len(self.labels):
                raise ValueError(
                    f"Could not map all class folders from {root} to labels {self.labels}. "
                    f"Found folders: {list(self.ds.class_to_idx.keys())}"
                )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, str]:
        path, target = self.ds.samples[idx]
        image = Image.open(path).convert("RGB")
        mapped_target = self.target_remap[target]
        return image, mapped_target, str(path)


def collate_no_stack(batch):
    images, labels, paths = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(paths)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_label(generated_text: str, labels: Sequence[str]) -> Optional[str]:
    text = normalize_text(generated_text)
    if not text:
        return None

    allowed = {l.lower(): l for l in labels}
    tokens = text.split()

    # Match the first non-negated label mention in the generated answer.
    # This is safer than scanning labels in a fixed order: if a verbose model
    # says "the answer is Fear", we should not let an echoed label list earlier
    # in the prompt bias the parser toward "Surprise".
    candidates: list[tuple[int, str]] = []
    for label in labels:
        key = label.lower()
        for match in re.finditer(rf"\b{re.escape(key)}\b", text):
            prefix = text[max(0, match.start() - 5) : match.start()].strip()
            if prefix.endswith("not"):
                continue
            candidates.append((match.start(), label))

    for token, mapped in SYNONYM_MAP.items():
        if mapped.lower() not in allowed:
            continue
        for match in re.finditer(rf"\b{re.escape(token)}\b", text):
            prefix = text[max(0, match.start() - 5) : match.start()].strip()
            if prefix.endswith("not"):
                continue
            candidates.append((match.start(), allowed[mapped.lower()]))

    if candidates:
        return sorted(candidates, key=lambda item: item[0])[0][1]

    # Last fallback: match first token if it equals one label.
    if tokens and tokens[0] in allowed:
        return allowed[tokens[0]]

    return None


def build_instruction(labels: Sequence[str]) -> str:
    joined = ", ".join(labels)
    return (
        "Classify the facial expression in this image. "
        f"Choose exactly one label from: {joined}. "
        "Reply with only the label text."
    )


def build_prompt(labels: Sequence[str], model_key: str) -> str:
    instruction = build_instruction(labels)
    if model_key == "llava15":
        return f"USER: <image>\n{instruction}\nASSISTANT:"
    return instruction


def load_model_and_processor(model_key: str, dtype: torch.dtype, device: torch.device):
    if model_key == "instructblip":
        model_id = "Salesforce/instructblip-vicuna-7b"
        processor = InstructBlipProcessor.from_pretrained(model_id)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    elif model_key == "llava15":
        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    else:
        raise ValueError(f"Unsupported model key: {model_key}")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if model_key == "llava15":
            tokenizer.padding_side = "left"

    model.to(device)
    model.eval()
    return model_id, model, processor


def generate_batch(
    model_key: str,
    model,
    processor,
    images: Sequence[Image.Image],
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 16,
) -> List[str]:
    if model_key == "instructblip":
        inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True)
    else:
        inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decoder-only VLMs often return the full prompt plus new tokens. If we
    # parse that directly, the label list in the prompt can be mistaken for
    # the answer. Slice generated ids when possible, and still strip echoed
    # prompt text as a fallback for encoder-decoder models.
    input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    if input_len and generated.shape[1] > input_len:
        decoded = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)
    else:
        decoded = processor.batch_decode(generated, skip_special_tokens=True)

    return [strip_echoed_prompt(text, prompt) for text in decoded]


def strip_echoed_prompt(text: str, prompt: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    if cleaned.startswith(prompt):
        cleaned = cleaned[len(prompt) :].strip()

    lower_cleaned = cleaned.lower()
    lower_prompt = prompt.lower()
    if lower_cleaned.startswith(lower_prompt):
        cleaned = cleaned[len(prompt) :].strip()
        lower_cleaned = cleaned.lower()

    for marker in ["ASSISTANT:", "Reply with only the label text."]:
        lower_marker = marker.lower()
        if lower_marker in lower_cleaned:
            idx = lower_cleaned.rfind(lower_marker)
            cleaned = cleaned[idx + len(marker) :].strip()
            lower_cleaned = cleaned.lower()

    return cleaned


def evaluate(
    model_key: str,
    model,
    processor,
    dataloader: DataLoader,
    labels: Sequence[str],
    device: torch.device,
    output_dir: Path,
    max_new_tokens: int,
) -> EvalResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(labels, model_key)

    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)

    rows = []
    total = 0
    correct = 0
    parsed = 0

    for images, y_true, paths in dataloader:
        texts = generate_batch(
            model_key=model_key,
            model=model,
            processor=processor,
            images=images,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        for gt_idx, pred_text, p in zip(y_true.tolist(), texts, paths):
            pred_label = extract_label(pred_text, labels)
            if pred_label is None:
                pred_idx = labels.index("Neutral") if "Neutral" in labels else 0
                parsed_ok = 0
            else:
                pred_idx = labels.index(pred_label)
                parsed_ok = 1
                parsed += 1

            cm[gt_idx, pred_idx] += 1
            total += 1
            if gt_idx == pred_idx:
                correct += 1

            rows.append(
                {
                    "path": p,
                    "gt_label": labels[gt_idx],
                    "pred_label": labels[pred_idx],
                    "parsed_ok": parsed_ok,
                    "raw_generation": pred_text,
                }
            )

    acc = (correct / total) if total > 0 else 0.0

    pred_path = output_dir / "predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "gt_label", "pred_label", "parsed_ok", "raw_generation"],
        )
        writer.writeheader()
        writer.writerows(rows)

    result = EvalResult(
        accuracy=acc,
        total=total,
        correct=correct,
        confusion_matrix=cm.tolist(),
        labels=list(labels),
    )

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "parsed": parsed,
                "parse_rate": (parsed / total) if total else 0.0,
                "labels": list(labels),
                "confusion_matrix": cm.tolist(),
            },
            f,
            indent=2,
        )

    return result


def maybe_limit_dataset(ds: Dataset, max_samples: int) -> Dataset:
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds

    indices = np.random.RandomState(42).choice(len(ds), size=max_samples, replace=False)

    class SubsetDS(Dataset):
        def __init__(self, base: Dataset, idxs: np.ndarray):
            self.base = base
            self.idxs = idxs.tolist()

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i: int):
            return self.base[self.idxs[i]]

    return SubsetDS(ds, indices)


def build_dataset(args) -> Tuple[Dataset, List[str]]:
    if args.dataset == "rafdb":
        labels = RAFDB_LABELS
        split_root = Path(args.data_root) / args.split
        if not split_root.exists():
            alt = Path(args.data_root) / "original" / args.split
            if alt.exists():
                split_root = alt
        if not split_root.exists():
            raise FileNotFoundError(
                f"RAF-DB split folder not found. Tried: {Path(args.data_root) / args.split} and "
                f"{Path(args.data_root) / 'original' / args.split}"
            )
        ds = ImageFolderWithPath(split_root, labels)
        return ds, labels

    if args.dataset == "ferplus":
        labels = FERPLUS_LABELS

        # Option A: ImageFolder split.
        split_root = Path(args.data_root) / args.split
        if split_root.exists():
            try:
                ds = ImageFolderWithPath(split_root, labels)
                return ds, labels
            except Exception:
                pass

        # Option B: CSV + image root.
        if args.ferplus_csv and args.ferplus_image_root:
            ds = FERPlusCSVDataset(
                image_root=Path(args.ferplus_image_root),
                csv_path=Path(args.ferplus_csv),
                labels=labels,
            )
            return ds, labels

        raise FileNotFoundError(
            "FERPlus data not found in ImageFolder format, and CSV arguments were not provided. "
            "Use either --data-root with split folder, or provide --ferplus-csv and --ferplus-image-root."
        )

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot VLM evaluation on RAF-DB / FERPlus")
    p.add_argument("--model", choices=["instructblip", "llava15"], required=True)
    p.add_argument("--dataset", choices=["rafdb", "ferplus"], required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--output-dir", type=str, required=True)

    # FERPlus fallback options.
    p.add_argument("--ferplus-csv", type=str, default="")
    p.add_argument("--ferplus-image-root", type=str, default="")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return p.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print("[WARN] Non-float32 dtype requested on CPU; using float32 for compatibility.")
        dtype = torch.float32

    ds, labels = build_dataset(args)
    ds = maybe_limit_dataset(ds, args.max_samples)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_no_stack,
    )

    model_id, model, processor = load_model_and_processor(args.model, dtype=dtype, device=device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] model={args.model} ({model_id})")
    print(f"[INFO] dataset={args.dataset} split={args.split} size={len(ds)}")
    print(f"[INFO] output_dir={out_dir}")

    result = evaluate(
        model_key=args.model,
        model=model,
        processor=processor,
        dataloader=loader,
        labels=labels,
        device=device,
        output_dir=out_dir,
        max_new_tokens=args.max_new_tokens,
    )

    print("[RESULT] accuracy={:.4f} ({}/{})".format(result.accuracy, result.correct, result.total))


if __name__ == "__main__":
    main()
