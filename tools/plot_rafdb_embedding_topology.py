#!/usr/bin/env python3
"""Generate RAF-DB t-SNE/UMAP topology plots from SS-VLM v2 checkpoints.

The script loads one or more v2 checkpoints, extracts normalized test-set
embeddings from RAF-DB, reduces them to 2-D with t-SNE or UMAP, and saves a
tight-cropped PDF suitable for the manuscript.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


EMOTIONS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
CLASS_COLORS = {
    "Surprise": "#d62728",
    "Fear": "#9467bd",
    "Disgust": "#2ca02c",
    "Happiness": "#ffb000",
    "Sadness": "#1f77b4",
    "Anger": "#ff7f0e",
    "Neutral": "#6f6f6f",
}


@dataclass(frozen=True)
class RunSpec:
    label: str
    variant: str
    checkpoint: Path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_run_spec(text: str) -> RunSpec:
    """Parse LABEL|VARIANT|CHECKPOINT."""
    parts = [part.strip() for part in text.split("|")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--run must use the format 'LABEL|VARIANT|CHECKPOINT', "
            "for example 'SFRA-RAG|sfra_v2|outputs_v2/models/seed_sweep/sfra_v2_l005_seed42.pth'"
        )
    label, variant, checkpoint = parts
    if variant not in {"plain_vit", "vit_gem", "sfra_v2", "sfra_legacy"}:
        raise argparse.ArgumentTypeError(f"Unknown model variant: {variant}")
    return RunSpec(label=label, variant=variant, checkpoint=Path(checkpoint))


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(description="Plot RAF-DB embedding topology from v2 checkpoints.")
    parser.add_argument("--data-root", default=os.environ.get("SSVLM_DATA_DIR", "/home/rdas/RAF-DB"))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--method", default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional class-balanced sample cap.")
    parser.add_argument("--perplexity", type=float, default=35.0, help="t-SNE perplexity.")
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.72)
    parser.add_argument("--no-panel-titles", action="store_true")
    parser.add_argument("--save-embeddings", default="", help="Optional .npz path for extracted embeddings.")
    parser.add_argument(
        "--output",
        default="outputs_v2/figures/rafdb_embedding_topology_tsne.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        help="Run spec in the form LABEL|VARIANT|CHECKPOINT. Repeat for multiple panels.",
    )
    parser.add_argument("--pad-inches", type=float, default=0.0)
    args = parser.parse_args()
    if args.run is None:
        args.run = [
            RunSpec("Plain ViT", "plain_vit", root / "outputs_v2/models/seed_sweep/plain_vit_ce_seed2026.pth"),
            RunSpec("SFRA-RAG", "sfra_v2", root / "outputs_v2/models/seed_sweep/sfra_v2_l005_seed42.pth"),
        ]
    return args


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_pipeline(root: Path):
    pipeline_path = root / "Pipeline/SS-VLM_Pipeline_v2.py"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Could not find v2 pipeline: {pipeline_path}")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("ssvlm_pipeline_v2_for_topology", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_balanced_indices(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= len(labels):
        return np.arange(len(labels))
    rng = random.Random(seed)
    per_class = max(1, max_samples // len(EMOTIONS))
    selected: list[int] = []
    for cls_idx in range(len(EMOTIONS)):
        cls_indices = np.where(labels == cls_idx)[0].tolist()
        rng.shuffle(cls_indices)
        selected.extend(cls_indices[:per_class])
    if len(selected) < max_samples:
        remaining = [idx for idx in range(len(labels)) if idx not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)])
    return np.array(sorted(selected), dtype=int)


def extract_embeddings(module, run: RunSpec, data_root: str, split: str, batch_size: int, num_workers: int, device):
    checkpoint = run.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found for {run.label}: {checkpoint}")

    loader = module.get_eval_loader(data_root, split, batch_size, num_workers)
    model = module.SSVLMExperimentModel(num_classes=module.NUM_CLASSES, model_variant=run.variant).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_paths: list[str] = []
    dataset_samples = getattr(loader.dataset, "samples", None)
    offset = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            embeddings = model(imgs, mode="extract").detach().cpu().numpy()
            all_embeddings.append(embeddings)
            all_labels.append(labels.numpy())
            if dataset_samples is not None:
                all_paths.extend(sample[0] for sample in dataset_samples[offset : offset + len(labels)])
            offset += len(labels)

    return {
        "label": run.label,
        "variant": run.variant,
        "checkpoint": str(checkpoint),
        "embeddings": np.concatenate(all_embeddings, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "paths": np.array(all_paths, dtype=object),
    }


def reduce_embeddings(embeddings: np.ndarray, method: str, args: argparse.Namespace) -> np.ndarray:
    if method == "tsne":
        perplexity = min(float(args.perplexity), max(5.0, (len(embeddings) - 1) / 3.0))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            metric="cosine",
            random_state=args.seed,
        )
        return reducer.fit_transform(embeddings)

    try:
        import umap
    except ImportError as exc:
        raise SystemExit("UMAP requested but umap-learn is not installed. Install with: pip install umap-learn") from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    return reducer.fit_transform(embeddings)


def plot_panels(run_outputs: list[dict], output: Path, args: argparse.Namespace) -> None:
    ncols = len(run_outputs)
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 3.1), squeeze=False)
    axes_list = axes[0]

    for ax, item in zip(axes_list, run_outputs):
        coords = item["coords"]
        labels = item["labels"]
        for cls_idx, emotion in enumerate(EMOTIONS):
            mask = labels == cls_idx
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=args.point_size,
                alpha=args.alpha,
                color=CLASS_COLORS[emotion],
                linewidths=0,
                label=emotion,
            )
        if not args.no_panel_titles:
            ax.set_title(item["label"], fontsize=10, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#555555")

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(EMOTIONS), 7),
        frameon=False,
        fontsize=8,
        markerscale=1.8,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(pad=0.2, rect=(0, 0.08, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=args.pad_inches)
    plt.close(fig)
    print(f"Saved embedding topology figure: {output}")


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    root = repo_root_from_script()
    args.run = [
        RunSpec(run.label, run.variant, resolve_path(run.checkpoint, root))
        for run in args.run
    ]
    output = resolve_path(Path(args.output), root)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    module = load_pipeline(root)
    extracted = [
        extract_embeddings(
            module,
            run,
            data_root=args.data_root,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        for run in args.run
    ]

    if args.max_samples > 0:
        for item in extracted:
            keep = class_balanced_indices(item["labels"], args.max_samples, args.seed)
            item["embeddings"] = item["embeddings"][keep]
            item["labels"] = item["labels"][keep]
            item["paths"] = item["paths"][keep] if len(item["paths"]) else item["paths"]

    for item in extracted:
        print(f"Reducing {item['label']} embeddings with {args.method}: {item['embeddings'].shape}")
        item["coords"] = reduce_embeddings(item["embeddings"], args.method, args)

    if args.save_embeddings:
        save_path = resolve_path(Path(args.save_embeddings), root)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for index, item in enumerate(extracted):
            prefix = f"run{index}_{item['label'].lower().replace(' ', '_').replace('-', '_')}"
            payload[f"{prefix}_embeddings"] = item["embeddings"]
            payload[f"{prefix}_coords"] = item["coords"]
            payload[f"{prefix}_labels"] = item["labels"]
            payload[f"{prefix}_paths"] = item["paths"]
        np.savez_compressed(save_path, **payload)
        print(f"Saved embeddings/cache: {save_path}")

    plot_panels(extracted, output, args)


if __name__ == "__main__":
    main()
