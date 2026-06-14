#!/usr/bin/env python3
"""Generate the v2 RAF-DB Grad-CAM panel for manuscript Figure 8.

The figure compares the current v2 Plain ViT, ViT+GeM, and SFRA-RAG visual
encoder checkpoints on deterministic RAF-DB test examples.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

cache_root = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


EMOTIONS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
DEFAULT_CLASSES = ["Happiness", "Surprise", "Fear"]
DEFAULT_FIXED_EXAMPLES = {"Fear": "test_0377_aligned.jpg"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SpectralCoordinateAttentionV2(nn.Module):
    """Less-saturated SFRA block with a learnable residual strength."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16, gamma_init: float = 0.1):
        super().__init__()
        mid_channels = max(8, in_channels // reduction_ratio)
        self.avg_pool = nn.AvgPool2d(3, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, in_channels),
            nn.Sigmoid(),
        )
        self.conv_shared = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )
        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.mlp[0].weight, nonlinearity="linear")
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        for module in [self.conv_h, self.conv_w]:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.size()
        low = self.avg_pool(x)
        high = x - low
        w_spec = self.mlp(self.gap(high).view(b, c)).view(b, c, 1, 1)

        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w))
        cat = torch.cat([x_h, x_w.permute(0, 1, 3, 2)], dim=2)
        fused = self.conv_shared(cat)
        f_h, f_w = torch.split(fused, [h, w], dim=2)
        a_h = self.sigmoid(self.conv_h(f_h))
        a_w = self.sigmoid(self.conv_w(f_w.permute(0, 1, 3, 2)))
        gate = w_spec * a_h * a_w
        return identity + self.gamma * identity * gate


class GeMPooling(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1)),
        ).pow(1.0 / self.p)


class SSVLMExperimentModel(nn.Module):
    def __init__(self, num_classes: int = 7, model_variant: str = "sfra_v2"):
        super().__init__()
        if model_variant not in {"plain_vit", "vit_gem", "sfra_v2"}:
            raise ValueError(f"Unsupported model_variant for this plot: {model_variant}")
        # The checkpoint contains trained backbone weights, so this avoids
        # internet/cache dependence from timm's pretrained loader.
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        self.embed_dim = 768
        self.model_variant = model_variant
        self.use_gem = model_variant in {"vit_gem", "sfra_v2"}
        self.sfra = SpectralCoordinateAttentionV2(self.embed_dim) if model_variant == "sfra_v2" else None
        self.gem = GeMPooling(p=3)
        self.head_norm = nn.LayerNorm(self.embed_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 128),
        )
        self.head = nn.Linear(self.embed_dim, num_classes)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)
        if tokens.ndim == 2:
            embed = tokens
        elif not self.use_gem:
            embed = tokens[:, 0, :]
        else:
            if tokens.shape[1] == 197:
                tokens = tokens[:, 1:, :]
            b, n, c = tokens.shape
            h = w = int(math.sqrt(n))
            if h * w != n:
                raise RuntimeError(f"Cannot reshape {n} ViT tokens into a square feature map")
            fmap = tokens.permute(0, 2, 1).reshape(b, c, h, w)
            if self.sfra is not None:
                fmap = self.sfra(fmap)
            embed = self.gem(fmap).flatten(1)
        return self.head_norm(embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._embed(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot current v2 RAF-DB Grad-CAM Figure 8.")
    parser.add_argument("--data-root", required=True, help="RAF-DB root containing train/test folders.")
    parser.add_argument(
        "--plain-checkpoint",
        default="outputs_v2/models/sweep/plain_vit_ce_seed42.pth",
        help="Current v2 Plain ViT checkpoint. Default uses seed 42 for same-seed Figure 8 comparison.",
    )
    parser.add_argument(
        "--vit-gem-checkpoint",
        default="outputs_v2/models/sweep/vit_gem_ce_seed42.pth",
        help="Current v2 ViT+GeM checkpoint. Default uses seed 42 for same-seed Figure 8 comparison.",
    )
    parser.add_argument(
        "--sfra-checkpoint",
        default="outputs_v2/models/seed_sweep/sfra_v2_l005_seed42.pth",
        help="Current v2 SFRA checkpoint, preferably the best RAG-fused run.",
    )
    parser.add_argument(
        "--output",
        action="append",
        default=None,
        help="Output PDF path. Repeat to save both repo and manuscript copies.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        choices=EMOTIONS,
        help="Emotion rows to include.",
    )
    parser.add_argument("--split", default="test", help="RAF-DB split to sample from.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-candidates", type=int, default=600)
    parser.add_argument("--allow-fallback", action="store_true", help="Use first class image if no jointly-correct example is found.")
    parser.add_argument(
        "--fixed-example",
        action="append",
        default=[],
        metavar="EMOTION=FILENAME",
        help=(
            "Force a row to use a specific image basename, e.g. Fear=test_0377_aligned.jpg. "
            "Repeat as needed. By default, Fear keeps the previous manuscript image."
        ),
    )
    parser.add_argument(
        "--no-default-fixed-examples",
        action="store_true",
        help="Disable built-in fixed examples such as Fear=test_0377_aligned.jpg.",
    )
    parser.add_argument("--pad-inches", type=float, default=0.0)
    return parser.parse_args()


def split_path(data_root: Path, split: str) -> Path:
    candidates = [
        data_root / split,
        data_root / "DATASET" / split,
        data_root / "original" / split,
        data_root / "DATASET" / "original" / split,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find split '{split}' under {data_root}")


def load_model(checkpoint: Path, model_variant: str, device: torch.device) -> SSVLMExperimentModel:
    model = SSVLMExperimentModel(num_classes=len(EMOTIONS), model_variant=model_variant).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def class_to_dataset_idx(dataset: datasets.ImageFolder, emotion: str) -> int:
    emotion_idx = EMOTIONS.index(emotion)
    for candidate in [str(emotion_idx + 1), emotion, emotion.lower()]:
        if candidate in dataset.class_to_idx:
            return int(dataset.class_to_idx[candidate])
    return emotion_idx


def predict(model: SSVLMExperimentModel, tensor: torch.Tensor, device: torch.device) -> int:
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
    return int(torch.argmax(logits, dim=1).item())


def parse_fixed_examples(args: argparse.Namespace) -> dict[str, str]:
    fixed = {} if args.no_default_fixed_examples else dict(DEFAULT_FIXED_EXAMPLES)
    for item in args.fixed_example:
        if "=" not in item:
            raise ValueError(f"Invalid --fixed-example '{item}'. Expected EMOTION=FILENAME.")
        emotion, filename = item.split("=", 1)
        emotion = emotion.strip()
        filename = filename.strip()
        if emotion not in EMOTIONS:
            raise ValueError(f"Invalid fixed-example emotion '{emotion}'. Choices: {', '.join(EMOTIONS)}")
        if not filename:
            raise ValueError(f"Invalid --fixed-example '{item}': filename is empty.")
        fixed[emotion] = filename
    return fixed


def select_examples(
    dataset: datasets.ImageFolder,
    models: list[SSVLMExperimentModel],
    classes: list[str],
    device: torch.device,
    max_candidates: int,
    allow_fallback: bool,
    fixed_examples: dict[str, str],
) -> list[tuple[str, Path, torch.Tensor, int]]:
    selected = []
    for emotion in classes:
        target_idx = class_to_dataset_idx(dataset, emotion)
        candidate_indices = [
            idx for idx, (_, label) in enumerate(dataset.samples) if int(label) == target_idx
        ][:max_candidates]
        if not candidate_indices:
            raise RuntimeError(f"No candidate images found for class '{emotion}'")

        fixed_filename = fixed_examples.get(emotion)
        if fixed_filename:
            fixed_matches = [
                idx
                for idx, (path, label) in enumerate(dataset.samples)
                if int(label) == target_idx and Path(path).name == fixed_filename
            ]
            if not fixed_matches:
                raise RuntimeError(
                    f"Fixed example for {emotion} was not found in {dataset.root}: {fixed_filename}"
                )
            idx = fixed_matches[0]
            tensor, label = dataset[idx]
            selected.append((emotion, Path(dataset.samples[idx][0]), tensor, int(label)))
            continue

        fallback = None
        chosen = None
        for idx in candidate_indices:
            tensor, label = dataset[idx]
            path = Path(dataset.samples[idx][0])
            if fallback is None:
                fallback = (emotion, path, tensor, int(label))
            predictions = [predict(model, tensor, device) for model in models]
            if all(pred == int(label) for pred in predictions):
                chosen = (emotion, path, tensor, int(label))
                break

        if chosen is None:
            if not allow_fallback:
                raise RuntimeError(
                    f"No jointly-correct {emotion} image found in first {len(candidate_indices)} candidates. "
                    "Re-run with --allow-fallback or increase --max-candidates."
                )
            chosen = fallback
        selected.append(chosen)
    return selected


def grad_cam(
    model: SSVLMExperimentModel,
    tensor: torch.Tensor,
    target_label: int,
    device: torch.device,
) -> np.ndarray:
    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output

    def backward_hook(_module, _grad_inputs, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Hooking the final transformer block before attention gives non-zero
    # patch-token gradients for both CLS-token Plain ViT and SFRA-GeM variants.
    hook_module = model.backbone.blocks[-1].norm1
    forward_handle = hook_module.register_forward_hook(forward_hook)
    backward_handle = hook_module.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    try:
        image = tensor.unsqueeze(0).to(device)
        logits = model(image)
        score = logits[0, target_label]
        score.backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

    patch_activations = activations.detach()[0, 1:, :]
    patch_gradients = gradients.detach()[0, 1:, :]
    weights = patch_gradients.mean(dim=0)
    cam = F.relu((patch_activations * weights).sum(dim=1))
    side = int(math.sqrt(cam.numel()))
    if side * side != cam.numel():
        raise RuntimeError(f"Cannot reshape {cam.numel()} patch scores into a square CAM")
    cam = cam.reshape(side, side)
    cam = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy()


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    image = (tensor.cpu() * STD + MEAN).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    heat = plt.get_cmap("turbo")(cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)


def plot_panel(
    examples: list[tuple[str, Path, torch.Tensor, int]],
    plain_model: SSVLMExperimentModel,
    vit_gem_model: SSVLMExperimentModel,
    sfra_model: SSVLMExperimentModel,
    outputs: list[Path],
    device: torch.device,
    pad_inches: float,
) -> None:
    rows = len(examples)
    fig, axes = plt.subplots(rows, 4, figsize=(7.3, 1.78 * rows), squeeze=False)
    column_titles = [
        "Input image",
        "Plain ViT Grad-CAM",
        "ViT+GeM Grad-CAM",
        "SFRA-RAG Grad-CAM",
    ]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold", pad=5)

    for row, (emotion, path, tensor, label) in enumerate(examples):
        rgb = tensor_to_rgb(tensor)
        plain_cam = grad_cam(plain_model, tensor, label, device)
        vit_gem_cam = grad_cam(vit_gem_model, tensor, label, device)
        sfra_cam = grad_cam(sfra_model, tensor, label, device)

        panels = [
            rgb,
            overlay_cam(rgb, plain_cam),
            overlay_cam(rgb, vit_gem_cam),
            overlay_cam(rgb, sfra_cam),
        ]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].text(
                    0.02,
                    0.98,
                    emotion,
                    transform=axes[row, col].transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
                )
                axes[row, col].text(
                    0.02,
                    0.03,
                    path.name,
                    transform=axes[row, col].transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=5.5,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
                )

    fig.tight_layout(pad=0.08, w_pad=0.35, h_pad=0.35)
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=pad_inches)
        print(f"Saved Figure 8 v2 Grad-CAM panel: {output}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN.flatten().tolist(), std=STD.flatten().tolist()),
        ]
    )
    dataset = datasets.ImageFolder(split_path(Path(args.data_root), args.split), transform=transform)
    plain_model = load_model(Path(args.plain_checkpoint), model_variant="plain_vit", device=device)
    vit_gem_model = load_model(Path(args.vit_gem_checkpoint), model_variant="vit_gem", device=device)
    sfra_model = load_model(Path(args.sfra_checkpoint), model_variant="sfra_v2", device=device)
    fixed_examples = parse_fixed_examples(args)
    examples = select_examples(
        dataset=dataset,
        models=[plain_model, vit_gem_model, sfra_model],
        classes=args.classes,
        device=device,
        max_candidates=args.max_candidates,
        allow_fallback=args.allow_fallback,
        fixed_examples=fixed_examples,
    )

    outputs = [Path(p) for p in (args.output or ["outputs_v2/figures/heatmap_v2_rafdb.pdf"])]
    plot_panel(examples, plain_model, vit_gem_model, sfra_model, outputs, device, args.pad_inches)


if __name__ == "__main__":
    main()
