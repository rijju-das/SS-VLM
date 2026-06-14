#!/usr/bin/env python3
"""SS-VLM v2 experimental pipeline.

This file intentionally lives beside, rather than replaces,
``SS-VLM_Pipeline.py``. It keeps the old code/results untouched while adding:

- a true plain ViT baseline using the CLS token;
- reproducible seed controls;
- differential learning rates for the pretrained ViT backbone vs new heads;
- SupCon warmup and tunable loss/optimizer hyperparameters;
- less-saturated SFRA initialization with a learnable residual scale;
- optional OpenFace AU score fusion during RAG prediction.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import joblib
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_legacy_pipeline():
    legacy_path = Path(__file__).with_name("SS-VLM_Pipeline.py")
    spec = importlib.util.spec_from_file_location("legacy_ssvlm_pipeline", legacy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy_pipeline()

DATA_DIR = os.environ.get("SSVLM_DATA_DIR", "/home/rdas/RAF-DB")
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-4
BACKBONE_LR = 1e-5
HEAD_LR = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
NUM_CLASSES = 7
TEMP = 0.07
LAMBDA_CONT = 0.05
SUPCON_START_EPOCH = 6
EARLY_STOP_PATIENCE = 30
EARLY_STOP_MIN_DELTA = 0.0
TOP_K_RETRIEVAL = 9
K_PROTOTYPES = 5
KMEANS_SEED = 42
RAG_FUSION_MODE = "weighted"
RAG_FUSION_ALPHA = 0.25
RAG_SIM_TEMPERATURE = 0.10
AU_FUSION_BETA = 0.05
AU_SIM_TEMPERATURE = 0.35
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPENFACE_AU_COLUMNS = legacy.OPENFACE_AU_COLUMNS
FACS_PRIOR_MAPPING = legacy.FACS_PRIOR_MAPPING
EMOTIONS = legacy.EMOTIONS

AU_EMOTION_PROTOTYPES = {
    0: ["AU01_r", "AU02_r", "AU05_r", "AU26_r"],
    1: ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU07_r", "AU20_r", "AU25_r", "AU26_r"],
    2: ["AU09_r", "AU10_r"],
    3: ["AU06_r", "AU12_r"],
    4: ["AU01_r", "AU04_r", "AU15_r", "AU17_r"],
    5: ["AU04_r", "AU07_r", "AU23_r", "AU25_r"],
}


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_path(data_dir: str, split: str) -> str:
    return legacy._split_path(data_dir, split)


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    seed: int,
    num_workers: int,
    use_class_weights: bool,
):
    train_transform, eval_transform = build_transforms()
    train_dataset = datasets.ImageFolder(root=split_path(data_dir, "train"), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=split_path(data_dir, "test"), transform=eval_transform)

    if use_class_weights:
        targets = [sample[1] for sample in train_dataset.samples]
        weights = compute_class_weight("balanced", classes=np.unique(targets), y=targets)
        class_weights = torch.tensor(weights, dtype=torch.float, device=DEVICE)
    else:
        class_weights = None

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, class_weights


def get_eval_loader(data_dir: str, split: str, batch_size: int, num_workers: int):
    _, eval_transform = build_transforms()
    dataset = datasets.ImageFolder(root=split_path(data_dir, split), transform=eval_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features.shape[0], device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        return -mean_log_prob_pos.mean()


class SpectralCoordinateAttentionV2(nn.Module):
    """Less-saturated SFRA block with a learnable residual strength."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16, gamma_init: float = 0.1):
        super().__init__()
        mid_channels = max(8, in_channels // reduction_ratio)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, in_channels),
            nn.Sigmoid(),
        )
        self.conv_shared = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.mlp[0].weight, nonlinearity="linear")
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        for module in [self.conv_h, self.conv_w]:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
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
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)


class SSVLMExperimentModel(nn.Module):
    """One model class for true baseline, strong baseline, and SFRA-v2 variants."""

    def __init__(self, num_classes: int = 7, model_variant: str = "sfra_v2"):
        super().__init__()
        if model_variant not in {"plain_vit", "vit_gem", "sfra_v2", "sfra_legacy"}:
            raise ValueError(f"Unknown model_variant: {model_variant}")
        self.model_variant = model_variant
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.embed_dim = 768
        self.use_gem = model_variant in {"vit_gem", "sfra_v2", "sfra_legacy"}
        if model_variant == "sfra_v2":
            self.sfra = SpectralCoordinateAttentionV2(self.embed_dim)
        elif model_variant == "sfra_legacy":
            self.sfra = legacy.SpectralCoordinateAttention(self.embed_dim)
        else:
            self.sfra = None
        self.gem = GeMPooling(p=3)
        self.head_norm = nn.LayerNorm(self.embed_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 128),
        )
        self.head = nn.Linear(self.embed_dim, num_classes)

    def _embed(self, x):
        tokens = self.backbone.forward_features(x)
        if tokens.ndim == 2:
            embed = tokens
        elif self.model_variant == "plain_vit":
            embed = tokens[:, 0, :]
        else:
            if tokens.shape[1] == 197:
                tokens = tokens[:, 1:, :]
            b, n, c = tokens.shape
            h = w = int(n**0.5)
            fmap = tokens.permute(0, 2, 1).view(b, c, h, w)
            if self.sfra is not None:
                fmap = self.sfra(fmap)
            embed = self.gem(fmap).flatten(1)
        return self.head_norm(embed)

    def forward(self, x, mode: str = "train"):
        embed = self._embed(x)
        if mode == "extract":
            return F.normalize(embed, dim=1)
        logits = self.head(embed)
        if mode == "inference":
            return logits, F.normalize(embed, dim=1)
        proj = F.normalize(self.projector(embed), dim=1)
        return logits, proj


def make_optimizer(model, backbone_lr, head_lr, weight_decay):
    backbone_params = []
    new_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            new_params.append(param)
    return optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": new_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def train_pipeline_v2(
    save_path,
    data_dir,
    model_variant,
    epochs,
    batch_size,
    seed,
    num_workers,
    metrics_dir,
    early_stop_patience,
    early_stop_min_delta,
    lambda_cont,
    supcon_start_epoch,
    backbone_lr,
    head_lr,
    weight_decay,
    label_smoothing,
    use_class_weights,
    deterministic,
):
    set_seed(seed, deterministic=deterministic)
    train_loader, test_loader, class_weights = get_dataloaders(
        data_dir,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        use_class_weights=use_class_weights,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    run_name = os.path.splitext(os.path.basename(save_path))[0]
    history_csv = os.path.join(metrics_dir, f"{run_name}_train_history.csv")
    summary_json = os.path.join(metrics_dir, f"{run_name}_train_summary.json")
    fields = [
        "epoch",
        "train_loss",
        "train_ce_loss",
        "train_supcon_loss",
        "lambda_effective",
        "train_acc",
        "val_acc",
        "backbone_lr",
        "head_lr",
        "is_best",
    ]
    with open(history_csv, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    model = SSVLMExperimentModel(num_classes=NUM_CLASSES, model_variant=model_variant).to(DEVICE)
    optimizer = make_optimizer(model, backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    criterion_supcon = SupConLoss(temperature=TEMP)

    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_completed = 0
    early_stopped = False

    for epoch in range(epochs):
        epochs_completed = epoch + 1
        lambda_effective = lambda_cont if epochs_completed >= supcon_start_epoch else 0.0
        model.train()
        train_correct = train_total = 0
        train_loss_sum = train_ce_sum = train_supcon_sum = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epochs_completed}/{epochs}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits, proj_feats = model(imgs, mode="train")
            loss_ce = criterion_ce(logits, labels)
            if lambda_effective > 0:
                loss_con = criterion_supcon(proj_feats, labels)
            else:
                loss_con = torch.zeros((), device=DEVICE)
            loss = loss_ce + lambda_effective * loss_con
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss_sum += loss.item() * labels.size(0)
            train_ce_sum += loss_ce.item() * labels.size(0)
            train_supcon_sum += loss_con.item() * labels.size(0)
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{loss_ce.item():.3f}",
                con=f"{loss_con.item():.3f}",
                lam=f"{lambda_effective:.3f}",
            )

        backbone_lr_current = optimizer.param_groups[0]["lr"]
        head_lr_current = optimizer.param_groups[1]["lr"]
        scheduler.step()

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs, mode="inference")[0]
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        train_acc = train_correct / train_total
        is_best = val_acc > best_acc + early_stop_min_delta
        if is_best:
            best_acc = val_acc
            best_epoch = epochs_completed
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best accuracy: {best_acc:.4f} saved to {save_path}")
        else:
            epochs_without_improvement += 1

        with open(history_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(
                {
                    "epoch": epochs_completed,
                    "train_loss": train_loss_sum / train_total,
                    "train_ce_loss": train_ce_sum / train_total,
                    "train_supcon_loss": train_supcon_sum / train_total,
                    "lambda_effective": lambda_effective,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "backbone_lr": backbone_lr_current,
                    "head_lr": head_lr_current,
                    "is_best": int(is_best),
                }
            )

        print(
            f"Epoch {epochs_completed}: train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} best={best_acc:.4f}"
        )
        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            early_stopped = True
            print(
                f"Early stopping at epoch {epochs_completed}; "
                f"best={best_acc:.4f} at epoch {best_epoch}"
            )
            break

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": save_path,
                "data_dir": data_dir,
                "model_variant": model_variant,
                "seed": seed,
                "requested_epochs": epochs,
                "epochs_completed": epochs_completed,
                "best_epoch": best_epoch,
                "best_val_acc": best_acc,
                "early_stopped": early_stopped,
                "early_stop_patience": early_stop_patience,
                "early_stop_min_delta": early_stop_min_delta,
                "lambda_cont": lambda_cont,
                "supcon_start_epoch": supcon_start_epoch,
                "backbone_lr": backbone_lr,
                "head_lr": head_lr,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                "batch_size": batch_size,
                "use_class_weights": use_class_weights,
                "history_csv": history_csv,
            },
            f,
            indent=2,
        )

    print(f"Training completed. Best validation accuracy: {best_acc:.4f}")
    print(f"Training history saved to: {history_csv}")
    print(f"Training summary saved to: {summary_json}")
    return save_path


class RAGEngineV2(legacy.RAG_Engine):
    def __init__(
        self,
        model_path,
        model_variant,
        llm_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        openface_au_csv=None,
        top_k=TOP_K_RETRIEVAL,
        fusion_mode=RAG_FUSION_MODE,
        fusion_alpha=RAG_FUSION_ALPHA,
        retrieval_temperature=RAG_SIM_TEMPERATURE,
        k_prototypes=K_PROTOTYPES,
        kmeans_seed=KMEANS_SEED,
        load_llm=True,
        au_fusion_beta=AU_FUSION_BETA,
        au_temperature=AU_SIM_TEMPERATURE,
        au_model_path=None,
    ):
        self.device = DEVICE
        self.model_variant = model_variant
        self.model = SSVLMExperimentModel(num_classes=NUM_CLASSES, model_variant=model_variant).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.au_model_path = au_model_path
        self.learned_au_model = self._load_learned_au_model(au_model_path)
        self.bank_vectors = None
        self.bank_labels = None
        self.bank_au_summaries = None
        self.openface_aus = self._load_openface_au_csv(openface_au_csv)
        self.top_k = top_k
        self.fusion_mode = fusion_mode
        self.fusion_alpha = fusion_alpha
        self.retrieval_temperature = retrieval_temperature
        self.k_prototypes = k_prototypes
        self.kmeans_seed = kmeans_seed
        self.use_sfra = model_variant in {"sfra_v2", "sfra_legacy"}
        self.au_fusion_beta = au_fusion_beta
        self.au_temperature = au_temperature
        self.tokenizer = None
        self.llm = None
        if load_llm:
            print(f"Loading LLM: {llm_model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    def _load_learned_au_model(self, au_model_path):
        if not au_model_path:
            return None
        path = Path(au_model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Learned AU model not found: {path}")
        payload = joblib.load(path)
        if "model" not in payload or "feature_columns" not in payload:
            raise ValueError(
                "Expected a joblib payload from tools/train_au_emotion_tabular.py "
                "with keys: model, feature_columns."
            )
        print(f"Loaded learned AU model: {path}")
        print(f"  Model type: {payload.get('model_type', 'unknown')}")
        print(f"  Feature set: {payload.get('feature_set', 'unknown')}")
        return payload

    def _load_openface_au_csv(self, csv_path):
        if not csv_path:
            return {}
        if not os.path.exists(csv_path):
            print(f"OpenFace AU CSV not found: {csv_path}. Falling back to FACS priors.")
            return {}

        required_columns = set(OPENFACE_AU_COLUMNS)
        if self.learned_au_model is not None:
            required_columns.update(self.learned_au_model.get("feature_columns", []))
        required_columns = sorted(required_columns)

        au_rows = {}
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            missing = [col for col in required_columns if col not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(
                    f"OpenFace AU CSV is missing column(s) required by the AU model: {missing}"
                )
            for row in reader:
                cleaned = {str(k).strip(): str(v).strip() for k, v in row.items()}
                if cleaned.get("error", "") or cleaned.get("openface_success", "").lower() in {"0", "false"}:
                    continue
                vector = {}
                for col in required_columns:
                    try:
                        vector[col] = float(cleaned.get(col, "nan"))
                    except ValueError:
                        vector[col] = float("nan")
                if all(math.isnan(value) for value in vector.values()):
                    continue
                for key_field in ("image_path", "relative_path", "path"):
                    if cleaned.get(key_field):
                        for key in legacy._path_keys(cleaned[key_field]):
                            au_rows[key] = vector

        print(f"Loaded OpenFace AU estimates for {len(au_rows)} path keys from {csv_path}")
        print(f"  AU columns loaded: {len(required_columns)}")
        return au_rows

    @staticmethod
    def _temperature_scale_probs(probs, temperature):
        probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
        logits = np.log(probs) / max(float(temperature), 1e-8)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)

    def _learned_au_distribution(self, vector):
        payload = self.learned_au_model
        if payload is None:
            return None
        values = []
        for col in payload["feature_columns"]:
            raw = vector.get(col, float("nan"))
            if not np.isfinite(raw):
                return None
            values.append(float(raw))
        probs = payload["model"].predict_proba(np.asarray(values, dtype=np.float32).reshape(1, -1))
        temperature = float(payload.get("temperature", 1.0))
        probs = self._temperature_scale_probs(probs, temperature=temperature)[0]
        return probs / max(float(probs.sum()), 1e-12)

    def _au_distribution(self, image_path):
        if self.au_fusion_beta <= 0:
            return None
        vector = self._lookup_openface_au(image_path)
        if vector is None:
            return None
        learned_dist = self._learned_au_distribution(vector)
        if learned_dist is not None:
            return learned_dist

        def au_value(col):
            value = vector.get(col, float("nan"))
            if not np.isfinite(value):
                return 0.0
            return max(0.0, min(5.0, float(value)))

        scores = np.zeros(NUM_CLASSES, dtype=float)
        all_signal_cols = sorted({col for cols in AU_EMOTION_PROTOTYPES.values() for col in cols})
        global_signal = np.mean([au_value(col) for col in all_signal_cols]) / 5.0
        for cls_idx, cols in AU_EMOTION_PROTOTYPES.items():
            scores[cls_idx] = np.mean([au_value(col) for col in cols]) / 5.0
        scores[6] = max(0.0, 1.0 - global_signal)
        if np.all(scores <= 1e-8):
            return None
        scaled = scores / max(float(self.au_temperature), 1e-6)
        scaled = scaled - np.max(scaled)
        dist = np.exp(scaled)
        dist = dist / max(dist.sum(), 1e-12)
        return dist

    def _au_source(self):
        if self.au_fusion_beta <= 0:
            return "disabled"
        if self.learned_au_model is not None:
            return f"learned_{self.learned_au_model.get('model_type', 'tabular')}"
        return "manual_facs_prior"

    def _fuse_prediction(self, classifier_probs, retrieval_dist, au_dist=None):
        classifier_probs = np.array(classifier_probs, dtype=float)
        retrieval_dist = np.array(retrieval_dist, dtype=float)
        classifier_pred = int(np.argmax(classifier_probs))
        retrieval_pred = int(np.argmax(retrieval_dist))
        au_pred = -1
        au_conf = 0.0

        if self.fusion_mode == "none":
            base_probs = classifier_probs
            source = "classifier"
        elif self.fusion_mode == "majority":
            base_probs = retrieval_dist
            source = "retrieval"
        elif self.fusion_mode == "adaptive":
            classifier_conf = float(classifier_probs[classifier_pred])
            uncertainty = 1.0 - classifier_conf
            alpha = min(0.75, max(0.0, self.fusion_alpha + uncertainty * 0.5))
            base_probs = (1.0 - alpha) * classifier_probs + alpha * retrieval_dist
            source = f"adaptive_alpha={alpha:.3f}"
        else:
            alpha = min(1.0, max(0.0, self.fusion_alpha))
            base_probs = (1.0 - alpha) * classifier_probs + alpha * retrieval_dist
            source = f"weighted_alpha={alpha:.3f}"

        if au_dist is not None and self.au_fusion_beta > 0:
            au_dist = np.array(au_dist, dtype=float)
            beta = min(0.5, max(0.0, self.au_fusion_beta))
            base_probs = (1.0 - beta) * base_probs + beta * au_dist
            au_pred = int(np.argmax(au_dist))
            au_conf = float(au_dist[au_pred])
            source = f"{source}+au_beta={beta:.3f}"

        final_probs = base_probs / max(base_probs.sum(), 1e-12)
        final_pred = int(np.argmax(final_probs))
        return {
            "classifier_pred": classifier_pred,
            "classifier_conf": float(classifier_probs[classifier_pred]),
            "retrieval_pred": retrieval_pred,
            "retrieval_conf": float(retrieval_dist[retrieval_pred]),
            "au_pred": au_pred,
            "au_conf": au_conf,
            "final_pred": final_pred,
            "final_conf": float(final_probs[final_pred]),
            "source": source,
            "final_probs": final_probs,
        }

    def run_inference(
        self,
        test_loader,
        num_samples_to_show=5,
        metrics_dir="outputs_v2/metrics",
        run_name="ss_vlm_v2",
        report_sampling="first",
        reports_per_class=1,
    ):
        if self.bank_vectors is None:
            raise ValueError("Memory bank not built. Call build_bank() first.")

        k_limit = max(1, min(int(self.top_k), len(self.bank_labels)))
        print(
            f"Running RAG inference: k={k_limit}, fusion={self.fusion_mode}, "
            f"alpha={self.fusion_alpha}, au_beta={self.au_fusion_beta}"
        )
        all_final_preds, all_classifier_preds, all_labels, all_reports = [], [], [], []
        prediction_rows = []
        k_stats = {
            k: {
                "classifier_correct": 0,
                "retrieval_correct": 0,
                "fused_correct": 0,
                "au_correct": 0,
                "au_available": 0,
                "retrieval_consistency_sum": 0.0,
            }
            for k in range(1, k_limit + 1)
        }
        final_correct = classifier_correct = retrieval_correct = 0
        au_correct = au_available = 0
        total = 0
        consistency_sum = 0.0
        dataset_samples = getattr(test_loader.dataset, "samples", None)
        sample_offset = 0
        report_counts_by_class = {class_idx: 0 for class_idx in range(NUM_CLASSES)}
        max_reports_total = max(0, int(num_samples_to_show))
        reports_per_class = max(0, int(reports_per_class))

        def should_generate_report(batch_idx, j, gt_lbl):
            if self.tokenizer is None or self.llm is None:
                return False
            if report_sampling == "class_balanced":
                return reports_per_class > 0 and report_counts_by_class[gt_lbl] < reports_per_class
            return batch_idx == 0 and j < max_reports_total

        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(tqdm(test_loader, desc="RAG Inference")):
                imgs = imgs.to(self.device)
                labels_np = labels.numpy()
                batch_start = sample_offset
                batch_size = len(labels_np)
                if dataset_samples is not None:
                    batch_paths = [dataset_samples[i][0] for i in range(batch_start, batch_start + batch_size)]
                else:
                    batch_paths = [""] * batch_size
                sample_offset += batch_size

                logits, embeddings = self.model(imgs, mode="inference")
                probs_np = F.softmax(logits, dim=1).cpu().numpy()
                embeddings_np = embeddings.cpu().numpy()
                total += batch_size

                for j in range(batch_size):
                    sample_index = batch_start + j
                    gt_lbl = int(labels_np[j])
                    query_feat = embeddings_np[j : j + 1]
                    classifier_probs = probs_np[j]
                    image_path = batch_paths[j]
                    sims = cosine_similarity(query_feat, self.bank_vectors)[0]
                    sorted_idx = np.argsort(sims)[::-1][:k_limit]
                    sorted_sims = sims[sorted_idx]
                    au_dist = self._au_distribution(image_path)

                    for k in range(1, k_limit + 1):
                        k_indices = sorted_idx[:k]
                        k_sims = sorted_sims[:k]
                        k_dist = self._retrieval_distribution(k_indices, k_sims)
                        k_decision = self._fuse_prediction(classifier_probs, k_dist, au_dist=au_dist)
                        k_ret_pred = int(np.argmax(k_dist))
                        k_stats[k]["classifier_correct"] += int(k_decision["classifier_pred"] == gt_lbl)
                        k_stats[k]["retrieval_correct"] += int(k_ret_pred == gt_lbl)
                        k_stats[k]["fused_correct"] += int(k_decision["final_pred"] == gt_lbl)
                        if au_dist is not None:
                            k_stats[k]["au_available"] += 1
                            k_stats[k]["au_correct"] += int(int(np.argmax(au_dist)) == gt_lbl)
                        k_stats[k]["retrieval_consistency_sum"] += (
                            np.sum(self.bank_labels[k_indices] == k_decision["final_pred"]) / k
                        )

                    top_k_idx = sorted_idx[:k_limit]
                    top_k_sims = sorted_sims[:k_limit]
                    ret_labels = self.bank_labels[top_k_idx]
                    retrieval_dist = self._retrieval_distribution(top_k_idx, top_k_sims)
                    decision = self._fuse_prediction(classifier_probs, retrieval_dist, au_dist=au_dist)
                    consistency = np.sum(ret_labels == decision["final_pred"]) / k_limit

                    final_correct += int(decision["final_pred"] == gt_lbl)
                    classifier_correct += int(decision["classifier_pred"] == gt_lbl)
                    retrieval_correct += int(decision["retrieval_pred"] == gt_lbl)
                    if au_dist is not None:
                        au_available += 1
                        au_correct += int(decision["au_pred"] == gt_lbl)
                    consistency_sum += consistency
                    all_final_preds.append(int(decision["final_pred"]))
                    all_classifier_preds.append(int(decision["classifier_pred"]))
                    all_labels.append(gt_lbl)
                    prediction_rows.append(
                        {
                            "sample_index": sample_index,
                            "image_path": image_path,
                            "gt_label": gt_lbl,
                            "gt_emotion": EMOTIONS[gt_lbl],
                            "pred_label": int(decision["final_pred"]),
                            "pred_emotion": EMOTIONS[int(decision["final_pred"])],
                            "confidence": float(decision["final_conf"]),
                            "correct": int(decision["final_pred"] == gt_lbl),
                            "classifier_pred_label": int(decision["classifier_pred"]),
                            "classifier_pred_emotion": EMOTIONS[int(decision["classifier_pred"])],
                            "classifier_confidence": float(decision["classifier_conf"]),
                            "classifier_correct": int(decision["classifier_pred"] == gt_lbl),
                            "retrieval_pred_label": int(decision["retrieval_pred"]),
                            "retrieval_pred_emotion": EMOTIONS[int(decision["retrieval_pred"])],
                            "retrieval_confidence": float(decision["retrieval_conf"]),
                            "retrieval_correct": int(decision["retrieval_pred"] == gt_lbl),
                            "au_pred_label": int(decision["au_pred"]),
                            "au_pred_emotion": EMOTIONS[int(decision["au_pred"])] if decision["au_pred"] >= 0 else "",
                            "au_confidence": float(decision["au_conf"]),
                            "au_available": int(au_dist is not None),
                            "au_source": self._au_source(),
                            "fusion_source": decision["source"],
                            "retrieval_consistency": float(consistency),
                            "top_k_indices": ";".join(str(int(i)) for i in top_k_idx),
                            "top_k_labels": ";".join(str(int(i)) for i in ret_labels),
                            "top_k_emotions": ";".join(EMOTIONS[int(i)] for i in ret_labels),
                            "top_k_similarities": ";".join(f"{float(s):.6f}" for s in top_k_sims),
                            "retrieval_distribution": ";".join(f"{float(v):.6f}" for v in retrieval_dist),
                            "au_distribution": ";".join(f"{float(v):.6f}" for v in au_dist) if au_dist is not None else "",
                            "fused_distribution": ";".join(f"{float(v):.6f}" for v in decision["final_probs"]),
                        }
                    )

                    if should_generate_report(batch_idx, j, gt_lbl):
                        report = self.generate_report(
                            decision["final_pred"],
                            decision["final_conf"],
                            top_k_idx,
                            classifier_idx=decision["classifier_pred"],
                            retrieval_idx=decision["retrieval_pred"],
                        )
                        all_reports.append(
                            {
                                "sample_index": sample_index,
                                "image_path": image_path,
                                "gt": EMOTIONS[gt_lbl],
                                "pred": EMOTIONS[int(decision["final_pred"])],
                                "classifier_pred": EMOTIONS[int(decision["classifier_pred"])],
                                "retrieval_pred": EMOTIONS[int(decision["retrieval_pred"])],
                                "au_pred": EMOTIONS[int(decision["au_pred"])] if decision["au_pred"] >= 0 else "",
                                "au_source": self._au_source(),
                                "confidence": float(decision["final_conf"]),
                                "consistency": float(consistency),
                                "retrieved_evidence": self._format_evidence(top_k_idx),
                                "report": report,
                            }
                        )
                        report_counts_by_class[gt_lbl] += 1

        accuracy = final_correct / total
        classifier_accuracy = classifier_correct / total
        retrieval_accuracy = retrieval_correct / total
        au_accuracy = au_correct / au_available if au_available else None
        avg_consistency = consistency_sum / total
        print("RAG inference results:")
        print(f"  Classifier Accuracy: {classifier_accuracy:.4f} ({classifier_correct}/{total})")
        print(f"  Retrieval Accuracy:  {retrieval_accuracy:.4f} ({retrieval_correct}/{total})")
        print(f"  RAG-Fused Accuracy:  {accuracy:.4f} ({final_correct}/{total})")
        if au_accuracy is not None:
            print(f"  AU-only Accuracy:    {au_accuracy:.4f} ({au_correct}/{au_available} AU-available samples)")
        print(f"  Avg Retrieval Consistency: {avg_consistency:.4f}")

        os.makedirs(metrics_dir, exist_ok=True)
        predictions_csv = os.path.join(metrics_dir, f"{run_name}_rag_predictions.csv")
        summary_json = os.path.join(metrics_dir, f"{run_name}_rag_summary.json")
        reports_json = os.path.join(metrics_dir, f"{run_name}_rag_sample_reports.json")
        k_sensitivity_csv = os.path.join(metrics_dir, f"{run_name}_k_sensitivity.csv")

        if prediction_rows:
            with open(predictions_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
                writer.writeheader()
                writer.writerows(prediction_rows)

        with open(k_sensitivity_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "k",
                    "classifier_accuracy",
                    "retrieval_accuracy",
                    "rag_fused_accuracy",
                    "au_accuracy_on_available",
                    "au_available",
                    "avg_retrieval_consistency",
                ],
            )
            writer.writeheader()
            for k, stats in k_stats.items():
                writer.writerow(
                    {
                        "k": k,
                        "classifier_accuracy": stats["classifier_correct"] / total,
                        "retrieval_accuracy": stats["retrieval_correct"] / total,
                        "rag_fused_accuracy": stats["fused_correct"] / total,
                        "au_accuracy_on_available": (
                            stats["au_correct"] / stats["au_available"] if stats["au_available"] else ""
                        ),
                        "au_available": stats["au_available"],
                        "avg_retrieval_consistency": stats["retrieval_consistency_sum"] / total,
                    }
                )

        cm = confusion_matrix(all_labels, all_final_preds, labels=list(range(NUM_CLASSES))).tolist()
        classifier_cm = confusion_matrix(all_labels, all_classifier_preds, labels=list(range(NUM_CLASSES))).tolist()
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": float(accuracy),
                    "correct": int(final_correct),
                    "classifier_accuracy": float(classifier_accuracy),
                    "classifier_correct": int(classifier_correct),
                    "retrieval_accuracy": float(retrieval_accuracy),
                    "retrieval_correct": int(retrieval_correct),
                    "au_accuracy_on_available": float(au_accuracy) if au_accuracy is not None else None,
                    "au_available": int(au_available),
                    "total": int(total),
                    "model_variant": self.model_variant,
                    "top_k_retrieval": int(k_limit),
                    "fusion_mode": self.fusion_mode,
                    "fusion_alpha": float(self.fusion_alpha),
                    "au_fusion_beta": float(self.au_fusion_beta),
                    "au_temperature": float(self.au_temperature),
                    "au_source": self._au_source(),
                    "au_model_path": str(self.au_model_path) if self.au_model_path else "",
                    "retrieval_temperature": float(self.retrieval_temperature),
                    "k_prototypes": int(self.k_prototypes),
                    "kmeans_seed": int(self.kmeans_seed),
                    "avg_retrieval_consistency": float(avg_consistency),
                    "confusion_matrix": cm,
                    "classifier_confusion_matrix": classifier_cm,
                    "classification_report": classification_report(
                        all_labels,
                        all_final_preds,
                        labels=list(range(NUM_CLASSES)),
                        target_names=EMOTIONS,
                        output_dict=True,
                        zero_division=0,
                    ),
                    "classifier_classification_report": classification_report(
                        all_labels,
                        all_classifier_preds,
                        labels=list(range(NUM_CLASSES)),
                        target_names=EMOTIONS,
                        output_dict=True,
                        zero_division=0,
                    ),
                    "predictions_csv": predictions_csv,
                    "k_sensitivity_csv": k_sensitivity_csv,
                },
                f,
                indent=2,
            )

        with open(reports_json, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2)

        print(f"RAG predictions saved to: {predictions_csv}")
        print(f"RAG summary saved to: {summary_json}")
        print(f"k-sensitivity saved to: {k_sensitivity_csv}")
        print(f"Sample reports saved to: {reports_json}")
        return accuracy, avg_consistency


def main():
    parser = argparse.ArgumentParser(description="SS-VLM v2 experimental pipeline")
    parser.add_argument("--mode", default="full", choices=["train", "rag", "full"])
    parser.add_argument("--model_path", default="outputs_v2/models/ssvlm_v2.pth")
    parser.add_argument("--data_dir", default=os.environ.get("SSVLM_DATA_DIR", DATA_DIR))
    parser.add_argument("--model_variant", default="sfra_v2", choices=["plain_vit", "vit_gem", "sfra_v2", "sfra_legacy"])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    parser.add_argument("--lambda_cont", type=float, default=LAMBDA_CONT)
    parser.add_argument("--supcon_start_epoch", type=int, default=SUPCON_START_EPOCH)
    parser.add_argument("--backbone_lr", type=float, default=BACKBONE_LR)
    parser.add_argument("--head_lr", type=float, default=HEAD_LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--metrics_dir", default="outputs_v2/metrics")
    parser.add_argument("--openface_au_csv", default=os.environ.get("SSVLM_OPENFACE_AU_CSV", ""))
    parser.add_argument(
        "--au_model_path",
        default="",
        help="Optional learned AU-to-emotion joblib model from tools/train_au_emotion_tabular.py.",
    )
    parser.add_argument("--top_k", type=int, default=TOP_K_RETRIEVAL)
    parser.add_argument("--rag_fusion", default=RAG_FUSION_MODE, choices=["none", "weighted", "majority", "adaptive"])
    parser.add_argument("--rag_alpha", type=float, default=RAG_FUSION_ALPHA)
    parser.add_argument("--retrieval_temperature", type=float, default=RAG_SIM_TEMPERATURE)
    parser.add_argument("--au_fusion_beta", type=float, default=AU_FUSION_BETA)
    parser.add_argument("--au_temperature", type=float, default=AU_SIM_TEMPERATURE)
    parser.add_argument("--k_prototypes", type=int, default=K_PROTOTYPES)
    parser.add_argument("--kmeans_seed", type=int, default=KMEANS_SEED)
    parser.add_argument("--num_reports", type=int, default=5)
    parser.add_argument("--report_sampling", default="first", choices=["first", "class_balanced"])
    parser.add_argument("--reports_per_class", type=int, default=1)
    parser.add_argument("--skip_llm_reports", action="store_true")
    args = parser.parse_args()

    if args.mode in {"train", "full"} and not args.skip_train:
        model_path = train_pipeline_v2(
            save_path=args.model_path,
            data_dir=args.data_dir,
            model_variant=args.model_variant,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            metrics_dir=args.metrics_dir,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            lambda_cont=args.lambda_cont,
            supcon_start_epoch=args.supcon_start_epoch,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            use_class_weights=not args.no_class_weights,
            deterministic=args.deterministic,
        )
    else:
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return
        print(f"Using existing model: {model_path}")

    if args.mode in {"rag", "full"}:
        load_llm = (
            not args.skip_llm_reports
            and (
                args.num_reports > 0
                or (args.report_sampling == "class_balanced" and args.reports_per_class > 0)
            )
        )
        rag = RAGEngineV2(
            model_path=model_path,
            model_variant=args.model_variant,
            openface_au_csv=args.openface_au_csv,
            top_k=args.top_k,
            fusion_mode=args.rag_fusion,
            fusion_alpha=args.rag_alpha,
            retrieval_temperature=args.retrieval_temperature,
            k_prototypes=args.k_prototypes,
            kmeans_seed=args.kmeans_seed,
            load_llm=load_llm,
            au_fusion_beta=args.au_fusion_beta,
            au_temperature=args.au_temperature,
            au_model_path=args.au_model_path,
        )
        train_eval_loader = get_eval_loader(args.data_dir, "train", args.batch_size, args.num_workers)
        test_loader = get_eval_loader(args.data_dir, "test", args.batch_size, args.num_workers)
        rag.build_bank(train_eval_loader)
        run_name = os.path.splitext(os.path.basename(model_path))[0]
        accuracy, consistency = rag.run_inference(
            test_loader,
            num_samples_to_show=args.num_reports if load_llm else 0,
            metrics_dir=args.metrics_dir,
            run_name=run_name,
            report_sampling=args.report_sampling,
            reports_per_class=args.reports_per_class,
        )
        print("=" * 60)
        print("SS-VLM v2 pipeline completed successfully")
        print(f"Final RAG-fused accuracy: {accuracy:.4f}")
        print(f"Retrieval consistency: {consistency:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
