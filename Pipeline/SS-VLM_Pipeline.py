import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import csv
import json
import math
import re
import timm
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import random
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 0. 配置 (Configuration)
# ==========================================
DATA_DIR = os.environ.get('SSVLM_DATA_DIR', '/home/rdas/RAF-DB')
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMP = 0.07         
LAMBDA_CONT = 0.1
EARLY_STOP_PATIENCE = 30
EARLY_STOP_MIN_DELTA = 0.0

# RAG Configuration
TOP_K_RETRIEVAL = 9  # Paper final selection: k=9 for optimal redundancy
K_PROTOTYPES = 5
KMEANS_SEED = 42
RAG_FUSION_MODE = "weighted"
RAG_FUSION_ALPHA = 0.25
RAG_SIM_TEMPERATURE = 0.10
WEIGHT_DECAY = 0.05  # Paper: AdamW with weight decay 0.05   
AU_ACTIVE_THRESHOLD = 1.0
OPENFACE_AU_COLUMNS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]
VALID_AU_IDS = {col.replace("_r", "") for col in OPENFACE_AU_COLUMNS}
AU_MENTION_RE = re.compile(r"\bAU\s*[_-]?\s*0?(\d{1,3})(?:_[rc])?\b", flags=re.I)

FACS_PRIOR_MAPPING = {
    0: "AU1+AU2+AU5+AU26 (canonical Surprise prior: brow raise, upper lid raise, jaw drop)",
    1: "AU1+AU2+AU4+AU5+AU7+AU20+AU25/AU26 (canonical Fear prior; mouth opening varies)",
    2: "AU9+AU10 (+ optional AU16/AU25) (canonical Disgust prior: nose wrinkle, upper lip raise)",
    3: "AU6+AU12 (Duchenne Happiness prior: cheek raise, lip corner puller)",
    4: "AU1+AU4+AU15 (canonical Sadness prior: inner brow raise, brow lowerer, lip corner depressor)",
    5: "AU4+AU7+AU23/AU24 (+ optional AU5) (canonical Anger prior: brow lowerer, lid tightener, lip tightener/pressor)",
    6: "No dominant emotion-specific AU prior for Neutral.",
}
EMOTIONS = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

def _path_keys(path):
    path = str(path).replace("\\", "/")
    parts = [p for p in path.split("/") if p]
    keys = {path, path.lower()}
    if parts:
        keys.add(parts[-1])
        keys.add(parts[-1].lower())
    for n in (2, 3):
        if len(parts) >= n:
            tail = "/".join(parts[-n:])
            keys.add(tail)
            keys.add(tail.lower())
    return keys

def load_openface_au_csv(csv_path):
    """Load OpenFace AU intensity rows keyed by several path variants."""
    if not csv_path:
        return {}
    if not os.path.exists(csv_path):
        print(f"⚠️ OpenFace AU CSV not found: {csv_path}. Falling back to FACS priors.")
        return {}

    au_rows = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {str(k).strip(): str(v).strip() for k, v in row.items()}
            if cleaned.get("error", "") or cleaned.get("openface_success", "").lower() in {"0", "false"}:
                continue
            vector = {}
            for col in OPENFACE_AU_COLUMNS:
                try:
                    vector[col] = float(cleaned.get(col, "nan"))
                except ValueError:
                    vector[col] = float("nan")
            if all(math.isnan(value) for value in vector.values()):
                continue
            for key_field in ("image_path", "relative_path", "path"):
                if cleaned.get(key_field):
                    for key in _path_keys(cleaned[key_field]):
                        au_rows[key] = vector

    print(f"✅ Loaded OpenFace AU estimates for {len(au_rows)} path keys from {csv_path}")
    return au_rows

# ==========================================
# 1. 数据加载
# ==========================================
def _split_path(data_dir, split):
    candidates = [
        os.path.join(data_dir, split),
        os.path.join(data_dir, 'DATASET', split),
        os.path.join(data_dir, 'original', split),
        os.path.join(data_dir, 'DATASET', 'original', split),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find split '{split}' under {data_dir}")


def _build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform


def get_eval_loader(data_dir=DATA_DIR, split='test', batch_size=BATCH_SIZE):
    """Load a deterministic RAF-DB split for validation, retrieval banks, or reports."""
    _, eval_transform = _build_transforms()
    split_path = _split_path(data_dir, split)
    dataset = datasets.ImageFolder(split_path, transform=eval_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )


def get_dataloaders(data_dir=DATA_DIR):
    """Load RAF-DB dataset with train/test split"""
    train_transform, test_transform = _build_transforms()
    
    try:
        train_path = _split_path(data_dir, 'train')
        test_path = _split_path(data_dir, 'test')
        
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
        
        # Compute class weights for imbalance
        targets = np.array(train_dataset.targets)
        class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=2, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                 num_workers=2, pin_memory=True)
        
        return train_loader, test_loader, class_weights
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

# ==========================================
# 2. 损失函数 (SupConLoss)
# ==========================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return -mean_log_prob_pos.mean()

# ==========================================
# 3. 核心模块: AFRN & GeM
# ==========================================
class SpectralCoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        mid_channels = max(8, in_channels // reduction_ratio)
        self.avg_pool = nn.AvgPool2d(3, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False), nn.GELU(),
            nn.Linear(mid_channels, in_channels, bias=True), nn.Sigmoid()
        )
        self.conv_shared = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False), nn.BatchNorm2d(mid_channels), nn.GELU()
        )
        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.conv_h, self.conv_w]:
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, -5.0)
        nn.init.constant_(self.mlp[-2].weight, 0)
        nn.init.constant_(self.mlp[-2].bias, -5.0)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        # Spectral
        low = self.avg_pool(x)
        high = x - low
        w_spec = self.mlp(self.gap(high).view(b, c)).view(b, c, 1, 1)
        # Coordinate
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w))
        cat = torch.cat([x_h, x_w.permute(0, 1, 3, 2)], dim=2)
        f = self.conv_shared(cat)
        f_h, f_w = torch.split(f, [h, w], dim=2)
        a_h = self.sigmoid(self.conv_h(f_h))
        a_w = self.sigmoid(self.conv_w(f_w.permute(0, 1, 3, 2)))
        return identity + identity * (w_spec * a_h * a_w)

class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)

# ==========================================
# 4. 模型定义 (ViT + AFRN + Dual-Head)
# ==========================================
class ViT_AFRN_Model(nn.Module):
    def __init__(self, num_classes=7, use_sfra=True):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.embed_dim = 768
        self.use_sfra = use_sfra
        
        self.afrn = SpectralCoordinateAttention(in_channels=self.embed_dim) if use_sfra else None
        self.gem = GeMPooling(p=3)
        self.head_norm = nn.LayerNorm(self.embed_dim)
        
        # Projection Head (SimCLR style): Only used for Loss, not for RAG
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 128)
        )
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x, mode='train'):
        """
        mode:
         - 'train': returns logits, proj_feats (for SupConLoss)
         - 'extract': returns embedding (for RAG Building)
         - 'inference': returns logits, embedding (for RAG Query)
        """
        x = self.backbone.forward_features(x)
        if x.shape[1] == 197: x = x[:, 1:, :]
        
        b, n, c = x.shape
        h = w = int(n**0.5)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        
        if self.use_sfra:
            x = self.afrn(x) # Spectral Refinement
        x = self.gem(x).flatten(1)
        embed = self.head_norm(x) # [B, 768] -> This is our "Visual Embedding"
        
        if mode == 'train':
            logits = self.head(embed)
            proj = F.normalize(self.projector(embed), dim=1) # [B, 128]
            return logits, proj
        
        elif mode == 'extract':
            return F.normalize(embed, dim=1) # Used for Memory Bank
            
        elif mode == 'inference':
            logits = self.head(embed)
            return logits, F.normalize(embed, dim=1)

# ==========================================
# 5. 训练主程序 (Training)
# ==========================================
def train_pipeline(
    save_path='outputs/models/ss_vlm_best.pth',
    data_dir=DATA_DIR,
    epochs=EPOCHS,
    metrics_dir='outputs/metrics',
    early_stop_patience=EARLY_STOP_PATIENCE,
    early_stop_min_delta=EARLY_STOP_MIN_DELTA,
    use_sfra=True,
    lambda_cont=LAMBDA_CONT,
):
    """Phase 1: Train the visual encoder with dual-head optimization"""
    train_loader, test_loader, class_weights = get_dataloaders(data_dir)
    if train_loader is None:
        print("Failed to load data!")
        return None

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    run_name = os.path.splitext(os.path.basename(save_path))[0]
    history_csv = os.path.join(metrics_dir, f"{run_name}_train_history.csv")
    summary_json = os.path.join(metrics_dir, f"{run_name}_train_summary.json")
    history_fields = [
        "epoch", "train_loss", "train_ce_loss", "train_supcon_loss",
        "train_acc", "val_acc", "lr", "is_best",
    ]
    with open(history_csv, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=history_fields).writeheader()

    model = ViT_AFRN_Model(num_classes=NUM_CLASSES, use_sfra=use_sfra).to(DEVICE)
    
    # Optimizer with paper settings: weight_decay=0.05
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    criterion_supcon = SupConLoss(temperature=TEMP)
    
    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_completed = 0
    early_stopped = False
    for epoch in range(epochs):
        epochs_completed = epoch + 1
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0
        train_ce_sum = 0.0
        train_supcon_sum = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            logits, proj_feats = model(imgs, mode='train')
            
            loss_ce = criterion_ce(logits, labels)
            if lambda_cont > 0:
                loss_con = criterion_supcon(proj_feats, labels)
            else:
                loss_con = torch.zeros((), device=DEVICE)
            loss = loss_ce + lambda_cont * loss_con
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss_sum += loss.item() * labels.size(0)
            train_ce_sum += loss_ce.item() * labels.size(0)
            train_supcon_sum += loss_con.item() * labels.size(0)
            
            loop.set_postfix(loss=f"{loss.item():.4f}", ce=f"{loss_ce.item():.3f}", con=f"{loss_con.item():.3f}")
        
        lr_current = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs, mode='inference')[0]  # Get logits only
                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        train_acc = train_correct / train_total
        avg_train_loss = train_loss_sum / train_total
        avg_ce_loss = train_ce_sum / train_total
        avg_supcon_loss = train_supcon_sum / train_total
        print(f"📊 Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        is_best = val_acc > best_acc + early_stop_min_delta
        if is_best:
            best_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"🎉 New Best Accuracy: {best_acc:.4f} (Saved to {save_path})")
        else:
            epochs_without_improvement += 1
            if early_stop_patience > 0:
                print(
                    f"⏳ No validation improvement for "
                    f"{epochs_without_improvement}/{early_stop_patience} epoch(s)"
                )

        with open(history_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history_fields)
            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_ce_loss": avg_ce_loss,
                "train_supcon_loss": avg_supcon_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": lr_current,
                "is_best": int(is_best),
            })

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            early_stopped = True
            print(
                f"🛑 Early stopping at epoch {epoch+1}. "
                f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}."
            )
            break
    
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": save_path,
            "data_dir": data_dir,
            "requested_epochs": epochs,
            "epochs_completed": epochs_completed,
            "best_epoch": best_epoch,
            "best_val_acc": best_acc,
            "early_stopped": early_stopped,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "use_sfra": use_sfra,
            "lambda_cont": lambda_cont,
            "history_csv": history_csv,
        }, f, indent=2)

    print(f"\n✅ Training completed! Best validation accuracy: {best_acc:.4f}")
    print(f"📁 Training history saved to: {history_csv}")
    print(f"📁 Training summary saved to: {summary_json}")
    return save_path

# ==========================================
# 6. RAG Engine (Phase 2 & 3)
# ==========================================
class RAG_Engine:
    def __init__(
        self,
        model_path,
        llm_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        openface_au_csv=None,
        top_k=TOP_K_RETRIEVAL,
        fusion_mode=RAG_FUSION_MODE,
        fusion_alpha=RAG_FUSION_ALPHA,
        retrieval_temperature=RAG_SIM_TEMPERATURE,
        k_prototypes=K_PROTOTYPES,
        kmeans_seed=KMEANS_SEED,
        load_llm=True,
        use_sfra=True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViT_AFRN_Model(num_classes=7, use_sfra=use_sfra).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.bank_vectors = None
        self.bank_labels = None
        self.bank_au_summaries = None
        self.openface_aus = load_openface_au_csv(openface_au_csv)
        self.top_k = top_k
        self.fusion_mode = fusion_mode
        self.fusion_alpha = fusion_alpha
        self.retrieval_temperature = retrieval_temperature
        self.k_prototypes = k_prototypes
        self.kmeans_seed = kmeans_seed
        self.use_sfra = use_sfra
        self.tokenizer = None
        self.llm = None
        
        if load_llm:
            print(f"🤖 Loading LLM: {llm_model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    def build_bank(self, loader):
        print("🔨 Building Prototype Memory Bank...")
        feats_all, labels_all, paths_all = [], [], []
        dataset = loader.dataset
        bank_loader = DataLoader(
            dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=getattr(loader, "num_workers", 0),
            pin_memory=getattr(loader, "pin_memory", False),
            drop_last=False,
        )
        dataset_samples = getattr(dataset, "samples", None)
        offset = 0
        with torch.no_grad():
            for imgs, labels in tqdm(bank_loader):
                imgs = imgs.to(self.device)
                feats = self.model(imgs, mode='extract') 
                feats_all.append(feats.cpu().numpy())
                labels_all.append(labels.numpy())
                if dataset_samples is not None:
                    batch_size = labels.size(0)
                    paths_all.extend([dataset_samples[i][0] for i in range(offset, offset + batch_size)])
                    offset += batch_size
        
        feats_all = np.concatenate(feats_all)
        labels_all = np.concatenate(labels_all)
        
        # KMeans clustering per class gives a compact retrieval bank.
        bank_vecs, bank_lbls, bank_au_summaries = [], [], []
        for c in range(7):
            idx = np.where(labels_all == c)[0]
            if len(idx) == 0: continue
            kmeans = KMeans(
                n_clusters=min(self.k_prototypes, len(idx)),
                n_init=10,
                random_state=self.kmeans_seed,
            ).fit(feats_all[idx])
            bank_vecs.append(kmeans.cluster_centers_)
            bank_lbls.extend([c] * len(kmeans.cluster_centers_))
            for cluster_id in range(len(kmeans.cluster_centers_)):
                member_indices = idx[kmeans.labels_ == cluster_id]
                member_paths = [paths_all[i] for i in member_indices] if paths_all else []
                bank_au_summaries.append(self._summarize_openface_aus(member_paths, c))
            
        self.bank_vectors = np.vstack(bank_vecs)
        self.bank_labels = np.array(bank_lbls)
        self.bank_au_summaries = bank_au_summaries

    def _lookup_openface_au(self, path):
        for key in _path_keys(path):
            if key in self.openface_aus:
                return self.openface_aus[key]
        return None

    def _summarize_openface_aus(self, member_paths, label_idx):
        au_vectors = [self._lookup_openface_au(path) for path in member_paths]
        au_vectors = [v for v in au_vectors if v is not None]
        if not au_vectors:
            return f"FACS-informed class prior: {FACS_PRIOR_MAPPING[label_idx]}"

        values = np.array([[v[col] for col in OPENFACE_AU_COLUMNS] for v in au_vectors], dtype=float)
        mean_values = np.nanmean(values, axis=0)
        active = [
            (col.replace("_r", ""), score)
            for col, score in zip(OPENFACE_AU_COLUMNS, mean_values)
            if np.isfinite(score) and score >= AU_ACTIVE_THRESHOLD
        ]
        active = sorted(active, key=lambda item: item[1], reverse=True)
        if active:
            active_text = ", ".join([f"{au}={score:.2f}" for au, score in active[:6]])
        else:
            active_text = f"no AU mean above {AU_ACTIVE_THRESHOLD:.1f}"
        return f"OpenFace-estimated AU intensities over {len(au_vectors)} cluster image(s): {active_text}"

    def _retrieval_distribution(self, retrieved_indices, retrieved_sims):
        """Convert retrieved prototype labels into a weighted class distribution."""
        if len(retrieved_indices) == 0:
            return np.ones(NUM_CLASSES, dtype=float) / NUM_CLASSES

        sims = np.array(retrieved_sims, dtype=float)
        temp = max(float(self.retrieval_temperature), 1e-6)
        scaled = sims / temp
        scaled = scaled - np.max(scaled)
        weights = np.exp(scaled)
        weights = weights / max(weights.sum(), 1e-12)

        dist = np.zeros(NUM_CLASSES, dtype=float)
        for idx, weight in zip(retrieved_indices, weights):
            dist[int(self.bank_labels[idx])] += float(weight)
        return dist

    def _fuse_prediction(self, classifier_probs, retrieval_dist):
        """Fuse classifier probabilities with retrieval evidence."""
        classifier_probs = np.array(classifier_probs, dtype=float)
        retrieval_dist = np.array(retrieval_dist, dtype=float)
        classifier_pred = int(np.argmax(classifier_probs))
        retrieval_pred = int(np.argmax(retrieval_dist))

        if self.fusion_mode == "none":
            final_probs = classifier_probs
            source = "classifier"
        elif self.fusion_mode == "majority":
            final_probs = retrieval_dist
            source = "retrieval"
        elif self.fusion_mode == "adaptive":
            classifier_conf = float(classifier_probs[classifier_pred])
            uncertainty = 1.0 - classifier_conf
            alpha = min(0.75, max(0.0, self.fusion_alpha + uncertainty * 0.5))
            final_probs = (1.0 - alpha) * classifier_probs + alpha * retrieval_dist
            source = f"adaptive_alpha={alpha:.3f}"
        else:
            alpha = min(1.0, max(0.0, self.fusion_alpha))
            final_probs = (1.0 - alpha) * classifier_probs + alpha * retrieval_dist
            source = f"weighted_alpha={alpha:.3f}"

        final_probs = final_probs / max(final_probs.sum(), 1e-12)
        final_pred = int(np.argmax(final_probs))
        return {
            "classifier_pred": classifier_pred,
            "classifier_conf": float(classifier_probs[classifier_pred]),
            "retrieval_pred": retrieval_pred,
            "retrieval_conf": float(retrieval_dist[retrieval_pred]),
            "final_pred": final_pred,
            "final_conf": float(final_probs[final_pred]),
            "source": source,
            "final_probs": final_probs,
        }

    def _format_evidence(self, retrieved_indices):
        evidence_texts = []
        for rank, idx in enumerate(retrieved_indices, start=1):
            lbl = int(self.bank_labels[idx])
            emo = EMOTIONS[lbl]
            if self.bank_au_summaries and idx < len(self.bank_au_summaries):
                au_desc = self.bank_au_summaries[idx]
            else:
                au_desc = f"FACS-informed class prior: {FACS_PRIOR_MAPPING[lbl]}"
            evidence_texts.append(f"{rank}. {emo} prototype: {au_desc}")
        return "\n".join(evidence_texts)

    def _extract_valid_au_ids(self, text):
        found = set()
        for match in AU_MENTION_RE.finditer(text or ""):
            au_id = f"AU{int(match.group(1)):02d}"
            if au_id in VALID_AU_IDS:
                found.add(au_id)
        return found

    def _extract_malformed_au_mentions(self, text):
        malformed = set()
        for match in re.finditer(r"\bAUs?\s+(\d+)\s+to\s+(\d+)", text or "", flags=re.I):
            start = f"AU{int(match.group(1)):02d}" if len(match.group(1)) <= 2 else f"AU{match.group(1)}"
            end = f"AU{int(match.group(2)):02d}" if len(match.group(2)) <= 2 else f"AU{match.group(2)}"
            if start not in VALID_AU_IDS or end not in VALID_AU_IDS:
                malformed.add(f"{start}-{end}")
        for match in re.finditer(r"\bAUs?\s+([0-9][0-9,\s]*(?:and\s+\d+)?)", text or "", flags=re.I):
            for raw_number in re.findall(r"\d+", match.group(1)):
                au_id = f"AU{int(raw_number):02d}" if len(raw_number) <= 2 else f"AU{raw_number}"
                if au_id not in VALID_AU_IDS:
                    malformed.add(au_id)
        for match in AU_MENTION_RE.finditer(text or ""):
            au_id = f"AU{int(match.group(1)):02d}" if len(match.group(1)) <= 2 else f"AU{match.group(1)}"
            if au_id not in VALID_AU_IDS:
                malformed.add(au_id)
        return malformed

    def _report_has_au_hallucination(self, report, evidence_str):
        if self._extract_malformed_au_mentions(report):
            return True
        report_aus = self._extract_valid_au_ids(report)
        evidence_aus = self._extract_valid_au_ids(evidence_str)
        return bool(report_aus and not report_aus.issubset(evidence_aus))

    def _build_safe_report(self, pred_emo, conf, classifier_emo, retrieval_emo, evidence_str):
        prototype_labels = []
        for line in evidence_str.splitlines():
            match = re.match(r"\d+\.\s+([A-Za-z]+)\s+prototype", line.strip())
            if match:
                prototype_labels.append(match.group(1))
        prototype_summary = ", ".join(prototype_labels[:9]) if prototype_labels else "not available"

        active_aus = sorted(set(re.findall(r"\bAU\d{2}=\d+(?:\.\d+)?", evidence_str)))
        if active_aus:
            au_summary = "Explicit active AU evidence: " + ", ".join(active_aus[:8]) + "."
        elif "no AU mean above" in evidence_str:
            au_summary = "The retrieved evidence reports no AU mean above the active threshold for the listed prototypes."
        elif "FACS-informed class prior" in evidence_str:
            au_summary = "The retrieved evidence provides FACS-informed class priors rather than image-specific AU intensities."
        else:
            au_summary = "No explicit AU identifier should be inferred beyond the retrieved evidence."

        support_text = "limited or mixed" if classifier_emo != retrieval_emo else "consistent"
        return (
            "Observation: "
            f"The final fused expression is {pred_emo} with confidence {conf:.2f}. "
            f"The classifier prediction is {classifier_emo}, and the retrieval-majority prediction is {retrieval_emo}.\n\n"
            "Evidence:\n"
            f"1. Retrieved prototype labels in rank order: {prototype_summary}.\n"
            f"2. {au_summary}\n\n"
            "Conclusion: "
            f"The expression label is reported as {pred_emo} with {support_text} retrieval evidence. "
            "No additional AU identifiers, clinical conditions, demographics, or non-facial attributes are inferred."
        )

    def generate_report(self, pred_idx, conf, retrieved_indices, classifier_idx=None, retrieval_idx=None):
        if self.tokenizer is None or self.llm is None:
            return ""

        pred_emo = EMOTIONS[pred_idx]
        classifier_emo = EMOTIONS[classifier_idx] if classifier_idx is not None else pred_emo
        retrieval_emo = EMOTIONS[retrieval_idx] if retrieval_idx is not None else pred_emo
        evidence_str = self._format_evidence(retrieved_indices)

        user_prompt = f"""Prediction summary:
- Final fused expression: {pred_emo}
- Final fused confidence: {conf:.2f}
- Classifier expression: {classifier_emo}
- Retrieval-majority expression: {retrieval_emo}

Retrieved prototype evidence:
{evidence_str}

Write exactly three short sections:
Observation:
Evidence:
Conclusion:

Rules:
- Use only the prediction summary and retrieved prototype evidence.
- Mention OpenFace-estimated AU intensities only when they appear in the evidence.
- Only mention AU identifiers that appear exactly as AUxx or AUxx=value in the evidence.
- Never describe cluster image counts as AU identifiers.
- Do not infer mental health disorders, demographics, intent, or non-facial symptoms.
- If evidence conflicts, say the expression label is supported with limited or mixed retrieval evidence."""

        messages = [
            {
                "role": "system",
                "content": "You write concise, evidence-grounded facial expression reports. You never add clinical disorders or facts not present in the evidence.",
            },
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"System: {messages[0]['content']}\nUser: {user_prompt}\nAssistant:"

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        output_ids = self.llm.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        report = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if self._report_has_au_hallucination(report, evidence_str):
            return self._build_safe_report(pred_emo, conf, classifier_emo, retrieval_emo, evidence_str)
        return report

    def run_inference(
        self,
        test_loader,
        num_samples_to_show=5,
        metrics_dir='outputs/metrics',
        run_name='ss_vlm',
        report_sampling='first',
        reports_per_class=1,
    ):
        """Phase 3: RAG inference with retrieval-fused prediction."""
        if self.bank_vectors is None:
            raise ValueError("Memory bank not built! Call build_bank() first.")
        
        k_limit = max(1, min(int(self.top_k), len(self.bank_labels)))
        print(
            f"🔍 Running RAG Inference "
            f"(k={k_limit}, fusion={self.fusion_mode}, alpha={self.fusion_alpha})..."
        )
        all_final_preds, all_classifier_preds, all_labels, all_reports = [], [], [], []
        prediction_rows = []
        k_stats = {
            k: {
                "classifier_correct": 0,
                "retrieval_correct": 0,
                "fused_correct": 0,
                "retrieval_consistency_sum": 0.0,
            }
            for k in range(1, k_limit + 1)
        }
        
        final_correct = 0
        classifier_correct = 0
        retrieval_correct = 0
        total = 0
        consistency_sum = 0
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
                labels = labels.numpy()
                batch_start = sample_offset
                batch_size = len(labels)
                if dataset_samples is not None:
                    batch_paths = [dataset_samples[i][0] for i in range(batch_start, batch_start + batch_size)]
                else:
                    batch_paths = [""] * batch_size
                sample_offset += batch_size
                
                # Get predictions and embeddings
                logits, embeddings = self.model(imgs, mode='inference')
                probs = F.softmax(logits, dim=1)
                
                embeddings = embeddings.cpu().numpy()
                probs_np = probs.cpu().numpy()
                total += len(labels)
                
                # Retrieval for each sample in batch
                for j in range(len(labels)):
                    sample_index = batch_start + j
                    query_feat = embeddings[j:j+1]
                    classifier_probs = probs_np[j]
                    gt_lbl = int(labels[j])
                    
                    # Cosine similarity retrieval.
                    sims = cosine_similarity(query_feat, self.bank_vectors)[0]
                    sorted_idx = np.argsort(sims)[::-1][:k_limit]
                    sorted_sims = sims[sorted_idx]

                    for k in range(1, k_limit + 1):
                        k_indices = sorted_idx[:k]
                        k_sims = sorted_sims[:k]
                        k_dist = self._retrieval_distribution(k_indices, k_sims)
                        k_decision = self._fuse_prediction(classifier_probs, k_dist)
                        k_ret_pred = int(np.argmax(k_dist))
                        k_stats[k]["classifier_correct"] += int(k_decision["classifier_pred"] == gt_lbl)
                        k_stats[k]["retrieval_correct"] += int(k_ret_pred == gt_lbl)
                        k_stats[k]["fused_correct"] += int(k_decision["final_pred"] == gt_lbl)
                        k_stats[k]["retrieval_consistency_sum"] += (
                            np.sum(self.bank_labels[k_indices] == k_decision["final_pred"]) / k
                        )

                    top_k_idx = sorted_idx[:k_limit]
                    top_k_sims = sorted_sims[:k_limit]
                    ret_labels = self.bank_labels[top_k_idx]
                    retrieval_dist = self._retrieval_distribution(top_k_idx, top_k_sims)
                    decision = self._fuse_prediction(classifier_probs, retrieval_dist)
                    
                    # Calculate retrieval consistency against the final fused decision.
                    consistency = np.sum(ret_labels == decision["final_pred"]) / k_limit
                    consistency_sum += consistency
                    final_correct += int(decision["final_pred"] == gt_lbl)
                    classifier_correct += int(decision["classifier_pred"] == gt_lbl)
                    retrieval_correct += int(decision["retrieval_pred"] == gt_lbl)
                    all_final_preds.append(int(decision["final_pred"]))
                    all_classifier_preds.append(int(decision["classifier_pred"]))
                    all_labels.append(gt_lbl)
                    prediction_rows.append({
                        "sample_index": sample_index,
                        "image_path": batch_paths[j],
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
                        "fusion_source": decision["source"],
                        "retrieval_consistency": float(consistency),
                        "top_k_indices": ";".join(str(int(i)) for i in top_k_idx),
                        "top_k_labels": ";".join(str(int(i)) for i in ret_labels),
                        "top_k_emotions": ";".join(EMOTIONS[int(i)] for i in ret_labels),
                        "top_k_similarities": ";".join(f"{float(s):.6f}" for s in top_k_sims),
                        "retrieval_distribution": ";".join(f"{float(v):.6f}" for v in retrieval_dist),
                        "fused_distribution": ";".join(f"{float(v):.6f}" for v in decision["final_probs"]),
                    })
                    
                    if should_generate_report(batch_idx, j, gt_lbl):
                        report = self.generate_report(
                            decision["final_pred"],
                            decision["final_conf"],
                            top_k_idx,
                            classifier_idx=decision["classifier_pred"],
                            retrieval_idx=decision["retrieval_pred"],
                        )
                        all_reports.append({
                            'sample_index': sample_index,
                            'image_path': batch_paths[j],
                            'gt': EMOTIONS[gt_lbl],
                            'pred': EMOTIONS[int(decision["final_pred"])],
                            'classifier_pred': EMOTIONS[int(decision["classifier_pred"])],
                            'retrieval_pred': EMOTIONS[int(decision["retrieval_pred"])],
                            'confidence': float(decision["final_conf"]),
                            'consistency': float(consistency),
                            'retrieved_evidence': self._format_evidence(top_k_idx),
                            'report': report
                        })
                        report_counts_by_class[gt_lbl] += 1
        
        # Print results
        accuracy = final_correct / total
        classifier_accuracy = classifier_correct / total
        retrieval_accuracy = retrieval_correct / total
        avg_consistency = consistency_sum / total
        print(f"\n📊 RAG Inference Results:")
        print(f"   Classifier Accuracy: {classifier_accuracy:.4f} ({classifier_correct}/{total})")
        print(f"   Retrieval-Majority Accuracy: {retrieval_accuracy:.4f} ({retrieval_correct}/{total})")
        print(f"   RAG-Fused Accuracy: {accuracy:.4f} ({final_correct}/{total})")
        print(f"   Avg Retrieval Consistency: {avg_consistency:.4f}")
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
                    "avg_retrieval_consistency",
                ],
            )
            writer.writeheader()
            for k, stats in k_stats.items():
                writer.writerow({
                    "k": k,
                    "classifier_accuracy": stats["classifier_correct"] / total,
                    "retrieval_accuracy": stats["retrieval_correct"] / total,
                    "rag_fused_accuracy": stats["fused_correct"] / total,
                    "avg_retrieval_consistency": stats["retrieval_consistency_sum"] / total,
                })

        cm = confusion_matrix(all_labels, all_final_preds, labels=list(range(NUM_CLASSES))).tolist()
        classifier_cm = confusion_matrix(all_labels, all_classifier_preds, labels=list(range(NUM_CLASSES))).tolist()
        cls_report = classification_report(
            all_labels,
            all_final_preds,
            labels=list(range(NUM_CLASSES)),
            target_names=EMOTIONS,
            output_dict=True,
            zero_division=0,
        )
        classifier_report = classification_report(
            all_labels,
            all_classifier_preds,
            labels=list(range(NUM_CLASSES)),
            target_names=EMOTIONS,
            output_dict=True,
            zero_division=0,
        )
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump({
                "accuracy": float(accuracy),
                "correct": int(final_correct),
                "classifier_accuracy": float(classifier_accuracy),
                "classifier_correct": int(classifier_correct),
                "retrieval_accuracy": float(retrieval_accuracy),
                "retrieval_correct": int(retrieval_correct),
                "total": int(total),
                "top_k_retrieval": int(k_limit),
                "fusion_mode": self.fusion_mode,
                "fusion_alpha": float(self.fusion_alpha),
                "retrieval_temperature": float(self.retrieval_temperature),
                "k_prototypes": int(self.k_prototypes),
                "kmeans_seed": int(self.kmeans_seed),
                "use_sfra": bool(self.use_sfra),
                "avg_retrieval_consistency": float(avg_consistency),
                "confusion_matrix": cm,
                "classifier_confusion_matrix": classifier_cm,
                "classification_report": cls_report,
                "classifier_classification_report": classifier_report,
                "predictions_csv": predictions_csv,
                "k_sensitivity_csv": k_sensitivity_csv,
            }, f, indent=2)

        with open(reports_json, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2)

        print(f"📁 RAG predictions saved to: {predictions_csv}")
        print(f"📁 RAG summary saved to: {summary_json}")
        print(f"📁 k-sensitivity saved to: {k_sensitivity_csv}")
        print(f"📁 Sample reports saved to: {reports_json}")
        
        # Print sample reports
        if all_reports:
            print(f"\n📝 Sample Clinical Reports (first {len(all_reports)} cases):\n")
            for i, sample in enumerate(all_reports):
                print(f"{'='*60}")
                print(f"Sample {i+1} | Ground Truth: {sample['gt']} | Prediction: {sample['pred']}")
                print(f"Confidence: {sample['confidence']:.2f} | Retrieval Consistency: {sample['consistency']:.2f}")
                print(f"{'-'*60}")
                print(sample['report'])
                print()
        
        return accuracy, avg_consistency

# ==========================================
# 7. Main Entry Point - Complete Pipeline
# ==========================================
def main():
    """
    SS-VLM Complete Pipeline:
    Phase 1: Train visual encoder with dual-head optimization
    Phase 2: Build prototype memory bank
    Phase 3: RAG inference with structured report generation
    """
    import argparse
    parser = argparse.ArgumentParser(description='SS-VLM: Spectral-Symbolic Vision-Language Model')
    parser.add_argument('--mode', type=str, default='full', choices=['train', 'rag', 'full'],
                        help='Run mode: train (phase 1 only), rag (phase 2&3 only), full (complete pipeline)')
    parser.add_argument('--model_path', type=str, default='outputs/models/ss_vlm_best.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SSVLM_DATA_DIR', DATA_DIR),
                        help='RAF-DB root containing train/test or DATASET/train and DATASET/test')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training and use existing model')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs; default: {EPOCHS}')
    parser.add_argument('--early_stop_patience', type=int, default=EARLY_STOP_PATIENCE,
                        help='Stop training after this many epochs without validation improvement; use 0 to disable')
    parser.add_argument('--early_stop_min_delta', type=float, default=EARLY_STOP_MIN_DELTA,
                        help='Minimum validation accuracy gain required to count as an improvement')
    parser.add_argument('--disable_sfra', action='store_true',
                        help='Disable the SFRA/AFRN refinement block for ViT baseline ablations')
    parser.add_argument('--lambda_cont', type=float, default=LAMBDA_CONT,
                        help=f'SupCon loss weight; use 0 for CE-only ablations; default: {LAMBDA_CONT}')
    parser.add_argument('--metrics_dir', type=str, default='outputs/metrics',
                        help='Directory for training history, RAG predictions, summaries, and reports')
    parser.add_argument('--openface_au_csv', type=str, default=os.environ.get('SSVLM_OPENFACE_AU_CSV', ''),
                        help='Optional CSV generated by tools/extract_openface2_aus.py for prototype-level AU summaries')
    parser.add_argument('--top_k', type=int, default=TOP_K_RETRIEVAL,
                        help=f'Number of prototypes retrieved per test image; default: {TOP_K_RETRIEVAL}')
    parser.add_argument('--rag_fusion', type=str, default=RAG_FUSION_MODE,
                        choices=['none', 'weighted', 'majority', 'adaptive'],
                        help='How retrieval evidence changes the final prediction')
    parser.add_argument('--rag_alpha', type=float, default=RAG_FUSION_ALPHA,
                        help='Retrieval weight for weighted/adaptive RAG fusion')
    parser.add_argument('--retrieval_temperature', type=float, default=RAG_SIM_TEMPERATURE,
                        help='Softmax temperature for similarity-weighted retrieval votes')
    parser.add_argument('--k_prototypes', type=int, default=K_PROTOTYPES,
                        help=f'K-Means prototypes per class; default: {K_PROTOTYPES}')
    parser.add_argument('--kmeans_seed', type=int, default=KMEANS_SEED,
                        help='Random seed for prototype K-Means clustering')
    parser.add_argument('--num_reports', type=int, default=5,
                        help='Number of sample qualitative reports to generate')
    parser.add_argument('--report_sampling', type=str, default='first',
                        choices=['first', 'class_balanced'],
                        help='How qualitative report samples are selected')
    parser.add_argument('--reports_per_class', type=int, default=1,
                        help='Reports per ground-truth class when --report_sampling class_balanced')
    parser.add_argument('--skip_llm_reports', action='store_true',
                        help='Skip LLM loading/report generation and only compute metrics')
    args = parser.parse_args()
    
    # Phase 1: Training
    if args.mode in ['train', 'full'] and not args.skip_train:
        print("="*60)
        print("🚀 PHASE 1: Training Visual Encoder")
        print("="*60)
        model_path = train_pipeline(
            save_path=args.model_path,
            data_dir=args.data_dir,
            epochs=args.epochs,
            metrics_dir=args.metrics_dir,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            use_sfra=not args.disable_sfra,
            lambda_cont=args.lambda_cont,
        )
        if model_path is None:
            print("Training failed!")
            return
    else:
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            return
        print(f"✅ Using existing model: {model_path}")
    
    # Phase 2 & 3: RAG
    if args.mode in ['rag', 'full']:
        print("\n" + "="*60)
        print("🚀 PHASE 2 & 3: Memory Bank Construction & RAG Inference")
        print("="*60)
        
        # Load deterministic data for retrieval/evaluation. Training augmentation would
        # make prototype construction non-reproducible, so RAG uses eval transforms.
        try:
            train_bank_loader = get_eval_loader(args.data_dir, split='train')
            test_loader = get_eval_loader(args.data_dir, split='test')
        except Exception as exc:
            print(f"Failed to load data: {exc}")
            print("Failed to load data!")
            return
        
        # Initialize RAG Engine
        rag = RAG_Engine(
            model_path,
            openface_au_csv=args.openface_au_csv,
            top_k=args.top_k,
            fusion_mode=args.rag_fusion,
            fusion_alpha=args.rag_alpha,
            retrieval_temperature=args.retrieval_temperature,
            k_prototypes=args.k_prototypes,
            kmeans_seed=args.kmeans_seed,
            load_llm=(
                not args.skip_llm_reports
                and (
                    args.num_reports > 0
                    or (args.report_sampling == 'class_balanced' and args.reports_per_class > 0)
                )
            ),
            use_sfra=not args.disable_sfra,
        )
        
        # Phase 2: Build memory bank
        rag.build_bank(train_bank_loader)
        
        # Phase 3: RAG inference
        run_name = os.path.splitext(os.path.basename(model_path))[0]
        accuracy, consistency = rag.run_inference(
            test_loader,
            num_samples_to_show=args.num_reports,
            metrics_dir=args.metrics_dir,
            run_name=run_name,
            report_sampling=args.report_sampling,
            reports_per_class=args.reports_per_class,
        )
        
        print("\n" + "="*60)
        print("✅ SS-VLM Pipeline Completed Successfully!")
        print(f"   Final RAG-Fused Accuracy: {accuracy:.4f}")
        print(f"   Retrieval Consistency (k={args.top_k}): {consistency:.4f}")
        print("="*60)

if __name__ == '__main__':
    main()
