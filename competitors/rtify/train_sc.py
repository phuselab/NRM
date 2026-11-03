#!/usr/bin/env python3
"""
competitors/rtify/train_sc.py
=============================
Two-phase RTify training on SC scene categorisation.

Phase 1 — CNN backbone pre-training on SUN_filtered (natural / manmade).
Phase 2 — RTify accumulator training on experiment-1 val split (4 seeds).
Evaluation  — experiment-1 test split → rtify_metrics_*.csv

Adapted from train_mnist_behavior.py in the original RTify code.
"""

import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── RTify path injection ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# RNN_decision is a vendored top-level module. Importing it via its file path
# keeps this directory off sys.path: the sibling `utils.py` would otherwise
# shadow `utils` for every later import in the same process (e.g.
# competitors/crnn's `from utils.loss import ...`).
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "competitors.rtify._vendored.RNN_decision",
    os.path.join(_HERE, "RNN_decision.py"),
)
_rnn_decision = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rnn_decision)
DiffDecisionMultiClass = _rnn_decision.DiffDecisionMultiClass

# ── Project imports ───────────────────────────────────────────────────────────
from datasets.sc.loader import SCDataset, sc_collate_fn  # noqa: E402
from scripts.eval_sc import (                                       # noqa: E402
    PROB_LIST,
    SCEvaluator,
    compute_human_nc_global,
    compute_human_nc_per_prob,
)
from scripts.results_io import display_path, refresh_path, refresh_tag  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — TRANSFORMS AND IMAGE DATASETS
# ══════════════════════════════════════════════════════════════════════════════

RTIFY_TRANSFORM = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


class SUNFilteredDataset(Dataset):
    """
    SUN_filtered train or val split.
    natural/ → label 0,  manmade/ → label 1.
    """

    def __init__(self, root: str, transform=None):
        self.transform = transform or RTIFY_TRANSFORM
        self.class_to_idx = {"natural": 0, "manmade": 1}
        self.samples: list[tuple[str, int]] = []
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Missing subfolder: {cls_dir}")
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))
        print(f"  SUN_filtered ({display_path(root)}): {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class SCImageDataset(Dataset):
    """
    Wraps a SCDataset split and loads actual images for RTify inference.
    Returns all behavioural metadata together with the preprocessed image.
    """

    def __init__(self, sc_ds: SCDataset, transform=None):
        self.ds = sc_ds
        self.transform = transform or RTIFY_TRANSFORM

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        img = Image.open(item["image_path"]).convert("L")
        item["image"] = self.transform(img)
        return item


def _sc_image_collate(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])
    base = sc_collate_fn([{k: v for k, v in b.items() if k != "image"} for b in batch])
    base["images"] = images
    return base


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class CNNEncoder(nn.Module):
    """
    Lightweight CNN for 150×150 greyscale images.
    Mirrors the CNNEncoder in train_mnist_behavior.py.
    """

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # → 75×75
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # → 37×37
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),            # → 1×1
            nn.Flatten(),
            nn.Linear(128, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Accumulator(nn.Module):
    """
    GRU + RTify evidence accumulation module.
    Trainable parameters (Phase 2):
      gru, linear, w, b, threshold, fc1, fc2.
    Takes a fixed feature vector and unrolls for T steps.
    Mirrors RTifyModel in train_mnist_behavior.py.
    """

    def __init__(
        self,
        feat_dim:   int   = 128,
        hidden_dim: int   = 128,
        n_classes:  int   = 2,
        time_steps: int   = 40,
        sigma:      float = 2.0,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.sigma      = sigma
        self.n_classes  = n_classes

        self.gru      = nn.GRU(feat_dim, hidden_dim, batch_first=False)
        self.fc1      = nn.Linear(hidden_dim, n_classes)
        self.fc2      = nn.Linear(time_steps, 1)
        self.linear   = nn.Linear(hidden_dim, n_classes)
        self.w         = nn.Parameter(torch.ones(1, time_steps))
        self.b         = nn.Parameter(torch.zeros(1, time_steps))
        self.threshold = nn.Parameter(torch.tensor(0.2))

    def forward(self, feat: torch.Tensor):
        """
        feat : (B, feat_dim) — from frozen CNNEncoder.

        Returns
        -------
        decision_logits : (B, n_classes)
        rt_pred         : (B,)  normalised to [1/T, 1]
        """
        device = feat.device
        B = feat.shape[0]
        T = self.time_steps

        # ── GRU unroll with same feature repeated T times ─────────────────
        feat_seq = feat.unsqueeze(0).expand(T, -1, -1)       # (T, B, feat_dim)
        gru_out, _ = self.gru(feat_seq)                       # (T, B, hidden_dim)

        # ── Evidence trajectory ────────────────────────────────────────────
        # (B, T, C): per-class evidence at each step
        s_traj = self.linear(gru_out).permute(1, 0, 2)       # (B, T, C)
        s_traj = s_traj * self.w.unsqueeze(-1) + self.b.unsqueeze(-1)
        s_accumulated = torch.cumsum(s_traj, dim=1)           # (B, T, C)

        dsdt = torch.diff(s_accumulated, dim=1)
        dsdt = torch.cat([dsdt[:, :1], dsdt], dim=1)          # (B, T, C)

        # ── Differentiable decision times ──────────────────────────────────
        decision_time = DiffDecisionMultiClass.apply(
            s_accumulated - self.threshold, dsdt
        )                                                      # (B, C)
        decision_time_min = decision_time.min(dim=1).values   # (B,)

        # ── Soft temporal attention ────────────────────────────────────────
        t_range    = torch.arange(T, device=device, dtype=torch.float32)
        soft_index = torch.sigmoid(
            (decision_time_min.unsqueeze(1) - t_range + 0.5) * self.sigma
        )                                                      # (B, T)

        # ── Decision logits (weighted across time) ─────────────────────────
        logit_traj = self.fc1(gru_out).permute(1, 0, 2)       # (B, T, C)
        logit_traj = logit_traj * self.fc2.weight.unsqueeze(-1)  # (B, T, C)
        decision_logits = (
            (logit_traj * soft_index.unsqueeze(-1)).sum(dim=1)
            + self.fc2.bias
        )                                                      # (B, C)

        # ── Normalised RT in [1/T, 1] ──────────────────────────────────────
        rt_pred = (decision_time_min + 1) / T                  # (B,)

        return decision_logits, rt_pred


class RTifyModel(nn.Module):
    """
    Frozen CNNEncoder + Accumulator.
    Used during Phase 2 and evaluation.
    """

    def __init__(self, encoder: CNNEncoder, accumulator: Accumulator):
        super().__init__()
        self.encoder     = encoder
        self.accumulator = accumulator

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            feat = self.encoder(x)                 # gradients blocked
        return self.accumulator(feat)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOSSES
# ══════════════════════════════════════════════════════════════════════════════

def neg_pearson_corr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative Pearson correlation — minimising this maximises rank-order agreement.
    Returns 0.0 when either vector is constant (no gradient explosion).
    """
    p_std = pred.std()
    t_std = target.std()
    if p_std < 1e-8 or t_std < 1e-8:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    pc = pred   - pred.mean()
    tc = target - target.mean()
    r  = (pc * tc).sum() / (pc.norm() * tc.norm() + 1e-8)
    return -r


def rt_loss_combined(
    rt_pred:   torch.Tensor,   # (B,) model RT, normalised
    rt_target: torch.Tensor,   # (B,) human RT, normalised to [0,1]
) -> torch.Tensor:
    """
    MSE (scale calibration) + NegativePearsonCorrelation (rank-order agreement).
    Following train_mnist_behavior.py which uses MSE; Pearson adds rank signal.
    """
    return F.mse_loss(rt_pred, rt_target) + neg_pearson_corr_loss(rt_pred, rt_target)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PHASE 1: CNN BACKBONE PRE-TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_cnn_phase1(
    encoder:      CNNEncoder,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    n_epochs:     int,
    lr:           float,
    device:       torch.device,
    save_path:    str,
) -> CNNEncoder:
    """
    Pre-train CNNEncoder + linear head on SUN_filtered (Cross-Entropy).
    Saves best-val-accuracy encoder weights to save_path.
    """
    head      = nn.Linear(128, 2).to(device)    # binary classification head
    params    = list(encoder.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    for epoch in range(n_epochs):
        # ── Train ─────────────────────────────────────────────────────────
        encoder.train(); head.train()
        correct = total = 0
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = head(encoder(imgs))
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
        train_acc  = correct / total
        train_loss = total_loss / total

        # ── Val ───────────────────────────────────────────────────────────
        encoder.eval(); head.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)
                logits = head(encoder(imgs))
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)
        val_acc = val_correct / val_total

        print(f"  [Phase1] epoch {epoch+1:3d}/{n_epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(encoder.state_dict(), save_path)
            print(f"    → best encoder saved (val_acc={val_acc:.4f})")

    return encoder


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PHASE 2: ACCUMULATOR TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_accumulator(
    model:        RTifyModel,
    val_img_ds:   SCImageDataset,
    batch_size:   int,
    seed:         int,
    n_epochs:     int,
    lr:           float,
    rt_weight:    float,
    rt_min:       float,
    rt_range:     float,
    device:       torch.device,
    save_path:    str,
) -> Accumulator:
    """
    Train the accumulator module on experiment-1 val split.
    The CNNEncoder inside model is frozen (RTifyModel.forward blocks gradients).

    Loss: CE(pred_choice, human_choice) + RT_WEIGHT * rt_loss(rt_pred, rt_norm)
    RT normalisation: rt_norm = (rt_seconds - rt_min) / rt_range  ∈ [0, 1]
    Following train_mnist_behavior.py structure exactly.
    """
    torch.manual_seed(seed)
    nn.init.xavier_uniform_(model.accumulator.linear.weight)
    nn.init.zeros_(model.accumulator.linear.bias)
    nn.init.orthogonal_(model.accumulator.gru.weight_ih_l0)
    nn.init.orthogonal_(model.accumulator.gru.weight_hh_l0)
    with torch.no_grad():
        model.accumulator.w.fill_(1.0)
        model.accumulator.b.fill_(0.0)
        model.accumulator.threshold.fill_(0.2)

    optimizer = optim.Adam(model.accumulator.parameters(), lr=lr)
    label_crit = nn.CrossEntropyLoss()

    loader = DataLoader(
        val_img_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=_sc_image_collate,
        pin_memory=(device.type == "cuda"),
    )

    best_loss = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    for epoch in range(n_epochs):
        model.train()
        model.encoder.eval()           # encoder stays in eval mode (BN frozen)

        total_loss = total_ce = total_rt = 0.0
        correct = total = 0
        all_rt_pred, all_rt_true = [], []

        for batch in loader:
            imgs      = batch["images"].to(device)
            h_choices = batch["human_choices"].to(device)
            h_rts     = batch["rts"].to(device)               # seconds

            # Exclude trials with invalid RTs
            valid = h_rts.isfinite() & (h_rts > 0)
            if valid.sum() == 0:
                continue

            imgs      = imgs[valid]
            h_choices = h_choices[valid]
            h_rts     = h_rts[valid]

            rt_norm = ((h_rts - rt_min) / rt_range).clamp(0.0, 1.0)

            optimizer.zero_grad()
            logits, rt_pred = model(imgs)

            ce_loss  = label_crit(logits, h_choices)
            rt_loss  = rt_loss_combined(rt_pred, rt_norm)
            loss     = ce_loss + rt_weight * rt_loss

            loss.backward()
            optimizer.step()

            bs           = imgs.size(0)
            total_loss  += loss.item()  * bs
            total_ce    += ce_loss.item() * bs
            total_rt    += rt_loss.item() * bs
            correct     += (logits.argmax(1) == h_choices).sum().item()
            total       += bs

            all_rt_pred.extend(rt_pred.detach().cpu().numpy())
            all_rt_true.extend(rt_norm.cpu().numpy())

        if total == 0:
            continue
        acc     = correct / total
        avg_rt  = float(np.corrcoef(all_rt_pred, all_rt_true)[0, 1]) if len(set(all_rt_pred)) > 1 else 0.0
        avg_loss = total_loss / total

        print(f"  [Phase2 seed={seed}] epoch {epoch+1:3d}/{n_epochs}  "
              f"loss={avg_loss:.4f}  ce={total_ce/total:.4f}  "
              f"rt={total_rt/total:.4f}  acc={acc:.4f}  r_rt={avg_rt:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.accumulator.state_dict(), save_path)

    print(f"  → best accumulator saved: {display_path(save_path)}")
    return model.accumulator


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_eval_data(
    models:       list[RTifyModel],   # 4 seed models
    test_img_ds:  SCImageDataset,
    batch_size:   int,
    device:       torch.device,
    rt_min:       float,
    rt_range:     float,
) -> dict:
    """
    Forward all 4 seed RTify models on the test split.
    Assembles the `data` dict for SCEvaluator.compute_metrics().
    """
    loader = DataLoader(
        test_img_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=_sc_image_collate,
        pin_memory=(device.type == "cuda"),
    )

    img_idxs, true_labs, hum_ch, pred_ch = [], [], [], []
    true_rt, pred_rt = [], []
    prob_vals: list[float] = []

    rt_samples_by_prob:     dict[float, list] = defaultdict(list)
    human_rts_by_prob:      dict[float, list] = defaultdict(list)
    trial_pred_rts_by_prob: dict[float, list] = defaultdict(list)
    pred_probs_by_prob:     dict[float, list] = defaultdict(list)
    model_info_by_prob:     dict[float, dict] = defaultdict(
        lambda: {"pred_choices": [], "true_labels": [], "human_choices": []}
    )

    for m in models:
        m.eval()

    print("Collecting RTify predictions on test set...")
    for batch in tqdm(loader, desc="  eval"):
        imgs      = batch["images"].to(device)
        b_probs   = batch["probs"]
        h_rts     = batch["rts"]                               # (B,) seconds
        h_choices = batch["human_choices"]                     # (B,)
        t_labels  = batch["choices"]                           # (B,)
        b_img_idx = batch["image_index"]                       # (B,)

        # ── Run all 4 seed models ──────────────────────────────────────────
        all_rt_s  = []   # list of (B,) tensors, one per seed
        all_log_s = []   # list of (B, 2) tensors
        for m in models:
            logits, rt_n = m(imgs)                             # rt_n in [1/T, 1]
            rt_sec = rt_n * rt_range + rt_min                  # convert to seconds
            all_rt_s.append(rt_sec.cpu())
            all_log_s.append(logits.cpu())

        # Ensemble: median logits → choice, softmax for AUC
        logits_stack = torch.stack(all_log_s, dim=0)           # (4, B, 2)
        rt_stack     = torch.stack(all_rt_s, dim=0)            # (4, B)

        med_logits = logits_stack.median(dim=0).values          # (B, 2)
        med_rt     = rt_stack.median(dim=0).values              # (B,)

        choices   = med_logits.argmax(dim=1)                    # (B,)
        probs1    = F.softmax(med_logits, dim=1)[:, 1]          # P(manmade)

        img_idxs.append(b_img_idx)
        true_labs.append(t_labels)
        hum_ch.append(h_choices)
        pred_ch.append(choices)
        true_rt.append(h_rts)
        pred_rt.append(med_rt)
        prob_vals.extend(b_probs)

        for i, prob in enumerate(b_probs):
            rt4 = rt_stack[:, i].tolist()                      # 4 RT values
            rt_samples_by_prob[prob].extend(rt4)
            human_rts_by_prob[prob].append(float(h_rts[i]))
            trial_pred_rts_by_prob[prob].append(float(med_rt[i]))
            pred_probs_by_prob[prob].append(float(probs1[i]))
            model_info_by_prob[prob]["pred_choices"].append(int(choices[i]))
            model_info_by_prob[prob]["true_labels"].append(int(t_labels[i]))
            model_info_by_prob[prob]["human_choices"].append(int(h_choices[i]))

    return {
        "image_indices": torch.cat(img_idxs).numpy(),
        "true_labels":   torch.cat(true_labs).numpy(),
        "human_choices": torch.cat(hum_ch).numpy(),
        "pred_choices":  torch.cat(pred_ch).numpy(),
        "true_rts":      torch.cat(true_rt).numpy(),
        "pred_rts":      torch.cat(pred_rt).numpy(),
        "prob_values":   np.array(prob_vals, dtype=float),
        "levels":        [f"{p:.1f}" for p in prob_vals],
        "rt_samples_by_prob":     {k: np.array(v) for k, v in rt_samples_by_prob.items()},
        "human_rts_by_prob":      {k: np.array(v) for k, v in human_rts_by_prob.items()},
        "trial_pred_rts_by_prob": {k: np.array(v) for k, v in trial_pred_rts_by_prob.items()},
        "pred_probs_by_prob":     {k: np.array(v) for k, v in pred_probs_by_prob.items()},
        "model_info_by_prob":     dict(model_info_by_prob),
    }


def save_csvs(metrics: dict, output_dir: str, tag: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for prob, p in metrics["per_prob"].items():
        nc_p = p.get("nc", {})
        rows.append({
            "prob":             prob,
            "n_trials":         p["n_trials"],
            "human_acc":        p["human_acc"],
            "model_acc":        p["model_acc"],
            "nc_acc":           nc_p.get("acc",    float("nan")),
            "human_median_rt":  p["human_median_rt"],
            "model_median_rt":  p["model_median_rt"],
            "w1_ms":            p["w1_ms"],
            "nll_model":        p["nll_model"],
            "nll_hh":           p["nll_hh"],
            "nc_nll":           nc_p.get("nll",   float("nan")),
            "pearson_r":        p.get("pearson_r",  float("nan")),
            "spearman_r":       p.get("spearman_r", float("nan")),
            "ccc_rt":           p.get("ccc_rt",     float("nan")),
            "auc":              p.get("auc",         float("nan")),
            "kappa":            p.get("kappa",       float("nan")),
            "nc_kappa":         nc_p.get("kappa",   float("nan")),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, f"rtify_metrics_per_prob{tag}.csv"), index=False
    )

    glb = metrics["global"]
    nc  = glb.get("nc", {})
    pd.DataFrame([{
        "acc":         glb["acc"],
        "human_acc":   glb["human_acc"],
        "r_acc":       glb["r_acc"],
        "r_rt":        glb["r_rt"],
        "ccc_rt":      glb["ccc_rt"],
        "hh_r_rt":     glb["hh_r_rt"],
        "mean_nll":    glb["mean_nll"],
        "hh_mean_nll": glb["hh_mean_nll"],
        "mean_w1_ms":  glb["mean_w1_ms"],
        "mean_auc":    glb["mean_auc"],
        "mean_kappa":  glb["mean_kappa"],
        "nc_acc":      nc.get("acc",    float("nan")),
        "nc_nll":      nc.get("nll",    float("nan")),
        "nc_r_rt":     nc.get("r_rt",   float("nan")),
        "nc_r_acc":    nc.get("r_acc",  float("nan")),
        "nc_kappa":    nc.get("kappa",  float("nan")),
    }]).to_csv(os.path.join(output_dir, f"rtify_metrics_global{tag}.csv"), index=False)

    print(f"\nCSVs saved to {display_path(output_dir)}/")


def print_paper_table(metrics: dict) -> None:
    glb = metrics["global"]
    nc  = glb.get("nc", {})

    def _f(v, fmt=".3f"):
        return "—" if not np.isfinite(float(v)) else format(float(v), fmt)

    W = 100
    print("\n" + "=" * W)
    print("PAPER TABLE — RTify (CNN+GRU) on SC task")
    print("=" * W)
    header = (
        f"{'Model':22s}  "
        f"{'Acc(%)':>7}  "
        f"{'AUC':>6}  "
        f"{'Kappa':>7}  "
        f"{'r_acc':>6}  "
        f"{'r_RT':>6}  "
        f"{'CCC_RT':>7}  "
        f"{'NLL↓':>7}"
    )
    print(header)
    print("-" * W)
    print(
        f"{'Human (NC)':22s}  "
        f"{_f(nc.get('acc', float('nan')), '.1f'):>7}  "
        f"{_f(nc.get('auc', float('nan'))):>6}  "
        f"{_f(nc.get('kappa', float('nan'))):>7}  "
        f"{_f(nc.get('r_acc', float('nan'))):>6}  "
        f"{_f(nc.get('r_rt',  float('nan'))):>6}  "
        f"{'—':>7}  "
        f"{_f(glb['hh_mean_nll']):>7}"
    )
    print("-" * W)
    print(
        f"{'RTify (CNN+GRU)':22s}  "
        f"{glb['acc']*100:>7.1f}  "
        f"{_f(glb['mean_auc']):>6}  "
        f"{_f(glb['mean_kappa']):>7}  "
        f"{_f(glb['r_acc']):>6}  "
        f"{_f(glb['r_rt']):>6}  "
        f"{_f(glb['ccc_rt']):>7}  "
        f"{_f(glb['mean_nll']):>7}"
    )
    print("=" * W)

    print("\nPer-prob breakdown:")
    hdr2 = (
        f"  {'Prob':>5}  {'N':>5}  "
        f"{'Acc_H':>6}  {'Acc_M':>6}  "
        f"{'NC_Acc':>7}  "
        f"{'RT_H ms':>8}  {'RT_M ms':>8}  "
        f"{'W1 ms':>7}  {'NLL_M':>6}  {'NLL_HH':>7}  "
        f"{'r_RT':>6}  {'AUC':>6}  {'Kappa':>7}"
    )
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for prob in PROB_LIST:
        if prob not in metrics["per_prob"]:
            continue
        p    = metrics["per_prob"][prob]
        nc_p = p.get("nc", {})
        print(
            f"  {prob:5.1f}  {p['n_trials']:>5d}  "
            f"{p['human_acc']:>6.3f}  {p['model_acc']:>6.3f}  "
            f"{_f(nc_p.get('acc', float('nan'))):>7}  "
            f"{p['human_median_rt']*1000:>8.1f}  "
            f"{p['model_median_rt']*1000:>8.1f}  "
            f"{_f(p['w1_ms']):>7}  "
            f"{_f(p['nll_model']):>6}  "
            f"{_f(p['nll_hh']):>7}  "
            f"{_f(p.get('pearson_r', float('nan'))):>6}  "
            f"{_f(p.get('auc',   float('nan'))):>6}  "
            f"{_f(p.get('kappa', float('nan'))):>7}"
        )
    print("=" * W)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Configuration ─────────────────────────────────────────────────────────
    # Data
    TRAIN_DIR  = os.environ.get("SC_SUN_TRAIN_DIR", os.path.join(REPO_ROOT, "datasets", "sc", "sun_filtered", "train"))
    VAL_DIR    = os.environ.get("SC_SUN_VAL_DIR", os.path.join(REPO_ROOT, "datasets", "sc", "sun_filtered", "val"))
    BATCH_SIZE = 128

    # Phase 1 — CNN pre-training
    CNN_EPOCHS = 30
    CNN_LR     = 1e-3

    # Phase 2 — RTify accumulator
    T_STEPS    = 40
    RT_EPOCHS  = 50
    RT_LR      = 1e-4
    RT_WEIGHT  = 1.38 #2.0
    SEEDS      = [0, 1, 2, 3]

    # Misc
    RANDOM_SEED = 42
    OUTPUT_DIR  = os.path.join(REPO_ROOT, "outputs", "sc")
    MODEL_DIR   = os.path.join(REPO_ROOT, "competitors", "weights", "sc")

    TRAIN_CNN         = True   # set False to skip Phase 1
    TRAIN_ACCUMULATORS = True  # set False to skip Phase 2

    # Trained weights are saved with the _REFRESH suffix next to the released
    # ones (rtify_cnn_REFRESH.pth, rtify_accumulator_seed<k>_REFRESH.pth), which
    # are never overwritten; the metrics CSVs of such a run carry the suffix
    # too. With TRAIN_CNN = False the released CNN is used.

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cnn_save_path = str(refresh_path(os.path.join(MODEL_DIR, "rtify_cnn.pth"),
                                     TRAIN_CNN))
    FEAT_DIM   = 128
    HIDDEN_DIM = 128

    # ── PHASE 1 — SUN_filtered data loaders ───────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1 — CNN backbone pre-training on SUN_filtered")
    print("=" * 70)

    sun_train = SUNFilteredDataset(TRAIN_DIR)
    sun_val   = SUNFilteredDataset(VAL_DIR)

    sun_train_loader = DataLoader(
        sun_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    sun_val_loader = DataLoader(
        sun_val, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )

    encoder = CNNEncoder(feat_dim=FEAT_DIM).to(device)

    if TRAIN_CNN:
        train_cnn_phase1(
            encoder=encoder,
            train_loader=sun_train_loader,
            val_loader=sun_val_loader,
            n_epochs=CNN_EPOCHS,
            lr=CNN_LR,
            device=device,
            save_path=cnn_save_path,
        )

    print(f"\nLoading best CNN from {display_path(cnn_save_path)}")
    encoder.load_state_dict(torch.load(cnn_save_path, map_location=device))
    encoder.eval()

    # Freeze encoder — no gradients will flow through it in Phase 2
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ── experiment-1 splits (same seed as NRM) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("DATA — experiment-1 splits")
    print("=" * 70)

    sc_dataset = SCDataset(
        csv_path=os.path.join(REPO_ROOT, "datasets", "sc", "experiment_1.csv"),
        image_dir=os.path.join(REPO_ROOT, "datasets", "sc", "stimuli"),
    )
    _, val_ds, test_ds = sc_dataset.train_val_test_split(
        val_size=0.10, test_size=0.20, seed=RANDOM_SEED,
    )

    val_img_ds  = SCImageDataset(val_ds)
    test_img_ds = SCImageDataset(test_ds)

    # RT normalisation from val split (following train_mnist_behavior.py)
    val_rts = val_ds.df["rt"].dropna().values.astype(np.float32)
    rt_min   = float(val_rts.min())
    rt_range = float(val_rts.max() - val_rts.min())
    if rt_range < 1e-6:
        rt_range = 1.0
    print(f"RT normalisation (from val split): min={rt_min:.4f}s  range={rt_range:.4f}s")

    # ── PHASE 2 — Accumulator training (4 seeds) ───────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2 — RTify accumulator training on experiment-1 val split")
    print("=" * 70)

    accumulators: list[Accumulator] = []

    for seed in SEEDS:
        acc_path = str(refresh_path(
            os.path.join(MODEL_DIR, f"rtify_accumulator_seed{seed}.pth"),
            TRAIN_CNN or TRAIN_ACCUMULATORS))
        accumulator = Accumulator(
            feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM,
            n_classes=2, time_steps=T_STEPS, sigma=2.0,
        ).to(device)

        if TRAIN_ACCUMULATORS:
            print(f"\n  Seed {seed} — training accumulator...")
            model_phase2 = RTifyModel(encoder=encoder, accumulator=accumulator).to(device)
            train_accumulator(
                model=model_phase2,
                val_img_ds=val_img_ds,
                batch_size=BATCH_SIZE,
                seed=seed,
                n_epochs=RT_EPOCHS,
                lr=RT_LR,
                rt_weight=RT_WEIGHT,
                rt_min=rt_min,
                rt_range=rt_range,
                device=device,
                save_path=acc_path,
            )

        accumulator.load_state_dict(torch.load(acc_path, map_location=device))
        accumulator.eval()
        accumulators.append(accumulator)

    # ── EVALUATION — experiment-1 test split ───────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION — experiment-1 test split")
    print("=" * 70)

    models = [
        RTifyModel(encoder=encoder, accumulator=acc).to(device)
        for acc in accumulators
    ]
    for m in models:
        m.eval()

    data = build_eval_data(
        models=models,
        test_img_ds=test_img_ds,
        batch_size=BATCH_SIZE,
        device=device,
        rt_min=rt_min,
        rt_range=rt_range,
    )

    evaluator = SCEvaluator(model=None, device=device)
    metrics   = evaluator.compute_metrics(data)

    save_csvs(metrics, OUTPUT_DIR, tag=refresh_tag(TRAIN_CNN or TRAIN_ACCUMULATORS))
    print_paper_table(metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
