"""
competitors/scan/train_sat.py

Adapts the original SATBench SCAN code to train and evaluate
SCAN (Scalable Neural Network) on the same data (identical 70/10/20 split)
used for NRM, instead of the standard ImageNet dataset.

Identical workflow to the original SCAN code:
  1. train.py  → train the model, save checkpoint
  2. test.py   → load checkpoint, produce predictions for each exit
  3. Analysis  → produce SAT curves compatible with NRM / CNet / ConvRNN

The only difference from the original code is the data source:
  - SATDataset (NRM pipeline) instead of ImageNet folders
  - same remove_rt_outliers([-100,+100]), tutorial block removal, split seed=42

Architectural notes (sresnet.py):
  - ResNet-18 with 4 early exits (one per layer1-4) + 1 final ensemble
  - Natively 5 exits: no need to select a subset
  - Direct mapping: exit0(layer1/shallow)→500ms, ..., exit4(ensemble)→1500ms
  - Train from scratch with: CE(deepest) + KD*softCE(shallow←deepest) +
    (1-KD)*CE(shallow) + 5e-7*feature_loss  [lambda_KD=0.5]
  - SGD(lr=0.1, momentum=0.9, wd=5e-4), 50 epochs, no LR decay
    (the 75/130/180 milestones are never reached in 50 epochs)

Usage:
  python competitors/scan/train_sat.py --perturb color-gray [--eval_only]
  python competitors/scan/train_sat.py --perturb color-blur [--eval_only]
  python competitors/scan/train_sat.py --perturb gray-noise [--eval_only]
"""

import argparse
import shutil
import tempfile
import collections
import csv
import glob as _glob
import importlib.util
import json
import os
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from scripts.results_io import display_path


# ── paths ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
SCAN_ROOT  = _SCRIPT_DIR
IMAGE_DIR = os.environ.get("SAT_IMAGE_DIR", os.path.join(REPO_ROOT, "datasets", "sat", "data"))
# IMAGE_DIR can also be set via SAT_IMAGE_DIR environment variable

# Put the vendored SCAN code on the path to import sresnet
if SCAN_ROOT not in sys.path:
    sys.path.insert(0, SCAN_ROOT)

PERTURB_TO_FILE = {
    "color-gray": os.path.join(REPO_ROOT, "datasets/sat", "color-gray.txt"),
    "color-blur": os.path.join(REPO_ROOT, "datasets/sat", "color-blur.txt"),
    "gray-noise": os.path.join(REPO_ROOT, "datasets/sat", "gray-noise.txt"),
}

# ── released checkpoints ──────────────────────────────────────────────────────
# Loaded by file path rather than imported as `competitors.released_weights`:
# these scripts put their own vendored directories on sys.path, so a package
# import would depend on the order that happened in.
_rw_spec = importlib.util.spec_from_file_location(
    "competitors_released_weights",
    os.path.join(REPO_ROOT, "competitors", "released_weights.py"),
)
released_weights = importlib.util.module_from_spec(_rw_spec)
_rw_spec.loader.exec_module(released_weights)
resolve_sat_checkpoint = released_weights.resolve_sat_checkpoint
keep_final_checkpoint = released_weights.keep_final_checkpoint

# ── SAT benchmark constants ───────────────────────────────────────────────────
# SCAN has 5 native exits → 1:1 mapping with the 5 human beeps
# (from test.py: outputs.reverse() → [out1, out2, out3, out4] shallowest→deepest + ensemble)
_BEEP_MS_TO_EXIT = {500: 0, 900: 1, 1100: 2, 1300: 3, 1500: 4}
_BEEP_TIMES_S    = sorted(ms / 1000.0 for ms in _BEEP_MS_TO_EXIT)
_N_CHOICES       = 16
_CHANCE          = 1.0 / _N_CHOICES
_N_EXITS         = 5    # 4 early-exit + 1 ensemble


# =============================================================================
# 1. Dataset — same pipeline as scripts/train_nrm_sat.py
# =============================================================================

# Per-model plots (the seven PNGs) and the per-model .tex/.csv are off by
# default: only sat_level_curves.json and satbench_metrics_full.json feed
# pipelines/sat.ipynb. `--diagnostics` turns them on.
WRITE_DIAGNOSTICS = False


class _SATImageDataset(Dataset):
    """
    Wrap a SATDataset subset (post-split) so it yields (image_tensor, label)
    in the format expected by SCAN (224×224, ToTensor without extra normalisation,
    faithful to SCAN's create_transforms without extraaug).
    """
    _TRAIN_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # SCAN does not use ImageNet normalisation by default (only with extraaug)
        # → ToTensor maps values to [0,1], as in the original
    ])
    _EVAL_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def __init__(self, sat_subset, train: bool):
        self.df  = sat_subset.df.reset_index(drop=True)
        self.tfm = self._TRAIN_TFM if train else self._EVAL_TFM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.tfm(image), int(row["choice"])


def _build_loader(sat_subset, train: bool, batch_size: int, num_workers: int):
    ds = _SATImageDataset(sat_subset, train=train)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def load_sat_splits(file_path: str):
    """Same pipeline as scripts/train_nrm_sat.py."""
    from datasets.sat.loader import SATDataset

    print("\n" + "=" * 70)
    print("LOADING SAT DATA  (pipeline NRM)")
    print("=" * 70)

    dataset = SATDataset(file_path=file_path, image_dir=IMAGE_DIR, load_images=True)
    dataset.remove_rt_outliers(outlier_range_ms=[-100, 100])

    exp_mask = dataset.df["block"] != "cattut"
    n_rem    = int((~exp_mask).sum())
    if n_rem:
        dataset.df = dataset.df[exp_mask].reset_index(drop=True)
        print(f"  Removed {n_rem} tutorial trials ({len(dataset.df)} experimental).")

    train_ds, val_ds, test_ds = dataset.train_val_test_split(
        val_size=0.1, test_size=0.2, seed=42, shuffle=True,
    )
    print(f"  Split — Train: {len(train_ds.df)}, Val: {len(val_ds.df)}, "
          f"Test: {len(test_ds.df)}")
    return train_ds, val_ds, test_ds


# =============================================================================
# 2. SCAN loss — identical to the original train.py
# =============================================================================

def _soft_cross_entropy(outputs: torch.Tensor, targets: torch.Tensor,
                         temperature: float = 3.0) -> torch.Tensor:
    """
    Self-distillation loss (soft cross-entropy with temperature scaling).
    Identical to CrossEntropy() in SCAN's original train.py / test.py.
    """
    log_softmax_out = F.log_softmax(outputs / temperature, dim=1)
    softmax_tgt     = F.softmax(targets / temperature, dim=1)
    return -(log_softmax_out * softmax_tgt).sum(dim=1).mean()


def scan_loss(
        outputs: list[torch.Tensor],
        labels: torch.Tensor,
        feature_loss: torch.Tensor,
        lambda_kd: float = 0.5,
        criterion: nn.Module | None = None,
) -> torch.Tensor:
    """
    Full SCAN loss:
      - CE on the deepest classifier (outputs[0] = out4, deepest)
      - For each shallower exit: KD * softCE(shallow, teacher) + (1-KD) * CE(shallow, labels)
      - feature alignment loss: feature_loss * 5e-7

    Note: outputs = [out4, out3, out2, out1] (from sresnet.forward), deepest first.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    loss = torch.FloatTensor([0.0]).to(labels.device)

    # deepest classifier: hard CE only
    loss = loss + criterion(outputs[0], labels)

    # teacher = output of the deepest classifier (detached)
    teacher = outputs[0].detach()
    teacher.requires_grad_(False)

    for idx in range(1, len(outputs)):
        loss = loss + _soft_cross_entropy(outputs[idx], teacher) * lambda_kd * 9
        loss = loss + criterion(outputs[idx], labels) * (1.0 - lambda_kd)

    if lambda_kd != 0.0:
        loss = loss + feature_loss * 5e-7

    return loss


# =============================================================================
# 3. Training
# =============================================================================

def train_scan(
        net: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        n_epochs: int = 50,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        lambda_kd: float = 0.5,
        lr_milestones: list[int] | None = None,
        save_dir: str | None = None,
        save_freq: int = 10,
        eval_freq: int = 5,
        debug: bool = False,
) -> dict:
    """
    Train SCAN from scratch.
    Loss: scan_loss (CE deepest + KD soft-CE shallow ← deepest + feature align).
    Optimiser: SGD(lr, momentum, weight_decay).
    LR schedule: /10 at the given milestones (default: [75,130,180], never reached in 50 ep).
    """
    if lr_milestones is None:
        lr_milestones = [75, 130, 180]   # as in SCAN: never reached in 50 epochs

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.SGD(net.parameters(), lr=lr,
                            momentum=momentum, weight_decay=weight_decay)

    metrics: dict = {"train": collections.defaultdict(list),
                     "val":   collections.defaultdict(list)}

    print(f"\n{'='*70}\nTRAINING SCAN  ({n_epochs} ep, lr={lr}, λ_kd={lambda_kd})\n{'='*70}")

    for epoch in range(n_epochs):
        # Original LR schedule (unchanged from train.py)
        if epoch in lr_milestones:
            for pg in optimizer.param_groups:
                pg["lr"] /= 10
            print(f"  [LR] Epoch {epoch}: lr /= 10")

        net.train()
        sum_loss   = 0.0
        total      = 0
        correct_by_exit = [0] * _N_EXITS   # [exit0=shallow, ..., exit3=deep, exit4=ensemble]

        for batch_i, (images, labels) in enumerate(train_loader):
            if debug and batch_i > 1:
                break

            images  = images.to(device)
            labels  = labels.to(device)

            # forward: outputs=[out4,out3,out2,out1] (deepest→shallowest), feature_loss
            outputs, feature_loss = net(images)

            # ensemble for accuracy logging (not part of the training loss)
            with torch.no_grad():
                ensemble_logits = sum(outputs) / len(outputs)
                ensemble_logits.detach_()

            loss = scan_loss(outputs, labels, feature_loss, lambda_kd, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            total    += labels.size(0)

            # accuracy per exit (order: shallowest→deepest + ensemble)
            reversed_outputs = list(reversed(outputs))   # [out1, out2, out3, out4]
            for e, logits in enumerate(reversed_outputs):
                preds = logits.argmax(dim=1)
                correct_by_exit[e] += preds.eq(labels).sum().item()
            ens_preds = ensemble_logits.argmax(dim=1)
            correct_by_exit[4] += ens_preds.eq(labels).sum().item()

            acc_str = "  ".join(
                f"e{e}:{100*correct_by_exit[e]/total:.1f}%" for e in range(_N_EXITS)
            )
            sys.stdout.write(
                f"\r  Ep {epoch+1}/{n_epochs}  b{batch_i+1}/{len(train_loader)}"
                f"  loss={sum_loss/(batch_i+1):.4f}  {acc_str}"
            )
            sys.stdout.flush()
        print()

        accs_train = [100 * c / total for c in correct_by_exit]
        metrics["train"]["loss"].append(sum_loss / len(train_loader))
        metrics["train"]["accs"].append(accs_train)

        # validation
        if epoch % eval_freq == 0 or epoch == n_epochs - 1:
            val_loss, val_accs = _eval_epoch(net, val_loader, device,
                                              lambda_kd, debug)
            metrics["val"]["loss"].append((epoch, val_loss))
            metrics["val"]["accs"].append((epoch, val_accs))
            acc_str = "  ".join(
                f"e{e}:{val_accs[e]:.1f}%" for e in range(_N_EXITS)
            )
            print(f"  Val  loss={val_loss:.4f}  {acc_str}")

        # checkpoint
        if save_dir and (epoch % save_freq == 0 or epoch == n_epochs - 1):
            ckpt_path = os.path.join(save_dir, f"ckpt__epoch_{epoch:04d}.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": net.state_dict(),
                "optim_state": optimizer.state_dict(),
            }, ckpt_path)
            if not debug:
                print(f"  Checkpoint saved: {display_path(ckpt_path)}")

    print("\nTraining SCAN completed.")
    return metrics


def _eval_epoch(
        net: nn.Module,
        loader: DataLoader,
        device: torch.device,
        lambda_kd: float = 0.5,
        debug: bool = False,
) -> tuple[float, list[float]]:
    """Loss and per-exit accuracies over a dataloader."""
    criterion = nn.CrossEntropyLoss()
    net.eval()
    sum_loss   = 0.0
    total      = 0
    correct_by_exit = [0] * _N_EXITS

    with torch.no_grad():
        for batch_i, (images, labels) in enumerate(loader):
            if debug and batch_i > 1:
                break
            images  = images.to(device)
            labels  = labels.to(device)
            outputs, feature_loss = net(images)
            ensemble_logits = sum(outputs) / len(outputs)
            loss = scan_loss(outputs, labels, feature_loss, lambda_kd, criterion)
            sum_loss += loss.item()
            total    += labels.size(0)

            reversed_outputs = list(reversed(outputs))
            for e, logits in enumerate(reversed_outputs):
                correct_by_exit[e] += logits.argmax(dim=1).eq(labels).sum().item()
            correct_by_exit[4] += ensemble_logits.argmax(dim=1).eq(labels).sum().item()

    accs = [100.0 * c / total for c in correct_by_exit]
    return sum_loss / len(loader), accs


# =============================================================================
# 4. Evaluation — produces output_reps in the shared format (T=5)
# =============================================================================

def evaluate_scan(
        net: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
) -> dict:
    """
    Run the forward pass over the test set and build output_reps with 5 "exits":
        exit 0 → out1 (shallowest, layer1) → beep 500ms
        exit 1 → out2 (layer2)             → beep 900ms
        exit 2 → out3 (layer3)             → beep 1100ms
        exit 3 → out4 (deepest, layer4)    → beep 1300ms
        exit 4 → ensemble (mean of all 4)  → beep 1500ms

    Faithful to the order of test.py: after outputs.reverse() and F.softmax for each exit.

    Returns dict con:
        predictions: LongTensor (5, N)
        correct:     BoolTensor (5, N)
        target:      LongTensor (N,)
        softmax:     FloatTensor (5, N, 16)
    """
    all_softmax = [[] for _ in range(_N_EXITS)]
    all_targets = []

    print(f"\n{'='*70}\nEVALUATION SCAN  (test set)\n{'='*70}")

    net.eval()
    with torch.no_grad():
        for batch_i, (images, labels) in enumerate(test_loader):
            images = images.to(device)

            outputs, _ = net(images)           # [out4, out3, out2, out1]
            ensemble   = sum(outputs) / len(outputs)

            # test.py: reverse → [out1, out2, out3, out4] → softmax
            outputs.reverse()
            for e, logits in enumerate(outputs):
                all_softmax[e].append(F.softmax(logits, dim=-1).cpu())
            # ensemble (index 4)
            all_softmax[4].append(F.softmax(ensemble, dim=-1).cpu())

            all_targets.append(labels)
            sys.stdout.write(f"\r  Batch {batch_i+1}/{len(test_loader)}")
            sys.stdout.flush()
    print()

    target_t  = torch.cat(all_targets, dim=0)                       # (N,)
    softmax_t = torch.stack(
        [torch.cat(all_softmax[e], dim=0) for e in range(_N_EXITS)],
        dim=0,
    )                                                                 # (5, N, 16)
    preds_t   = softmax_t.argmax(dim=-1)                            # (5, N)
    correct_t = preds_t == target_t.unsqueeze(0)                    # (5, N)

    print("\nAccuracy per exit:")
    exit_labels = ["shallow(e1→500ms)", "e2→900ms", "e3→1100ms",
                   "deep(e4→1300ms)", "ensemble→1500ms"]
    for e in range(_N_EXITS):
        acc = correct_t[e].float().mean().item() * 100
        print(f"  exit {e} [{exit_labels[e]}]: {acc:.2f}%")

    return {
        "predictions": preds_t,
        "correct":     correct_t,
        "target":      target_t,
        "softmax":     softmax_t,
    }


# =============================================================================
# 5. SAT benchmark analysis — identical to CNet/ConvRNN, labelled "SCAN"
# =============================================================================

def _build_human_data_dict(test_ds) -> dict:
    from datasets.sat.loader import CATEGORIES as _CATS
    df         = test_ds.df.reset_index(drop=True)
    cat_to_idx = {cat: i for i, cat in enumerate(_CATS)}

    def _resp_to_idx(r):
        if isinstance(r, str):
            return cat_to_idx.get(r.strip().lower(), -1)
        return -1

    return {
        "levels":        df["level_str"].tolist(),
        "image_indices": df["image_index"].values.astype(int),
        "human_choices": np.array([_resp_to_idx(r) for r in df["response"]], dtype=int),
        "true_labels":   df["choice"].values.astype(int),
        "true_rts":      df["duration"].values.astype(float) / 1000.0,
        "time_limits":   df["time_limit"].values.astype(float) / 1000.0,
        "srids":         df["srid"].tolist(),
    }


def _make_per_beep_entry(beep_s: float, per_img_acc: np.ndarray,
                          per_img_rt: np.ndarray | None = None) -> dict:
    n  = len(per_img_acc)
    rt = per_img_rt if per_img_rt is not None else np.full(n, beep_s)
    return {
        "beep_s":        beep_s,
        "rt_mean":       float(rt.mean()),
        "rt_std":        float(rt.std())          if n > 1 else 0.0,
        "rt_sem":        float(rt.std() / n**0.5) if n > 1 else 0.0,
        "acc_mean":      float(per_img_acc.mean()),
        "acc_std":       float(per_img_acc.std())          if n > 1 else 0.0,
        "acc_sem":       float(per_img_acc.std() / n**0.5) if n > 1 else 0.0,
        "n_images":      n,
        "n_trials":      n,
        "per_image_rt":  rt.tolist(),
        "per_image_acc": per_img_acc.tolist(),
    }


def _build_model_curves(test_ds, output_reps: dict) -> dict:
    """Build model_curves for scripts/eval_sat.py from the SCAN predictions."""
    from scripts.eval_sat import infer_scenario, _make_curves_dict
    from collections import defaultdict as _dd

    df             = test_ds.df.reset_index(drop=True)
    N              = len(df)
    correct        = output_reps["correct"]           # (5, N)
    time_limits_ms = df["time_limit"].values.astype(int)
    true_labels    = df["choice"].values.astype(int)
    levels_list    = df["level_str"].tolist()
    image_indices  = df["image_index"].values.astype(int)
    sat_mask       = time_limits_ms > 0

    assert correct.shape[1] == N

    scenario, ordered_levels, display, colors = infer_scenario(levels_list)

    raw = {
        lv: {cat: _dd(lambda: _dd(list)) for cat in range(_N_CHOICES)}
        for lv in ordered_levels
    }

    for i in range(N):
        if not sat_mask[i]:
            continue
        lv = str(levels_list[i])
        if lv not in ordered_levels:
            continue
        tl_ms = int(time_limits_ms[i])
        if tl_ms not in _BEEP_MS_TO_EXIT:
            continue
        exit_t = _BEEP_MS_TO_EXIT[tl_ms]
        beep_s = tl_ms / 1000.0
        img_id = int(image_indices[i])
        cat    = int(true_labels[i])
        corr   = float(correct[exit_t, i].item())
        raw[lv][cat][beep_s][img_id].append(corr)

    overall_by_level = {}
    by_cat_by_level  = {}

    for lv in ordered_levels:
        by_cat_by_level[lv] = {}
        all_beep_acc: dict[float, list] = _dd(list)

        for cat in range(_N_CHOICES):
            cat_beep = {}
            for beep_s in _BEEP_TIMES_S:
                img_dict = dict(raw[lv][cat].get(beep_s, {}))
                if not img_dict:
                    continue
                per_img = np.array([np.mean(v) for v in img_dict.values()])
                cat_beep[beep_s] = _make_per_beep_entry(beep_s, per_img)
                all_beep_acc[beep_s].extend(per_img.tolist())
            if cat_beep:
                by_cat_by_level[lv][cat] = _make_curves_dict(cat_beep)

        overall_beep = {}
        for beep_s in _BEEP_TIMES_S:
            acc_list = all_beep_acc.get(beep_s, [])
            if not acc_list:
                continue
            overall_beep[beep_s] = _make_per_beep_entry(beep_s, np.array(acc_list))
        overall_by_level[lv] = _make_curves_dict(overall_beep)

    return {
        "scenario":       scenario,
        "ordered_levels": ordered_levels,
        "display_labels": display,
        "colors":         colors,
        "overall":        overall_by_level,
        "by_category":    by_cat_by_level,
    }


def _build_pooled_curves(test_ds, output_reps: dict) -> tuple:
    """Human and SCAN curves pooled over every level."""
    from scripts.eval_sat import _make_curves_dict
    from collections import defaultdict as _dd

    df             = test_ds.df.reset_index(drop=True)
    N              = len(df)
    correct        = output_reps["correct"]
    time_limits_ms = df["time_limit"].values.astype(int)
    true_labels    = df["choice"].values.astype(int)
    image_indices  = df["image_index"].values.astype(int)
    sat_mask       = time_limits_ms > 0

    from datasets.sat.loader import CATEGORIES as _CATS
    cat_to_idx    = {c: i for i, c in enumerate(_CATS)}
    human_choices = np.array(
        [cat_to_idx.get(str(r).strip().lower(), -1) if isinstance(r, str) else -1
         for r in df["response"]], dtype=int,
    )
    true_rts_ms = df["duration"].values.astype(float)

    h_raw: dict = {cat: _dd(lambda: _dd(lambda: {"rts": [], "correct": []}))
                   for cat in range(_N_CHOICES)}
    c_raw: dict = {cat: _dd(lambda: _dd(list)) for cat in range(_N_CHOICES)}

    for i in range(N):
        if not sat_mask[i]:
            continue
        tl_ms = int(time_limits_ms[i])
        if tl_ms not in _BEEP_MS_TO_EXIT:
            continue
        beep_s = tl_ms / 1000.0
        img_id = int(image_indices[i])
        cat    = int(true_labels[i])
        h_ch   = int(human_choices[i])
        rt_s   = float(true_rts_ms[i]) / 1000.0
        h_raw[cat][beep_s][img_id]["rts"].append(rt_s)
        h_raw[cat][beep_s][img_id]["correct"].append(int(h_ch == cat))
        exit_t = _BEEP_MS_TO_EXIT[tl_ms]
        c_raw[cat][beep_s][img_id].append(float(correct[exit_t, i].item()))

    def _to_curves(raw_dict, use_rts=False):
        by_cat   = {}
        all_beep: dict[float, list] = _dd(list)
        all_rt:   dict[float, list] = _dd(list)
        for cat in range(_N_CHOICES):
            cat_beep = {}
            for beep_s in _BEEP_TIMES_S:
                img_dict = dict(raw_dict[cat].get(beep_s, {}))
                if not img_dict:
                    continue
                if use_rts:
                    per_img_acc = np.array([np.mean(v["correct"]) for v in img_dict.values()])
                    per_img_rt  = np.array([np.mean(v["rts"])     for v in img_dict.values()])
                else:
                    per_img_acc = np.array([np.mean(v) for v in img_dict.values()])
                    per_img_rt  = None
                cat_beep[beep_s] = _make_per_beep_entry(beep_s, per_img_acc, per_img_rt)
                all_beep[beep_s].extend(per_img_acc.tolist())
                if per_img_rt is not None:
                    all_rt[beep_s].extend(per_img_rt.tolist())
            if cat_beep:
                by_cat[cat] = _make_curves_dict(cat_beep)
        overall_beep = {}
        for beep_s in _BEEP_TIMES_S:
            acc_list = all_beep.get(beep_s, [])
            if not acc_list:
                continue
            rt_arr = np.array(all_rt[beep_s]) if all_rt.get(beep_s) else None
            overall_beep[beep_s] = _make_per_beep_entry(beep_s, np.array(acc_list), rt_arr)
        overall = _make_curves_dict(overall_beep)
        return by_cat, overall

    human_by_cat, human_overall = _to_curves(h_raw, use_rts=True)
    model_by_cat, model_overall = _to_curves(c_raw, use_rts=False)
    return human_by_cat, human_overall, model_by_cat, model_overall


def _plot_sat_curves_beep_vs_acc(
        human_by_cat: dict,
        model_by_cat: dict,
        human_overall: dict,
        model_overall: dict,
        save_path: str | os.PathLike,
        perturb: str = "",
) -> None:
    """17-subplot figure (overall + 16 categories) — SCAN vs Human."""
    from datasets.sat.loader import CATEGORIES as _CATS

    err_key  = "sem"
    n_cols   = 4
    n_panels = _N_CHOICES + 1
    n_rows   = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4.0, n_rows * 3.2),
                              squeeze=False)
    axes_flat = axes.flatten()

    def _plot_one(ax, h_data, m_data, title):
        chance_pct = _CHANCE * 100.0
        for data, color, lbl, ls, mk in [
            (h_data, "#4472C4", "Human", "-",  "o"),
            (m_data, "#FF0000", "SCAN",  "--", "D"),
        ]:
            if data is None:
                continue
            beeps = data["beeps_s"]
            pd_   = data["per_beep"]
            xs    = np.array(beeps)
            acc   = np.array([pd_[b]["acc_mean"] for b in beeps]) * 100.0
            yerr  = np.array([pd_[b][f"acc_{err_key}"] for b in beeps]) * 100.0
            xs_f  = np.concatenate([[0.0], xs])
            acc_f = np.concatenate([[chance_pct], acc])
            ye_f  = np.concatenate([[0.0], yerr])
            ax.errorbar(xs_f, acc_f, yerr=ye_f, fmt=f"{mk}{ls}",
                        color=color, label=lbl,
                        capsize=2, linewidth=1.6, markersize=4)
        ax.axhline(chance_pct, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_ylim([0, 105])
        ax.set_title(title, fontsize=8, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    _plot_one(axes_flat[0], human_overall, model_overall, f"Overall [{perturb}]")
    axes_flat[0].legend(fontsize=7, loc="lower right")

    for cat_idx in range(_N_CHOICES):
        ax = axes_flat[cat_idx + 1]
        nm = _CATS[cat_idx] if cat_idx < len(_CATS) else str(cat_idx)
        _plot_one(ax, human_by_cat.get(cat_idx), model_by_cat.get(cat_idx), nm)

    for i in range(_N_CHOICES + 1, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.supxlabel("Beep time (s)", fontsize=9, y=0.01)
    fig.supylabel("Accuracy (%)",  fontsize=9, x=0.01)
    fig.suptitle(f"SAT Curves — SCAN vs Human  [{perturb}]",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SCAN] Saved {display_path(save_path)}")


def run_scan_sat_analysis(
        test_ds,
        output_reps: dict,
        perturb: str,
        save_dir: str | os.PathLike,
        ranking_key: str = "scan",
        refresh: bool = False,
) -> dict:
    """
    SAT benchmark pipeline for SCAN — produces the same figures and metrics
    as scripts/eval_sat.py (NRM), run_cnet_sat_analysis (CNet)
    and run_convrnn_sat_analysis (ConvRNN).

    Output (in save_dir):
      sat_curves_beep_vs_acc.png
      sat_level_beep_vs_acc.png
      sat_level_by_cat_beep_vs_acc.png
      satbench_fig4_rmse_boxplot.png
      satbench_fig5_corr_boxplot.png
      satbench_fig6_cat_barplot.png
      satbench_fig7_steepness.png
      sat_level_curves.json
      sat_level_metrics_table.tex
      sat_level_metrics_table.csv
    """
    sys.path.insert(0, REPO_ROOT)

    from scripts.eval_sat import (
        calculate_human_curves_by_level,
        save_satbench_metrics_full,
        save_curves_to_json,
        generate_latex_table,
        CATEGORIES as _CATS,
    )
    from metrics.relational_metrics import (
        _compute_per_observer_curves,
        compute_satbench_metrics,
    )
    from metrics.plots_sat import (
        plot_satbench_figures,
        plot_sat_by_level,
        plot_sat_by_level_and_category,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SAT BENCHMARK ANALYSIS  (SCAN — compatible with NRM)")
    print("=" * 70)

    print("  Extracting the human data from the test set...")
    data = _build_human_data_dict(test_ds)

    print("  Computing human curves per level...")
    human_curves = calculate_human_curves_by_level(data, n_choices=_N_CHOICES)
    scenario     = human_curves["scenario"]
    ordered_lvls = human_curves["ordered_levels"]
    print(f"  Scenario: {scenario}  |  Levels: {ordered_lvls}")

    if not ordered_lvls:
        print("  [WARN] No level found - SAT analysis skipped.")
        return {}

    print("  Building SCAN curves per level (from output_reps)...")
    model_curves = _build_model_curves(test_ds, output_reps)
    model_curves["scenario"] = scenario

    print("  Computing per-observer curves (srid)...")
    per_obs_curves = _compute_per_observer_curves(data, ordered_lvls,
                                                   n_choices=_N_CHOICES)
    n_obs = len(per_obs_curves) if per_obs_curves else 0
    print(f"  {n_obs} observers in the test set.")

    print("  Building pooled curves (human + SCAN)...")
    h_by_cat, h_overall, m_by_cat, m_overall = _build_pooled_curves(
        test_ds, output_reps
    )

    # JSON curve
    save_curves_to_json(human_curves, model_curves, save_dir / "sat_level_curves.json")

    # sat_curves_beep_vs_acc.png
    if WRITE_DIAGNOSTICS:
        _plot_sat_curves_beep_vs_acc(
            h_by_cat, m_by_cat, h_overall, m_overall,
            save_path=save_dir / "sat_curves_beep_vs_acc.png",
            perturb=perturb,
        )

    # sat_level_beep_vs_acc.png
    if WRITE_DIAGNOSTICS:
        plot_sat_by_level(
            human_curves, model_curves,
            x_mode="beep", use_sem=True,
            save_path=save_dir / "sat_level_beep_vs_acc.png",
        )

    # sat_level_by_cat_beep_vs_acc.png
    if WRITE_DIAGNOSTICS:
        plot_sat_by_level_and_category(
            human_curves, model_curves,
            x_mode="beep", use_sem=True, n_choices=_N_CHOICES,
            save_path=save_dir / "sat_level_by_cat_beep_vs_acc.png",
        )

    # SATBench metrics
    print("  Computing metrics (eRMSE, Spearman, steepness)...")
    metrics = compute_satbench_metrics(
        human_curves, model_curves, per_obs_curves, n_choices=_N_CHOICES
    )
    metrics["colors"] = human_curves["colors"]

    # Full per-observer metrics JSON
    save_satbench_metrics_full(metrics, save_dir)

    # Canonical Table-3 scalars, where pipelines/ranking.ipynb looks for them.
    from scripts.results_io import save_sat_metrics
    canonical = save_sat_metrics(perturb, ranking_key, metrics, refresh=refresh)
    print(f"  [SAT metrics] Canonical scalars saved to {display_path(canonical)}")

    # Figure 4-7
    if WRITE_DIAGNOSTICS:
        plot_satbench_figures(
            metrics, human_curves, model_curves,
            save_dir=save_dir,
            prefix="satbench",
            n_choices=_N_CHOICES,
        )

    # LaTeX table
    if WRITE_DIAGNOSTICS:
        latex_str  = generate_latex_table(metrics, scenario_name=f"{scenario} SCAN")
        latex_path = save_dir / "sat_level_metrics_table.tex"
        latex_path.write_text(latex_str)
        print(f"  [SCAN] Saved {display_path(latex_path)}")

        # CSV table
        csv_path = save_dir / "sat_level_metrics_table.csv"
        disp     = metrics.get("display_labels", {})
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["level", "display", "rmse", "spearman_r",
                             "steepness_human", "steepness_model"])
            for lv in ordered_lvls:
                row = metrics["overall"].get(lv, {})
                writer.writerow([
                    lv, disp.get(lv, lv),
                    row.get("rmse",            float("nan")),
                    row.get("spearman_r",      float("nan")),
                    row.get("steepness_human", float("nan")),
                    row.get("steepness_model", float("nan")),
                ])
            avg = metrics.get("mean_over_levels", {})
            writer.writerow([
                "Average", "Average",
                avg.get("rmse",            float("nan")),
                avg.get("spearman_r",      float("nan")),
                avg.get("steepness_human", float("nan")),
                avg.get("steepness_model", float("nan")),
            ])
        print(f"  [SCAN] Saved {display_path(csv_path)}")

        # Print accuracy per beep/level
        print("\n  SCAN accuracy per level/deadline:")
        for lv in ordered_lvls:
            lbl = human_curves["display_labels"].get(lv, lv)
            ov  = model_curves["overall"].get(lv)
            if ov is None:
                continue
            beep_str = "  ".join(
                f"{b:.1f}s→{ov['per_beep'][b]['acc_mean']*100:.1f}%"
                for b in ov["beeps_s"]
            )
            print(f"    [{lbl}]  {beep_str}")

        print("\nSAT benchmark SCAN completed.")
    return {"metrics": metrics, "save_dir": str(save_dir)}


# =============================================================================
# 6. Main script
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="SCAN ResNet-18 on the SAT data (NRM pipeline) - train + eval + analysis"
    )
    p.add_argument("--perturb", required=True,
                   choices=["color-gray", "color-blur", "gray-noise"])
    p.add_argument("--diagnostics", action="store_true",
                   help="Also write the per-model PNGs and the per-model "
                        ".tex/.csv. Off by default: pipelines/sat.ipynb "
                        "only reads sat_level_curves.json and "
                        "satbench_metrics_full.json.")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training and load the last checkpoint.")
    p.add_argument("--weights", default=None,
                   help="Explicit checkpoint. Without this flag the released "
                        "checkpoint in competitors/weights/sat/ is used.")
    p.add_argument("--refresh", action="store_true",
                   help="Write every output with the _REFRESH suffix "
                        "(outputs/sat/<task>/<model>_REFRESH/).")
    p.add_argument("--refresh_weights", action="store_true",
                   help="Evaluate the retrained "
                        "competitors/weights/sat/<model>_<perturbation>_REFRESH.pt "
                        "instead of the released checkpoint; implies --refresh.")
    # hyperparameters (paper appendix: 50 epochs, SGD lr=0.1, batch=128, lambda_KD=0.5)
    p.add_argument("--n_epochs",     type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--lambda_kd",    type=float, default=0.5,
                   help="Self-distillation loss coefficient.")
    p.add_argument("--depth",        type=int,   default=18,
                   choices=[18, 34, 50, 101, 152])
    p.add_argument("--eval_freq",    type=int,   default=5)
    p.add_argument("--save_freq",    type=int,   default=10)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--device",       type=int,   default=0)
    p.add_argument("--debug",        action="store_true", default=False,
                   help="Run only 2 batches per epoch (smoke test).")
    return p.parse_args()


def main():
    args = parse_args()
    global WRITE_DIAGNOSTICS
    WRITE_DIAGNOSTICS = bool(getattr(args, 'diagnostics', False))

    import sresnet as _sresnet

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device : {device}")
    print(f"Perturb: {args.perturb}")

    torch.manual_seed(100)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(100)
    np.random.seed(100)

    exp_name = f"scan_resnet{args.depth}_SAT_{args.perturb}"

    # ── output and scratch directories ────────────────────────────────────────
    # A training run, --refresh or --refresh_weights writes every output with
    # the _REFRESH suffix, so the reference outputs are never overwritten.
    sys.path.insert(0, REPO_ROOT)
    from scripts.results_io import sat_model_dir
    refresh = args.refresh or args.refresh_weights or not args.eval_only
    sat_save_dir = str(sat_model_dir(args.perturb, "scan", refresh=refresh))
    figs_dir = out_dir = sat_save_dir
    os.makedirs(sat_save_dir, exist_ok=True)
    # Periodic checkpoints and training logs go to a scratch directory that is
    # removed at the end; only the final checkpoint is kept, as
    # competitors/weights/sat/<model>_<perturbation>_REFRESH.pt.
    exp_root = tempfile.mkdtemp(prefix=f"{exp_name}_")
    ckpt_dir = os.path.join(exp_root, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    data_file = PERTURB_TO_FILE[args.perturb]
    train_ds, val_ds, test_ds = load_sat_splits(data_file)

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding SCAN ResNet-{args.depth} (num_classes={_N_CHOICES})...")
    depth_to_fn = {
        18: _sresnet.resnet18, 34: _sresnet.resnet34,
        50: _sresnet.resnet50, 101: _sresnet.resnet101, 152: _sresnet.resnet152,
    }
    net = depth_to_fn[args.depth](num_classes=_N_CHOICES, align="CONV").to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Total parameters: {n_params:,}")

    # ── training ──────────────────────────────────────────────────────────────
    if not args.eval_only:
        train_loader = _build_loader(train_ds, train=True,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers)
        val_loader   = _build_loader(val_ds, train=False,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers)

        train_metrics = train_scan(
            net          = net,
            train_loader = train_loader,
            val_loader   = val_loader,
            device       = device,
            n_epochs     = args.n_epochs,
            lr           = args.lr,
            momentum     = args.momentum,
            weight_decay = args.weight_decay,
            lambda_kd    = args.lambda_kd,
            save_dir     = ckpt_dir,
            save_freq    = args.save_freq,
            eval_freq    = args.eval_freq,
            debug        = args.debug,
        )

        # save config and metrics
        with open(os.path.join(exp_root, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        with open(os.path.join(exp_root, "train_metrics.json"), "w") as f:
            json.dump(
                {k: {kk: [float(x) for x in vv] if isinstance(vv[0], float) else vv
                     for kk, vv in v.items()}
                 for k, v in train_metrics.items()},
                f, indent=2, default=str,
            )

        keep_final_checkpoint(ckpt_dir, "scan", args.perturb,
                              pattern="ckpt__epoch_*.pt")

    # ── load checkpoint ─────────────────────────────────────────────────────
    # The checkpoint a training run just saved (or, with --refresh_weights, a
    # previous one), else the released weights in competitors/weights/sat/.
    # Both layouts are `{"epoch", "model_state", "optim_state"}`.
    selected_ckpt = resolve_sat_checkpoint(
        model="scan", perturb=args.perturb, explicit=args.weights,
        refresh_weights=args.refresh_weights or not args.eval_only,
    )
    ckpt_data = torch.load(selected_ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ckpt_data["model_state"])
    print(f"  Checkpoint epoch {ckpt_data.get('epoch', '?')} loaded.")

    # ── evaluation ────────────────────────────────────────────────────────────
    test_loader = _build_loader(test_ds, train=False,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    output_reps = evaluate_scan(net, test_loader, device)

    out_rep_path = os.path.join(out_dir, "output_representations__test__OSD.pt")
    torch.save(output_reps, out_rep_path)
    print(f"Saved: {display_path(out_rep_path)}")

    # ── SAT curve per exit (as in the original Analysis.py) ─────────────────────
    correct = output_reps["correct"]           # (5, N)
    accs    = correct.float().mean(dim=1)      # (5,)
    exit_labels = ["e1\n(500ms)", "e2\n(900ms)", "e3\n(1100ms)",
                   "e4\n(1300ms)", "ens\n(1500ms)"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = list(range(_N_EXITS))
    ax.plot(x_pos, accs.numpy() * 100, marker="D", linewidth=2, color="#FF0000",
            label=f"SCAN ResNet-{args.depth}  [{args.perturb}]")
    ax.axhline(_CHANCE * 100, ls="--", color="grey", linewidth=1,
               label=f"chance ({_CHANCE*100:.2f}%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exit_labels, fontsize=9)
    ax.set_xlabel("Exit (beep mapping)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"SAT Curve — SCAN ResNet-{args.depth}  [{args.perturb}]")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    curve_path = os.path.join(figs_dir, f"SAT_curve_{args.perturb}.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved: {display_path(curve_path)}")

    # ── SAT benchmark analysis ────────────────────────────────────────────────
    run_scan_sat_analysis(
        test_ds     = test_ds,
        output_reps = output_reps,
        perturb     = args.perturb,
        save_dir    = sat_save_dir,
        ranking_key = "scan",
        refresh     = refresh,
    )

    shutil.rmtree(exp_root, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
