"""
competitors/msdnet/train_sat.py

Adapts the original SATBench MSDNet code to train and evaluate
MSDNet on the same data (identical 70/10/20 split) used for NRM,
instead of the standard ImageNet dataset.

Identical workflow to the original MSDNet code:
  1. train.py  → train the model, save checkpoint
  2. test.py   → load checkpoint, produce predictions for each exit
  3. Analysis  → produce SAT curves compatible with NRM / CNet / ConvRNN / SCAN

The only difference from the original code is the data source:
  - SATDataset (NRM pipeline) instead of ImageNet folders
  - same remove_rt_outliers([-100,+100]), tutorial block removal, split seed=42

Architectural notes (models/msdnet_mod.py):
  - MSDNetMod (small, depth='s'): multi-scale DenseNet backbone, 7 blocks/exits
  - Native: 7 exits (nBlocks=7), we use 5 (equidistant: blocks [0,1,2,3,4])
  - Direct mapping: exit0→500ms, exit1→900ms, ..., exit4→1500ms
  - Cumulative loss: sum of CrossEntropy over all 7 blocks
  - Train from scratch with: SGD(lr=0.1, mom=0.9, wd=1e-4), 300 epochs,
    batch_size=64; LR decayed ×0.1 at epochs [150, 225] (CIFAR schedule)

Usage:
  python competitors/msdnet/train_sat.py --perturb color-gray [--eval_only]
  python competitors/msdnet/train_sat.py --perturb color-blur [--eval_only]
  python competitors/msdnet/train_sat.py --perturb gray-noise [--eval_only]
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
import types
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

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
MSDNET_ROOT = _SCRIPT_DIR
# IMAGE_DIR can also be set via SAT_IMAGE_DIR environment variable
IMAGE_DIR = os.environ.get("SAT_IMAGE_DIR", os.path.join(REPO_ROOT, "datasets", "sat", "data"))

# Put the vendored MSDNet code on the path to import models, utils, etc.
if MSDNET_ROOT not in sys.path:
    sys.path.insert(0, MSDNET_ROOT)

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
# MSDNet: 7 native exits; the first 5 are used (equally spaced along the network)
# as in the original validateonhuman: BLOCKS_TO_USE = [0,1,2,3,4]
_BEEP_MS_TO_EXIT = {500: 0, 900: 1, 1100: 2, 1300: 3, 1500: 4}
_BEEP_TIMES_S    = sorted(ms / 1000.0 for ms in _BEEP_MS_TO_EXIT)
_N_CHOICES       = 16
_CHANCE          = 1.0 / _N_CHOICES
_N_EXITS         = 5     # first 5 of the 7 blocks
_N_BLOCKS        = 7     # total nBlocks of the model (all used in the loss)


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
    in the format expected by MSDNet (224×224, no ImageNet normalisation,
    training from scratch as for SCAN and CNet).
    """
    _TRAIN_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
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
# 2. MSDNet configuration — namespace replicating MSDNETConfig (depth='s')
# =============================================================================

def _make_msdnet_args(num_classes: int = 16) -> types.SimpleNamespace:
    """
    Build the argument namespace MSDNetMod.__init__ needs, replicating
    MSDNETConfig with depth='s' and data='imagenet' (224×224).

    Parameters of the 'small' model (from the paper, same config as train.py):
      base=3, nBlocks=7, step=2, stepmode='even'
      growthRate=6, nChannels=8, nScales=3
      grFactor=[1,1,1], bnFactor=[1,1,1]
      bottleneck=False, prune='max', reduction=0.5
    """
    a = types.SimpleNamespace()
    # model identity
    a.arch        = 'msdnetmod'
    a.data        = 'imagenet'    # → MSDNFirstLayer with 7×7 conv + MaxPool (224→56)
    a.num_classes = num_classes
    # small architecture (depth='s')
    a.base        = 3
    a.step        = 2
    a.stepmode    = 'even'
    a.nBlocks     = _N_BLOCKS     # 7
    a.growthRate  = 6
    a.nChannels   = 8
    a.bottleneck  = False
    a.grFactor    = [1, 1, 1]     # already a list (train.py does split('-'))
    a.bnFactor    = [1, 1, 1]
    a.nScales     = 3             # len(grFactor)
    a.prune       = 'max'
    a.reduction   = 0.5
    return a


# =============================================================================
# 3. Training
# =============================================================================

def _msdnet_loss(
        outputs: list,
        labels: torch.Tensor,
        criterion: nn.Module,
) -> torch.Tensor:
    """
    Cumulative MSDNet loss: sum of CrossEntropy over all blocks.
    Identical to train() in the original utils.py (criterion='ce').
    """
    loss = torch.zeros(1, device=labels.device)
    for logits in outputs:
        loss = loss + criterion(logits, labels)
    return loss


def _adjust_lr(optimizer: optim.Optimizer, epoch: int,
               base_lr: float, milestones: list[int]) -> float:
    """
    Multistep LR decay: divide by 10 at each milestone.
    Milestones default [150, 225] → CIFAR schedule per 300 epochs (paper MSDNet).
    """
    n_decays = sum(1 for m in milestones if epoch >= m)
    lr = base_lr * (0.1 ** n_decays)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_msdnet(
        net: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        n_epochs: int      = 300,
        lr: float          = 0.1,
        momentum: float    = 0.9,
        weight_decay: float = 1e-4,
        lr_milestones: list[int] | None = None,
        save_dir: str | None = None,
        save_freq: int     = 10,
        eval_freq: int     = 5,
        debug: bool        = False,
) -> dict:
    """
    Train MSDNet from scratch.
    Loss: cumulative CE over all nBlocks exits (as in the original train.py).
    Optimiser: SGD(lr, momentum, weight_decay).
    LR schedule: /10 at the given milestones (default: [150, 225]).
    """
    if lr_milestones is None:
        lr_milestones = [150, 225]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)

    metrics: dict = {"train": collections.defaultdict(list),
                     "val":   collections.defaultdict(list)}

    print(f"\n{'='*70}")
    print(f"TRAINING MSDNet  ({n_epochs} ep, lr={lr}, "
          f"milestones={lr_milestones}, bs={train_loader.batch_size})")
    print(f"{'='*70}")

    for epoch in range(n_epochs):
        cur_lr = _adjust_lr(optimizer, epoch, lr, lr_milestones)
        if epoch in lr_milestones:
            print(f"  [LR] Epoch {epoch}: lr → {cur_lr:.2e}")

        net.train()
        sum_loss        = 0.0
        total           = 0
        # track accuracy on the first 5 exits (the 5 mapped to the beeps)
        correct_by_exit = [0] * _N_EXITS

        for batch_i, (images, labels) in enumerate(train_loader):
            if debug and batch_i > 1:
                break

            images = images.to(device)
            labels = labels.to(device)

            # forward: list of nBlocks=7 logit tensors
            outputs = net(images)
            if not isinstance(outputs, list):
                outputs = [outputs]

            loss = _msdnet_loss(outputs, labels, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            total    += labels.size(0)

            # accuracy on the first 5 exits (exit 0 = block 0 = shallowest)
            for e in range(_N_EXITS):
                preds = outputs[e].argmax(dim=1)
                correct_by_exit[e] += preds.eq(labels).sum().item()

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
            val_loss, val_accs = _eval_epoch(net, val_loader, device, criterion, debug)
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

    print("\nTraining MSDNet completed.")
    return metrics


def _eval_epoch(
        net: nn.Module,
        loader: DataLoader,
        device: torch.device,
        criterion: nn.Module,
        debug: bool = False,
) -> tuple[float, list[float]]:
    """Cumulative loss and accuracies of the first 5 exits over a dataloader."""
    net.eval()
    sum_loss        = 0.0
    total           = 0
    correct_by_exit = [0] * _N_EXITS

    with torch.no_grad():
        for batch_i, (images, labels) in enumerate(loader):
            if debug and batch_i > 1:
                break
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            if not isinstance(outputs, list):
                outputs = [outputs]

            loss = _msdnet_loss(outputs, labels, criterion)
            sum_loss += loss.item()
            total    += labels.size(0)

            for e in range(_N_EXITS):
                correct_by_exit[e] += outputs[e].argmax(dim=1).eq(labels).sum().item()

    accs = [100.0 * c / total for c in correct_by_exit]
    return sum_loss / len(loader), accs


# =============================================================================
# 4. Evaluation — produces output_reps in the shared format (T=5)
# =============================================================================

def evaluate_msdnet(
        net: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
) -> dict:
    """
    Run the forward pass over the test set and build output_reps with 5 "exits":
        exit 0 → block 0 (shallowest) → beep 500ms
        exit 1 → block 1              → beep 900ms
        exit 2 → block 2              → beep 1100ms
        exit 3 → block 3              → beep 1300ms
        exit 4 → block 4              → beep 1500ms

    Faithful to validateonhuman() in utils.py: BLOCKS_TO_USE = [0,1,2,3,4].

    Returns a dict with:
        predictions: LongTensor (5, N)
        correct:     BoolTensor (5, N)
        target:      LongTensor (N,)
        softmax:     FloatTensor (5, N, 16)
    """
    all_softmax = [[] for _ in range(_N_EXITS)]
    all_targets = []

    print(f"\n{'='*70}\nEVALUATION MSDNet  (test set)\n{'='*70}")

    net.eval()
    with torch.no_grad():
        for batch_i, (images, labels) in enumerate(test_loader):
            images = images.to(device)

            outputs = net(images)      # list of 7 logit tensors [block0..block6]
            if not isinstance(outputs, list):
                outputs = [outputs]

            # only the first 5 blocks are used (BLOCKS_TO_USE = [0,1,2,3,4])
            for e in range(_N_EXITS):
                all_softmax[e].append(F.softmax(outputs[e], dim=-1).cpu())

            all_targets.append(labels)
            sys.stdout.write(f"\r  Batch {batch_i+1}/{len(test_loader)}")
            sys.stdout.flush()
    print()

    target_t  = torch.cat(all_targets, dim=0)                        # (N,)
    softmax_t = torch.stack(
        [torch.cat(all_softmax[e], dim=0) for e in range(_N_EXITS)],
        dim=0,
    )                                                                  # (5, N, 16)
    preds_t   = softmax_t.argmax(dim=-1)                             # (5, N)
    correct_t = preds_t == target_t.unsqueeze(0)                     # (5, N)

    print("\nAccuracy per exit:")
    exit_labels = ["block0→500ms", "block1→900ms", "block2→1100ms",
                   "block3→1300ms", "block4→1500ms"]
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
# 5. SAT benchmark analysis — identical to CNet/ConvRNN/SCAN, labelled "MSDNet"
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
    """Build model_curves for scripts/eval_sat.py from the MSDNet predictions."""
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
    """Human and MSDNet curves pooled over every level."""
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
                    per_img_acc = np.array(
                        [np.mean(v["correct"]) for v in img_dict.values()])
                    per_img_rt  = np.array(
                        [np.mean(v["rts"]) for v in img_dict.values()])
                else:
                    per_img_acc = np.array(
                        [np.mean(v) for v in img_dict.values()])
                    per_img_rt  = None
                cat_beep[beep_s] = _make_per_beep_entry(
                    beep_s, per_img_acc, per_img_rt)
                all_beep[beep_s].extend(per_img_acc.tolist())
                if per_img_rt is not None:
                    all_rt[beep_s].extend(per_img_rt.tolist())
            if cat_beep:
                by_cat[cat] = _make_curves_dict(cat_beep)

        overall = {}
        for beep_s in _BEEP_TIMES_S:
            accs = all_beep.get(beep_s, [])
            if not accs:
                continue
            rts = np.array(all_rt[beep_s]) if all_rt.get(beep_s) else None
            overall[beep_s] = _make_per_beep_entry(
                beep_s, np.array(accs), rts)
        overall_curves = _make_curves_dict(overall)
        return by_cat, overall_curves

    h_by_cat, h_overall = _to_curves(h_raw, use_rts=True)
    m_by_cat, m_overall = _to_curves(c_raw, use_rts=False)
    return h_by_cat, h_overall, m_by_cat, m_overall


def _plot_sat_curves_beep_vs_acc(
        human_by_cat: dict, model_by_cat: dict,
        human_overall: dict, model_overall: dict,
        save_path, perturb: str,
):
    """
    Pooled SAT curve plot: beep time vs accuracy per category + overall.
    Same style as CNet/ConvRNN/SCAN (diamond marker, red for MSDNet).
    """
    from datasets.sat.loader import CATEGORIES as _CATS

    def _plot_one(ax, hcurve, mcurve, title):
        if hcurve and hcurve.get("beeps_s"):
            bx = hcurve["beeps_s"]
            ay = [hcurve["per_beep"][b]["acc_mean"] * 100 for b in bx]
            ae = [hcurve["per_beep"][b]["acc_sem"]  * 100 for b in bx]
            ax.errorbar(bx, ay, yerr=ae, fmt="o-", color="steelblue",
                        linewidth=1.5, capsize=3, label="Human")
        if mcurve and mcurve.get("beeps_s"):
            bx = mcurve["beeps_s"]
            ay = [mcurve["per_beep"][b]["acc_mean"] * 100 for b in bx]
            ae = [mcurve["per_beep"][b]["acc_sem"]  * 100 for b in bx]
            ax.errorbar(bx, ay, yerr=ae, fmt="D-", color="#CC0000",
                        linewidth=1.5, capsize=3, label="MSDNet")
        ax.axhline(_CHANCE * 100, ls="--", color="grey", lw=0.8,
                   label=f"chance ({_CHANCE*100:.1f}%)")
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_xticks(_BEEP_TIMES_S)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

    n_cats   = _N_CHOICES
    n_cols   = 4
    n_rows   = (n_cats + 2 + n_cols - 1) // n_cols   # +1 overall
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3))
    axes_flat = axes.flatten()

    _plot_one(axes_flat[0], human_overall, model_overall, "Overall")

    for cat_idx in range(n_cats):
        ax = axes_flat[cat_idx + 1]
        nm = _CATS[cat_idx] if cat_idx < len(_CATS) else str(cat_idx)
        _plot_one(ax, human_by_cat.get(cat_idx), model_by_cat.get(cat_idx), nm)

    for i in range(n_cats + 1, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.supxlabel("Beep time (s)", fontsize=9, y=0.01)
    fig.supylabel("Accuracy (%)",  fontsize=9, x=0.01)
    fig.suptitle(f"SAT Curves — MSDNet vs Human  [{perturb}]",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [MSDNet] Saved {display_path(save_path)}")


def run_msdnet_sat_analysis(
        test_ds,
        output_reps: dict,
        perturb: str,
        save_dir: str | os.PathLike,
        ranking_key: str = "msdnet",
        refresh: bool = False,
) -> dict:
    """
    SAT benchmark pipeline for MSDNet — produces the same figures and metrics
    as scripts/eval_sat.py (NRM), run_cnet_sat_analysis (CNet),
    run_convrnn_sat_analysis (ConvRNN) and run_scan_sat_analysis (SCAN).

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
    print("SAT BENCHMARK ANALYSIS  (MSDNet — compatible with NRM)")
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

    print("  Building MSDNet curves per level (from output_reps)...")
    model_curves = _build_model_curves(test_ds, output_reps)
    model_curves["scenario"] = scenario

    print("  Computing per-observer curves (srid)...")
    per_obs_curves = _compute_per_observer_curves(data, ordered_lvls,
                                                   n_choices=_N_CHOICES)
    n_obs = len(per_obs_curves) if per_obs_curves else 0
    print(f"  {n_obs} observers in the test set.")

    print("  Building pooled curves (human + MSDNet)...")
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

    # Full per-observer metrics JSON
    save_satbench_metrics_full(metrics, save_dir)

    # Canonical Table-3 scalars, where pipelines/ranking.ipynb looks for them.
    from scripts.results_io import save_sat_metrics
    canonical = save_sat_metrics(perturb, ranking_key, metrics, refresh=refresh)
    print(f"  [SAT metrics] Canonical scalars saved to {display_path(canonical)}")

    # Figure SATbench (fig4–fig7)
    if WRITE_DIAGNOSTICS:
        plot_satbench_figures(
            metrics, human_curves, model_curves,
            save_dir=save_dir,
            prefix="satbench",
            n_choices=_N_CHOICES,
        )

    # LaTeX table
    if WRITE_DIAGNOSTICS:
        latex_str  = generate_latex_table(metrics, scenario_name=f"{scenario} MSDNet")
        tex_path   = save_dir / "sat_level_metrics_table.tex"
        tex_path.write_text(latex_str)
        print(f"  [MSDNet] Saved {display_path(tex_path)}")

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
        print(f"  [MSDNet] Saved {display_path(csv_path)}")

        # Print accuracy per beep/level
        print("\n  MSDNet accuracy per level/deadline:")
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

        print("\nSAT benchmark MSDNet completed.")
    return {"metrics": metrics, "save_dir": str(save_dir)}


# =============================================================================
# 6. Main script
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="MSDNet on the SAT data (NRM pipeline) - train + eval + analysis"
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
    # hyperparameters (paper: 300 epochs, SGD lr=0.1, batch=64, milestones=[150,225])
    p.add_argument("--n_epochs",     type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_milestones", type=int,  nargs="+", default=[150, 225],
                   help="Epochs at which lr is divided by 10 (default: 150 225).")
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

    import models as _msdnet_models   # imported from msdnet/ (sys.path above)

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device : {device}")
    print(f"Perturb: {args.perturb}")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    np.random.seed(0)

    exp_name = f"msdnet_small_SAT_{args.perturb}"

    # ── output and scratch directories ────────────────────────────────────────
    # A training run, --refresh or --refresh_weights writes every output with
    # the _REFRESH suffix, so the reference outputs are never overwritten.
    sys.path.insert(0, REPO_ROOT)
    from scripts.results_io import sat_model_dir
    refresh = args.refresh or args.refresh_weights or not args.eval_only
    sat_save_dir = str(sat_model_dir(args.perturb, "msdnet", refresh=refresh))
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
    print(f"\nBuilding MSDNet small (num_classes={_N_CHOICES})...")
    msdnet_cfg = _make_msdnet_args(num_classes=_N_CHOICES)
    net = _msdnet_models.msdnetmod(msdnet_cfg).to(device)
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

        train_metrics = train_msdnet(
            net           = net,
            train_loader  = train_loader,
            val_loader    = val_loader,
            device        = device,
            n_epochs      = args.n_epochs,
            lr            = args.lr,
            momentum      = args.momentum,
            weight_decay  = args.weight_decay,
            lr_milestones = args.lr_milestones,
            save_dir      = ckpt_dir,
            save_freq     = args.save_freq,
            eval_freq     = args.eval_freq,
            debug         = args.debug,
        )

        # save config and metrics
        with open(os.path.join(exp_root, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        with open(os.path.join(exp_root, "train_metrics.json"), "w") as f:
            def _to_serializable(v):
                if isinstance(v, (list, tuple)):
                    return [_to_serializable(x) for x in v]
                if isinstance(v, float):
                    return v
                return v
            json.dump(
                {k: {kk: _to_serializable(vv) for kk, vv in v.items()}
                 for k, v in train_metrics.items()},
                f, indent=2, default=str,
            )

        keep_final_checkpoint(ckpt_dir, "msdnet", args.perturb,
                              pattern="ckpt__epoch_*.pt")

    # ── load checkpoint ─────────────────────────────────────────────────────
    # The checkpoint a training run just saved (or, with --refresh_weights, a
    # previous one), else the released weights in competitors/weights/sat/.
    # Both layouts are `{"epoch", "model_state", "optim_state"}`.
    selected_ckpt = resolve_sat_checkpoint(
        model="msdnet", perturb=args.perturb, explicit=args.weights,
        refresh_weights=args.refresh_weights or not args.eval_only,
    )
    ckpt_data = torch.load(selected_ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ckpt_data["model_state"])
    print(f"  Checkpoint epoch {ckpt_data.get('epoch', '?')} loaded.")

    # ── evaluation ────────────────────────────────────────────────────────────
    test_loader = _build_loader(test_ds, train=False,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    output_reps = evaluate_msdnet(net, test_loader, device)

    out_rep_path = os.path.join(out_dir, "output_representations__test__OSD.pt")
    torch.save(output_reps, out_rep_path)
    print(f"Saved: {display_path(out_rep_path)}")

    # ── SAT curve per exit (analogous to the original validate for each block) ────
    correct = output_reps["correct"]    # (5, N)
    accs    = correct.float().mean(dim=1)   # (5,)
    exit_labels = ["b0\n(500ms)", "b1\n(900ms)", "b2\n(1100ms)",
                   "b3\n(1300ms)", "b4\n(1500ms)"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = list(range(_N_EXITS))
    ax.plot(x_pos, accs.numpy() * 100, marker="D", linewidth=2, color="#CC0000",
            label=f"MSDNet small  [{args.perturb}]")
    ax.axhline(_CHANCE * 100, ls="--", color="grey", linewidth=1,
               label=f"chance ({_CHANCE*100:.2f}%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exit_labels, fontsize=9)
    ax.set_xlabel("Exit / Block (beep mapping)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"SAT Curve — MSDNet small  [{args.perturb}]")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    curve_path = os.path.join(figs_dir, f"SAT_curve_{args.perturb}.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved: {display_path(curve_path)}")

    # ── SAT benchmark analysis ────────────────────────────────────────────────
    run_msdnet_sat_analysis(
        test_ds     = test_ds,
        output_reps = output_reps,
        perturb     = args.perturb,
        save_dir    = sat_save_dir,
        ranking_key = "msdnet",
        refresh     = refresh,
    )

    shutil.rmtree(exp_root, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
