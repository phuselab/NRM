"""
competitors/convrnn/train_sat.py

Adapts the original SATBench ConvRNN code to train and evaluate
ConvRNN (BL-Net) on the same data (identical 70/10/20 split) used for NRM,
instead of the standard ImageNet dataset.

Identical workflow to the original ConvRNN code:
  1. train.py  → train the model (readout only, rest frozen), save checkpoint
  2. test.py   → load checkpoint, produce predictions for each timestep
  3. Analysis  → produce SAT curves compatible with NRM / CNet

The only difference from the original code is:
  - PyTorch reimplementation of BL-Net (original uses TensorFlow/Keras)
  - Data source: SATDataset (NRM pipeline) instead of ImageNet folders
  - same remove_rt_outliers([-100,+100]), tutorial block removal, split seed=42

Architectural notes:
  - BL-Net has 8 timesteps (0-7) with recurrent lateral connections
  - Only the ReadoutDense is trainable; everything else (conv + BN) is frozen
  - ImageNet pre-trained weights: bl_imagenet.h5 (downloaded from OSF if missing: https://osf.io/9td5p/download)
  - Exit mapping (5 beep ← 8 timestep): exits=[0,2,4,6,7], analogous to CNet's [0,2,4,6,8]

Usage:
  python competitors/convrnn/train_sat.py --perturb color-gray [--eval_only] [--bl_weights competitors/weights/sat/convrnn_backbone.h5]
  python competitors/convrnn/train_sat.py --perturb color-blur [--eval_only]
  python competitors/convrnn/train_sat.py --perturb gray-noise [--eval_only]
"""

import argparse
import shutil
import tempfile
import collections
import csv
import glob as _glob
import importlib.util
import os
import sys
import json
import urllib.request
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from scripts.results_io import display_path


# ── paths ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
CONVRNN_ROOT  = _SCRIPT_DIR
IMAGE_DIR = os.environ.get("SAT_IMAGE_DIR", os.path.join(REPO_ROOT, "datasets", "sat", "data"))

PERTURB_TO_FILE = {
    "color-gray": os.path.join(REPO_ROOT, "datasets/sat", "color-gray.txt"),
    "color-blur": os.path.join(REPO_ROOT, "datasets/sat", "color-blur.txt"),
    "gray-noise": os.path.join(REPO_ROOT, "datasets/sat", "gray-noise.txt"),
}

#: ImageNet-pre-trained BL-Net backbone, as the release names it.
BL_IMAGENET_WEIGHTS = os.path.join(REPO_ROOT, "competitors", "weights", "sat",
                                   "convrnn_backbone.h5")

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
# Mapping: beep_ms → timestep index (from split_exit_accuracies.py, adapted to 8 timesteps)
# CNet uses exits=[0,2,4,6,8] of 9 exits; here [0,2,4,6,7] of 8 timesteps
_BEEP_MS_TO_EXIT = {500: 0, 900: 2, 1100: 4, 1300: 6, 1500: 7}
_BEEP_TIMES_S    = sorted(ms / 1000.0 for ms in _BEEP_MS_TO_EXIT)
_N_CHOICES       = 16
_CHANCE          = 1.0 / _N_CHOICES
_N_TIMESTEPS     = 8

# ── normalisation (BL-Net was pre-trained on image/127.5 - 1, i.e. [-1,1]) ──
_BLNET_MEAN = [0.5, 0.5, 0.5]
_BLNET_STD  = [0.5, 0.5, 0.5]


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
    in the format expected by BL-Net (224×224, [-1,1] normalisation).
    """
    _TRAIN_TFM = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_BLNET_MEAN, _BLNET_STD),
    ])
    _EVAL_TFM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_BLNET_MEAN, _BLNET_STD),
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

    # same outlier removal
    dataset.remove_rt_outliers(outlier_range_ms=[-100, 100])

    # remove tutorial blocks (block tutorial, time_limit=-1)
    exp_mask = dataset.df["block"] != "cattut"
    n_rem = int((~exp_mask).sum())
    if n_rem:
        dataset.df = dataset.df[exp_mask].reset_index(drop=True)
        print(f"  Removed {n_rem} tutorial trials ({len(dataset.df)} experimental).")

    # same 70/10/20 split, seed=42
    train_ds, val_ds, test_ds = dataset.train_val_test_split(
        val_size=0.1, test_size=0.2, seed=42, shuffle=True,
    )
    print(f"  Split — Train: {len(train_ds.df)}, Val: {len(val_ds.df)}, "
          f"Test: {len(test_ds.df)}")
    return train_ds, val_ds, test_ds


# =============================================================================
# 2. BL-Net — a PyTorch reimplementation faithful to rcnn_sat/bl_net.py
# =============================================================================

class BLNet(nn.Module):
    """
    Bottom-up + Lateral recurrent CNN (BL-Net), PyTorch.
    Faithful to the Keras implementation in bl_net.py (Spoerer et al. 2020).

    Architecture:
      7 BLConvLayers with lateral connections between successive timesteps.
      Each layer: b_conv (weights shared over t) + l_conv (weights shared over t)
      + BatchNorm (NOT shared, distinct for each (t, n)) + ReLU.
      ReadoutDense shared → softmax for each timestep.

    Layer specs: (filters, kernel_size)
      0: (96,  7)  → output 224×224×96
      1: (128, 5)  → output 112×112×128  (after MaxPool)
      2: (192, 3)  → output  56×56×192
      3: (256, 3)  → output  28×28×256
      4: (512, 3)  → output  14×14×512
      5: (1024,3)  → output   7×7×1024
      6: (2048,1)  → output   3×3×2048   (3 = floor(7/2))
      GlobalAvgPool → 2048 → Dense(n_classes) → Softmax
    """

    # (filters, kernel_size, in_channels_for_b_conv)
    _LAYER_SPECS = [
        (96,   7, 3),      # layer 0: image in (3ch)
        (128,  5, 96),     # layer 1: MaxPool(act[0])
        (192,  3, 128),    # layer 2: MaxPool(act[1])
        (256,  3, 192),    # layer 3: MaxPool(act[2])
        (512,  3, 256),    # layer 4: MaxPool(act[3])
        (1024, 3, 512),    # layer 5: MaxPool(act[4])
        (2048, 1, 1024),   # layer 6: MaxPool(act[5])
    ]

    def __init__(self, n_classes: int = 16, n_timesteps: int = 8):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_layers    = len(self._LAYER_SPECS)

        # b_conv and l_conv weights shared across timesteps (one instance for every t)
        self.b_convs = nn.ModuleList()
        self.l_convs = nn.ModuleList()
        for filters, ksize, in_ch_b in self._LAYER_SPECS:
            pad = ksize // 2
            self.b_convs.append(
                nn.Conv2d(in_ch_b, filters, ksize, padding=pad, bias=False)
            )
            # l_conv: lateral from activations[t-1][n] → same spatial, same filters
            self.l_convs.append(
                nn.Conv2d(filters, filters, ksize, padding=pad, bias=False)
            )

        # BatchNorm NOT shared: bns[t][n]
        self.bns = nn.ModuleList([
            nn.ModuleList([
                nn.BatchNorm2d(filters)
                for filters, _, __ in self._LAYER_SPECS
            ])
            for _ in range(n_timesteps)
        ])

        # Readout shared across timesteps
        last_filters = self._LAYER_SPECS[-1][0]
        self.readout = nn.Linear(last_filters, n_classes)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            outputs: list of n_timesteps tensors (B, n_classes), each a softmax
        """
        # activations[t][n]: feature map at timestep t, layer n
        activations: list[list[torch.Tensor | None]] = [
            [None] * self.n_layers for _ in range(self.n_timesteps)
        ]
        # cache of layer 0's b_conv at t=0 (the input image does not change across timesteps)
        b_conv_cache_0: torch.Tensor | None = None
        outputs: list[torch.Tensor] = []

        for t in range(self.n_timesteps):
            for n in range(self.n_layers):
                # ── bottom-up ──────────────────────────────────────────────
                if n == 0:
                    if t == 0:
                        b_out = self.b_convs[0](x)
                        b_conv_cache_0 = b_out          # cache per t>0
                    else:
                        b_out = b_conv_cache_0          # reuse the cache
                else:
                    b_in  = F.max_pool2d(activations[t][n - 1], kernel_size=2)
                    b_out = self.b_convs[n](b_in)

                # ── lateral (only if t > 0) ─────────────────────────────
                if t > 0:
                    l_out    = self.l_convs[n](activations[t - 1][n])
                    conv_sum = b_out + l_out
                else:
                    conv_sum = b_out

                # ── BatchNorm + ReLU ────────────────────────────────────
                activations[t][n] = F.relu(self.bns[t][n](conv_sum))

            # ── readout ────────────────────────────────────────────────────
            gap    = activations[t][-1].mean(dim=[2, 3])   # GlobalAvgPool
            logits = self.readout(gap)
            outputs.append(F.softmax(logits, dim=-1))

        return outputs


# =============================================================================
# 3. Loading the pre-trained weights from bl_imagenet.h5 (Keras → PyTorch)
# =============================================================================

def _ensure_h5py() -> None:
    """Install h5py if it is not available."""
    try:
        import h5py  # noqa: F401
    except ImportError:
        import subprocess
        print("  [INFO] h5py not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py"],
                              stdout=subprocess.DEVNULL)
        print("  [INFO] h5py installed.")


def _download_bl_imagenet(dest: str) -> bool:
    """Download bl_imagenet.h5 from OSF. Returns True on success."""
    url = "https://osf.io/9td5p/download"
    print(f"  Downloading bl_imagenet.h5 from {url} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved: {dest}")
        return True
    except Exception as e:
        print(f"  [WARN] Download failed: {e}")
        return False


def _read_h5_weights(f, *path_candidates) -> np.ndarray | None:
    """
    Look for an HDF5 dataset among the candidate paths (to tolerate layout differences).
    Returns the numpy array, or None if not found.
    """
    import h5py
    for path in path_candidates:
        try:
            obj = f[path]
            if isinstance(obj, h5py.Dataset):
                return np.array(obj)
        except (KeyError, TypeError):
            pass
    return None


def load_blnet_pretrained_weights(
        model: BLNet,
        h5_path: str,
        verbose: bool = True,
) -> None:
    """
    Load the ImageNet pre-trained weights from bl_imagenet.h5 (Keras format)
    into the PyTorch BLNet model.

    Keras → PyTorch weight mapping:
      Conv2D kernel: (kH, kW, in_C, out_C) → (out_C, in_C, kH, kW)
      Dense kernel:  (in_F, out_F)          → (out_F, in_F)
      BatchNorm gamma → weight, beta → bias,
                moving_mean → running_mean, moving_variance → running_var

    ReadoutDense is ignored if shapes do not match (different class count),
    exactly as `model.load_weights(..., skip_mismatch=True)` does in Keras.
    """
    import h5py

    _ensure_h5py()

    if not os.path.isfile(h5_path):
        print(f"  [WARN] {display_path(h5_path)} not found, using random init.")
        return

    loaded, skipped = 0, 0

    def _load_conv(layer: nn.Conv2d, keras_w: np.ndarray) -> bool:
        # Keras: (kH, kW, in_C, out_C) → PyTorch: (out_C, in_C, kH, kW)
        w_pt = torch.tensor(np.transpose(keras_w, (3, 2, 0, 1)), dtype=torch.float32)
        if w_pt.shape == layer.weight.shape:
            layer.weight.data.copy_(w_pt)
            return True
        return False

    def _load_bn(bn: nn.BatchNorm2d,
                 gamma: np.ndarray, beta: np.ndarray,
                 mean: np.ndarray, var: np.ndarray) -> bool:
        if (gamma.shape == bn.weight.shape and
                beta.shape  == bn.bias.shape   and
                mean.shape  == bn.running_mean.shape and
                var.shape   == bn.running_var.shape):
            bn.weight.data.copy_(torch.tensor(gamma, dtype=torch.float32))
            bn.bias.data.copy_(torch.tensor(beta,  dtype=torch.float32))
            bn.running_mean.copy_(torch.tensor(mean, dtype=torch.float32))
            bn.running_var.copy_(torch.tensor(var,  dtype=torch.float32))
            return True
        return False

    with h5py.File(h5_path, "r") as f:
        # Inspect the root to find the layout (TF1/TF2 differ)
        roots = list(f.keys())
        # In Keras2/TF2 with save_weights, the root can be 'model_weights'
        # or the model name ('bl_net')
        mw_key = None
        for candidate in ("model_weights", "bl_net", ""):
            if candidate == "" or candidate in roots:
                mw_key = candidate
                break

        def _get(layer_name: str, weight_name: str) -> np.ndarray | None:
            """Look the weight up under the various HDF5 path conventions."""
            candidates = []
            if mw_key:
                candidates += [
                    f"{mw_key}/{layer_name}/{layer_name}/{weight_name}",
                    f"{mw_key}/{layer_name}/{weight_name}",
                ]
            candidates += [
                f"{layer_name}/{layer_name}/{weight_name}",
                f"{layer_name}/{weight_name}",
            ]
            return _read_h5_weights(f, *candidates)

        # ── b_conv and l_conv ────────────────────────────────────────────────
        for n in range(model.n_layers):
            b_name = f"RCL_{n}_BConv"
            l_name = f"RCL_{n}_LConv"

            bw = _get(b_name, "kernel:0")
            if bw is not None and _load_conv(model.b_convs[n], bw):
                loaded += 1
            else:
                skipped += 1
                if verbose:
                    print(f"  [SKIP] {b_name} - not found or shape mismatch")

            lw = _get(l_name, "kernel:0")
            if lw is not None and _load_conv(model.l_convs[n], lw):
                loaded += 1
            else:
                skipped += 1
                if verbose:
                    print(f"  [SKIP] {l_name} - not found or shape mismatch")

        # ── BatchNorm ─────────────────────────────────────────────────────────
        for t in range(model.n_timesteps):
            for n in range(model.n_layers):
                bn_name = f"BatchNorm_Layer_{n}_Time_{t}"
                gamma = _get(bn_name, "gamma:0")
                beta  = _get(bn_name, "beta:0")
                mean  = _get(bn_name, "moving_mean:0")
                var   = _get(bn_name, "moving_variance:0")

                if all(x is not None for x in (gamma, beta, mean, var)):
                    if _load_bn(model.bns[t][n], gamma, beta, mean, var):
                        loaded += 1
                    else:
                        skipped += 1
                        if verbose:
                            print(f"  [SKIP] {bn_name} — shape mismatch")
                else:
                    skipped += 1
                    if verbose:
                        print(f"  [SKIP] {bn_name} - not found")

        # ── ReadoutDense (skipped if the shape != our classes) ────────────────────
        rk = _get("ReadoutDense", "kernel:0")
        rb = _get("ReadoutDense", "bias:0")
        if rk is not None and rb is not None:
            # Keras Dense kernel: (in_F, out_F) → PyTorch: (out_F, in_F)
            w_pt = torch.tensor(rk.T, dtype=torch.float32)
            b_pt = torch.tensor(rb,   dtype=torch.float32)
            if (w_pt.shape == model.readout.weight.shape and
                    b_pt.shape == model.readout.bias.shape):
                model.readout.weight.data.copy_(w_pt)
                model.readout.bias.data.copy_(b_pt)
                loaded += 1
                if verbose:
                    print("  [OK] ReadoutDense loaded.")
            else:
                if verbose:
                    print(f"  [SKIP] ReadoutDense — shape {w_pt.shape} ≠ "
                          f"{model.readout.weight.shape} (different #classes, ok).")

    if verbose:
        print(f"  Weights loaded: {loaded},  skipped: {skipped}")


# =============================================================================
# 4. Training — everything frozen except the readout, Adam, mean CE over 8 timesteps
# =============================================================================

def train_convrnn(
        model: BLNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        n_epochs: int = 100,
        lr: float = 0.005,
        eps: float = 0.1,
        save_dir: str | None = None,
        save_freq: int = 10,
        eval_freq: int = 10,
        debug: bool = False,
) -> dict:
    """
    Train only the ReadoutDense, everything else frozen.
    Loss = mean cross-entropy over the n_timesteps outputs.
    """
    # Freeze everything except the readout
    for param in model.parameters():
        param.requires_grad = False
    for param in model.readout.parameters():
        param.requires_grad = True

    # BN in eval mode → uses the pre-trained running_mean/var
    model.eval()

    optimizer = optim.Adam(model.readout.parameters(), lr=lr, eps=eps)
    criterion = nn.CrossEntropyLoss()

    metrics: dict = {"train": collections.defaultdict(list),
                     "val":   collections.defaultdict(list)}

    print(f"\n{'='*70}\nTRAINING ConvRNN  ({n_epochs} epochs, lr={lr}, eps={eps})\n{'='*70}")

    for epoch in range(n_epochs):
        # ── training ───────────────────────────────────────────────────────
        # model.train() is not called, so BN stays in eval mode
        # Only the readout has requires_grad=True, so gradients flow there only
        epoch_loss, epoch_acc = [], []

        for batch_i, (images, labels) in enumerate(train_loader):
            if debug and batch_i > 1:
                break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # outputs: list of _N_TIMESTEPS tensors (B, n_classes) - softmax
            outputs = model(images)

            # loss averaged over the timesteps
            # CrossEntropyLoss expects logits; the outputs are softmax, but
            # log(softmax) + NLLLoss is equivalent, so NLL on log-softmax is used
            loss = sum(
                F.cross_entropy(torch.log(out + 1e-9), labels)
                for out in outputs
            ) / len(outputs)

            loss.backward()
            optimizer.step()

            # accuracy: the last timestep is the reference
            preds = outputs[-1].argmax(dim=1)
            acc   = (preds == labels).float().mean().item()

            epoch_loss.append(loss.item())
            epoch_acc.append(acc)

            sys.stdout.write(
                f"\r  Epoch {epoch+1}/{n_epochs}  batch {batch_i+1}/{len(train_loader)}"
                f"  loss={np.mean(epoch_loss):.4f}  acc={np.mean(epoch_acc)*100:.1f}%"
            )
            sys.stdout.flush()

        print()  # newline after the progress line
        metrics["train"]["loss"].append(float(np.mean(epoch_loss)))
        metrics["train"]["acc"].append(float(np.mean(epoch_acc)))

        # ── validation ────────────────────────────────────────────────────
        if epoch % eval_freq == 0 or epoch == n_epochs - 1:
            val_loss, val_acc = _eval_epoch(model, val_loader, device, debug)
            metrics["val"]["loss"].append((epoch, val_loss))
            metrics["val"]["acc"].append((epoch, val_acc))
            print(f"  Val  loss={val_loss:.4f}  acc={val_acc*100:.2f}%")

        # ── checkpoint ─────────────────────────────────────────────────────
        if save_dir and (epoch % save_freq == 0 or epoch == n_epochs - 1):
            if not debug:
                ckpt_path = os.path.join(save_dir, f"ckpt__epoch_{epoch:04d}.pt")
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                }, ckpt_path)
                print(f"  Checkpoint saved: {display_path(ckpt_path)}")

    print("\nTraining ConvRNN completed.")
    return metrics


def _eval_epoch(
        model: BLNet,
        loader: DataLoader,
        device: torch.device,
        debug: bool = False,
) -> tuple[float, float]:
    """Loss/accuracy over a dataloader, without updating any weights."""
    losses, accs = [], []
    with torch.no_grad():
        for batch_i, (images, labels) in enumerate(loader):
            if debug and batch_i > 1:
                break
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = sum(
                F.cross_entropy(torch.log(out + 1e-9), labels)
                for out in outputs
            ) / len(outputs)
            preds   = outputs[-1].argmax(dim=1)
            acc     = (preds == labels).float().mean().item()
            losses.append(loss.item())
            accs.append(acc)
    return float(np.mean(losses)), float(np.mean(accs))


# =============================================================================
# 5. Evaluation — produces output_reps in CNet's format
# =============================================================================

def evaluate_convrnn(
        model: BLNet,
        test_loader: DataLoader,
        device: torch.device,
) -> dict:
    """
    Run the model's forward pass over the test set and build output_reps
    in the same format CNet's eval.py produces:
        {
          'predictions': LongTensor (T, N)
          'correct':     BoolTensor (T, N)
          'target':      LongTensor (N,)
          'softmax':     FloatTensor (T, N, n_classes)
        }
    with T = _N_TIMESTEPS = 8.
    """
    all_softmax  = [[] for _ in range(_N_TIMESTEPS)]
    all_targets  = []

    print(f"\n{'='*70}\nEVALUATION ConvRNN  (test set)\n{'='*70}")

    model.eval()
    with torch.no_grad():
        for batch_i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)    # list of T tensors (B, n_classes)
            for t in range(_N_TIMESTEPS):
                all_softmax[t].append(outputs[t].cpu())
            all_targets.append(labels)

            sys.stdout.write(f"\r  Batch {batch_i+1}/{len(test_loader)}")
            sys.stdout.flush()
    print()

    # Assemble the final tensors
    target_t   = torch.cat(all_targets, dim=0)                    # (N,)
    softmax_t  = torch.stack(
        [torch.cat(all_softmax[t], dim=0) for t in range(_N_TIMESTEPS)],
        dim=0
    )                                                               # (T, N, C)
    preds_t    = softmax_t.argmax(dim=-1)                         # (T, N)
    correct_t  = (preds_t == target_t.unsqueeze(0))               # (T, N)

    print("\nAccuracy per timestep:")
    for t in range(_N_TIMESTEPS):
        acc = correct_t[t].float().mean().item() * 100
        print(f"  t={t}: {acc:.2f}%")

    return {
        "predictions": preds_t,
        "correct":     correct_t,
        "target":      target_t,
        "softmax":     softmax_t,
    }


# =============================================================================
# 6. SAT benchmark analysis — identical to CNet, labelled "ConvRNN"
# =============================================================================

def _build_human_data_dict(test_ds) -> dict:
    """Extract the 'data' dict for scripts/eval_sat.py from the test-set DataFrame."""
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
    """Helper: build the per_beep entry in the curves_dict format."""
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
    """
    Build model_curves in the format scripts/eval_sat.py expects,
    using the ConvRNN predictions already computed in output_reps.
    Identical to CNet's _build_cnet_model_curves but with ConvRNN's _BEEP_MS_TO_EXIT.
    """
    from scripts.eval_sat import infer_scenario, _make_curves_dict
    from collections import defaultdict as _dd

    df             = test_ds.df.reset_index(drop=True)
    N              = len(df)
    correct        = output_reps["correct"]           # (T, N)
    time_limits_ms = df["time_limit"].values.astype(int)
    true_labels    = df["choice"].values.astype(int)
    levels_list    = df["level_str"].tolist()
    image_indices  = df["image_index"].values.astype(int)
    sat_mask       = time_limits_ms > 0

    scenario, ordered_levels, display, colors = infer_scenario(levels_list)

    assert correct.shape[1] == N

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
    """
    Human and ConvRNN curves pooled over every level (no grouping by level).
    Used for sat_curves_beep_vs_acc.png.
    """
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
    """17-subplot figure (overall + 16 categories) — ConvRNN vs Human."""
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
            (h_data, "#4472C4", "Human",    "-",  "o"),
            (m_data, "#70AD47", "ConvRNN",  "--", "^"),
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
    fig.suptitle(f"SAT Curves — ConvRNN vs Human  [{perturb}]",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ConvRNN] Saved {display_path(save_path)}")


def run_convrnn_sat_analysis(
        test_ds,
        output_reps: dict,
        perturb: str,
        save_dir: str | os.PathLike,
        ranking_key: str = "convrnn",
        refresh: bool = False,
) -> dict:
    """
    SAT benchmark pipeline for ConvRNN — produces the same figures and metrics
    as scripts/eval_sat.py for NRM and run_cnet_sat_analysis for CNet.

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
    print("SAT BENCHMARK ANALYSIS  (ConvRNN — compatible with NRM)")
    print("=" * 70)

    # ── 1. Human data dict ───────────────────────────────────────────────
    print("  Extracting the human data from the test set...")
    data = _build_human_data_dict(test_ds)

    # ── 2. Human curves grouped by level ─────────────────────────────────
    print("  Computing human curves per level...")
    human_curves = calculate_human_curves_by_level(data, n_choices=_N_CHOICES)
    scenario     = human_curves["scenario"]
    ordered_lvls = human_curves["ordered_levels"]
    print(f"  Scenario: {scenario}  |  Levels: {ordered_lvls}")

    if not ordered_lvls:
        print("  [WARN] No level found - SAT analysis skipped.")
        return {}

    # ── 3. ConvRNN curves per level ──────────────────────────────────────
    print("  Building ConvRNN curves per level (from output_reps)...")
    model_curves = _build_model_curves(test_ds, output_reps)
    model_curves["scenario"] = scenario

    # ── 4. Per-observer curves (for the eRMSE / Spearman boxplots) ─────────
    print("  Computing per-observer curves (srid)...")
    per_obs_curves = _compute_per_observer_curves(data, ordered_lvls,
                                                   n_choices=_N_CHOICES)
    n_obs = len(per_obs_curves) if per_obs_curves else 0
    print(f"  {n_obs} observers in the test set.")

    # ── 5. Pooled curves (all levels) ───────────────────────────────────
    print("  Building pooled curves (human + ConvRNN)...")
    h_by_cat, h_overall, m_by_cat, m_overall = _build_pooled_curves(
        test_ds, output_reps
    )

    # ── 6. JSON curve ─────────────────────────────────────────────────────
    json_path = save_dir / "sat_level_curves.json"
    save_curves_to_json(human_curves, model_curves, json_path)

    # ── 7. sat_curves_beep_vs_acc.png ─────────────────────────────────────
    if WRITE_DIAGNOSTICS:
        _plot_sat_curves_beep_vs_acc(
            h_by_cat, m_by_cat, h_overall, m_overall,
            save_path=save_dir / "sat_curves_beep_vs_acc.png",
            perturb=perturb,
        )

    # ── 8. sat_level_beep_vs_acc.png ──────────────────────────────────────
    if WRITE_DIAGNOSTICS:
        plot_sat_by_level(
            human_curves, model_curves,
            x_mode="beep", use_sem=True,
            save_path=save_dir / "sat_level_beep_vs_acc.png",
        )

    # ── 9. sat_level_by_cat_beep_vs_acc.png ──────────────────────────────
    if WRITE_DIAGNOSTICS:
        plot_sat_by_level_and_category(
            human_curves, model_curves,
            x_mode="beep", use_sem=True, n_choices=_N_CHOICES,
            save_path=save_dir / "sat_level_by_cat_beep_vs_acc.png",
        )

    # ── 10. SATBench metrics ──────────────────────────────────────────────
    print("  Computing metrics (eRMSE, Spearman, steepness)...")
    metrics = compute_satbench_metrics(
        human_curves, model_curves, per_obs_curves, n_choices=_N_CHOICES
    )
    metrics["colors"] = human_curves["colors"]

    # ── 11. Full per-observer metrics JSON ───────────────────────────────
    save_satbench_metrics_full(metrics, save_dir)

    # Canonical Table-3 scalars, where pipelines/ranking.ipynb looks for them.
    from scripts.results_io import save_sat_metrics
    canonical = save_sat_metrics(perturb, ranking_key, metrics, refresh=refresh)
    print(f"  [SAT metrics] Canonical scalars saved to {display_path(canonical)}")

    # ── 12. Figure 4-7 ────────────────────────────────────────────────────
    if WRITE_DIAGNOSTICS:
        plot_satbench_figures(
            metrics, human_curves, model_curves,
            save_dir=save_dir,
            prefix="satbench",
            n_choices=_N_CHOICES,
        )

    # ── 13. LaTeX table ───────────────────────────────────────────────────
    if WRITE_DIAGNOSTICS:
        latex_str  = generate_latex_table(metrics, scenario_name=f"{scenario} ConvRNN")
        latex_path = save_dir / "sat_level_metrics_table.tex"
        latex_path.write_text(latex_str)
        print(f"  [ConvRNN] Saved {display_path(latex_path)}")

        # ── 13. CSV metrics table ─────────────────────────────────────────────
        csv_path = save_dir / "sat_level_metrics_table.csv"
        disp     = metrics.get("display_labels", {})
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["level", "display", "rmse", "spearman_r",
                             "steepness_human", "steepness_model"])
            for lv in ordered_lvls:
                row = metrics["overall"].get(lv, {})
                writer.writerow([
                    lv,
                    disp.get(lv, lv),
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
        print(f"  [ConvRNN] Saved {display_path(csv_path)}")

        # ── 14. Print accuracy per beep/level ────────────────────────────────
        print("\n  ConvRNN accuracy per level/deadline:")
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

        print("\nSAT benchmark ConvRNN completed.")
    return {"metrics": metrics, "save_dir": str(save_dir)}


# =============================================================================
# 7. Main script
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="ConvRNN (BL-Net) on the SAT data (NRM pipeline) - train + eval + analysis"
    )
    p.add_argument("--perturb", required=True,
                   choices=["color-gray", "color-blur", "gray-noise"])
    p.add_argument("--diagnostics", action="store_true",
                   help="Also write the per-model PNGs and the per-model "
                        ".tex/.csv. Off by default: pipelines/sat.ipynb "
                        "only reads sat_level_curves.json and "
                        "satbench_metrics_full.json.")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training and load the last saved checkpoint.")
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
    # hyperparameters (from the paper: lr=0.005, eps=0.1, batch=128, epochs=100)
    p.add_argument("--n_epochs",   type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=0.005)
    p.add_argument("--adam_eps",   type=float, default=0.1)
    p.add_argument("--eval_freq",  type=int,   default=10)
    p.add_argument("--save_freq",  type=int,   default=10)
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--device",     type=int,   default=0)
    p.add_argument("--debug",      action="store_true", default=False,
                   help="Run only 2 batches per epoch (smoke test).")
    # path of the pre-trained weights
    p.add_argument("--bl_weights", default=BL_IMAGENET_WEIGHTS,
                   help="Path of the ImageNet pre-trained BL-Net backbone. "
                        "Default: competitors/weights/sat/convrnn_backbone.h5 "
                        "(downloaded if missing).")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Do not load bl_imagenet.h5; use a random init.")
    return p.parse_args()


def main():
    args = parse_args()
    global WRITE_DIAGNOSTICS
    WRITE_DIAGNOSTICS = bool(getattr(args, 'diagnostics', False))

    # ── device ───────────────────────────────────────────────────────────────
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device : {device}")
    print(f"Perturb: {args.perturb}")

    torch.manual_seed(42)
    np.random.seed(42)

    exp_name = f"convrnn_blnet_SAT_{args.perturb}"

    # ── output and scratch directories ────────────────────────────────────────
    # A training run, --refresh or --refresh_weights writes every output with
    # the _REFRESH suffix, so the reference outputs are never overwritten.
    sys.path.insert(0, REPO_ROOT)
    from scripts.results_io import sat_model_dir
    refresh = args.refresh or args.refresh_weights or not args.eval_only
    sat_save_dir = str(sat_model_dir(args.perturb, "convrnn", refresh=refresh))
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
    print("\nBuilding BL-Net...")
    model = BLNet(n_classes=_N_CHOICES, n_timesteps=_N_TIMESTEPS).to(device)
    n_params_total   = sum(p.numel() for p in model.parameters())
    n_params_readout = sum(p.numel() for p in model.readout.parameters())
    print(f"  Total parameters:   {n_params_total:,}")
    print(f"  Readout parameters: {n_params_readout:,}  (the only trainable ones)")

    # ── pre-trained weights ───────────────────────────────────────────────────
    if not args.no_pretrained:
        h5_path = args.bl_weights
        _ensure_h5py()
        if not os.path.isfile(h5_path):
            print(f"\nbl_imagenet.h5 not found in {display_path(h5_path)}")
            ok = _download_bl_imagenet(h5_path)
            if not ok:
                print("  [WARN] Proceeding without pre-trained weights.")
        if os.path.isfile(h5_path):
            print(f"\nLoading pre-trained weights from {display_path(h5_path)}")
            load_blnet_pretrained_weights(model, h5_path, verbose=True)
        else:
            print("  [WARN] bl_imagenet.h5 unavailable — random init.")
    else:
        print("  Pre-trained weights disabled (--no_pretrained).")

    # ── training ──────────────────────────────────────────────────────────────
    if not args.eval_only:
        train_loader = _build_loader(train_ds, train=True,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers)
        val_loader   = _build_loader(val_ds, train=False,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers)

        train_metrics = train_convrnn(
            model       = model,
            train_loader= train_loader,
            val_loader  = val_loader,
            device      = device,
            n_epochs    = args.n_epochs,
            lr          = args.lr,
            eps         = args.adam_eps,
            save_dir    = ckpt_dir,
            save_freq   = args.save_freq,
            eval_freq   = args.eval_freq,
            debug       = args.debug,
        )

        # save the training metrics
        metrics_path = os.path.join(exp_root, "train_metrics.json")
        with open(metrics_path, "w") as mf:
            # "train" holds per-epoch floats, "val" (epoch, value) pairs
            json.dump(train_metrics, mf, indent=2)
        print(f"  Training metrics saved: {display_path(metrics_path)}")

        keep_final_checkpoint(ckpt_dir, "convrnn", args.perturb,
                              pattern="ckpt__epoch_*.pt")

    # ── load checkpoint ─────────────────────────────────────────────────────
    # The checkpoint a training run just saved (or, with --refresh_weights, a
    # previous one), else the released weights in competitors/weights/sat/.
    # Both layouts are `{"epoch", "model_state", "optim_state"}`.
    selected_ckpt = resolve_sat_checkpoint(
        model="convrnn", perturb=args.perturb, explicit=args.weights,
        refresh_weights=args.refresh_weights or not args.eval_only,
    )
    ckpt_data = torch.load(selected_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state"])
    print(f"  Checkpoint epoch {ckpt_data.get('epoch', '?')} loaded.")

    # ── evaluation ────────────────────────────────────────────────────────────
    test_loader = _build_loader(test_ds, train=False,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    output_reps = evaluate_convrnn(model, test_loader, device)

    # save output_representations (same format as cnet/eval.py)
    out_rep_path = os.path.join(out_dir, "output_representations__test__OSD.pt")
    torch.save(output_reps, out_rep_path)
    print(f"Saved: {display_path(out_rep_path)}")

    # ── SAT curve (plot per timestep, as in Analysis.py) ──────────────────────
    correct = output_reps["correct"]                        # (T, N)
    accs    = correct.float().mean(dim=1)                   # (T,)

    t_steps = list(range(_N_TIMESTEPS))
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t_steps, accs.numpy() * 100, marker="o", linewidth=2,
             label=f"ConvRNN BL-Net  [{args.perturb}]")
    plt.axhline(_CHANCE * 100, ls="--", color="grey", linewidth=1,
                label=f"chance ({_CHANCE*100:.2f}%)")
    for ms, ex in sorted(_BEEP_MS_TO_EXIT.items()):
        plt.axvline(ex, color="orange", ls=":", lw=0.8, alpha=0.7)
        plt.text(ex + 0.05, 2, f"{ms}ms", fontsize=6, color="orange", va="bottom")
    plt.xlabel("Timestep")
    plt.ylabel("Accuracy (%)")
    plt.title(f"SAT Curve — ConvRNN BL-Net  [{args.perturb}]")
    plt.xticks(t_steps)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(figs_dir, f"SAT_curve_{args.perturb}.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved: {display_path(curve_path)}")

    # ── SAT benchmark analysis ────────────────────────────────────────────────
    run_convrnn_sat_analysis(
        test_ds     = test_ds,
        output_reps = output_reps,
        perturb     = args.perturb,
        save_dir    = sat_save_dir,
        ranking_key = "convrnn",
        refresh     = refresh,
    )

    shutil.rmtree(exp_root, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
