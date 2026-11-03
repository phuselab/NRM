"""
competitors/rtify/train_rdm.py
==============================
Interface between the RTify training pipeline and the NRM data pipeline.

Weights live in CKPT_DIR (competitors/weights/rdm/): rtify_backbone.pth,
rtify_backbone_states.pth (the ~33 GB cache of backbone states, not shipped)
and rtify_head.pth. Every phase that trains or recomputes something writes the
``_REFRESH`` twin of the file it produces (rtify_backbone_REFRESH.pth, ...);
the released files are never overwritten. Periodic checkpoints go to a scratch
directory that is removed at the end, so only the final weights are kept.

Workflow
--------
Phase 1 – Backbone training
    Train AlexNet_BN_LSTM (AlexNet + BatchNorm + LSTM) on *synthetic* RDM
    videos generated on-the-fly via RTify's RDM_generator.
    Curriculum: only 99.9% coherence for the first 10 epochs, then all levels.
    Final checkpoint: CKPT_DIR/rtify_backbone_REFRESH.pth

Phase 2 – Precompute backbone outputs
    Load the trained backbone, set model.output=True so it returns
    LSTM hidden states instead of logits, generate 2048 synthetic videos per
    coherence level, and save the (T, 4096) LSTM trajectories.
    File: CKPT_DIR/rtify_backbone_states_REFRESH.pth

Phase 3 – Human RT distributions
    Build per-coherence signed-RT histograms from teddm_data.csv using the
    EXACT same train/val/test split as NRM (stratified by subID ×
    coherence_idx).  signed_RT = RT * (2*correct − 1):
      positive → correct decision, negative → error.
    The split is taken from the pkl saved by build_rdm_nrm_dataset().

Phase 4 – RTify supervised training
    Load backbone checkpoint into AlexNet_BN_LSTM_wrapper.
    Freeze fc1 and fc2 (backbone decision heads).
    Train only:  linear (evidence function), w, b (temporal re-weighting),
                 and threshold.
    Loss: MSE between the model's soft signed-RT histogram and the human
          signed-RT histogram, computed per coherence.
    Optimiser: Adam with per-group learning rates.

    Best-validation head: CKPT_DIR/rtify_head_REFRESH.pth

Phase 5 – Evaluation
    Compare model signed-RT distributions vs human test distributions.
    Report accuracy, mean RT, and KS distance per coherence.
    Save histogram plots to outputs/rdm/rtify[_REFRESH]/.

Each phase consumes the ``_REFRESH`` output of an earlier phase that ran in
the same invocation, and the released file otherwise (``--refresh-weights``
makes every phase consume the ``_REFRESH`` files of a previous run).

Usage
-----
    python competitors/rtify/train_rdm.py [--skip-backbone] [--skip-precompute]
                                          [--skip-rtify] [--only-eval]
                                          [--refresh-weights]
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# ── Import RTify modules (adjust sys.path before any RTify import) ─────────────
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "RDM"))
sys.path.insert(0, str(_HERE / "dataset"))

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# RTify internals
from utils import set_seeds, SoftHistogram
from RNN_decision import DiffDecision
from data_loader import RDMDataset as SyntheticRDMDataset
from RDM_generator import RDM_generator

# NRM data pipeline
from datasets.rdm.loader import (
    build_rdm_nrm_dataset,
    load_rdm_nrm_dataset,
    PKL_OUT,
)
from scripts.results_io import benchmark_dir, display_path, refresh_path, refresh_tag


# ══════════════════════════════════════════════════════════════════════════════
# Constants and models
# ══════════════════════════════════════════════════════════════════════════════

# Architectures and the shared constants live in rtify_model.py so the pipeline
# notebooks can import them without pulling in this training script.
from rtify_model import (          # noqa: E402
    AlexNet_BN_LSTM,
    AlexNet_BN_LSTM_wrapper,
    BACKBONE_NAME,
    CKPT_DIR,
    COHERENCE_LIST,
    DOT_SIZE,
    DT,
    FS,
    GROUPS,
    N_COH,
    N_DOT,
    SPEED,
    STIM_SIZE,
    TIME_STEPS,
)

BACKBONE_PATH = CKPT_DIR / "rtify_backbone.pth"
STATES_PATH = CKPT_DIR / "rtify_backbone_states.pth"
HEAD_PATH = CKPT_DIR / "rtify_head.pth"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 – Backbone training
# ══════════════════════════════════════════════════════════════════════════════

def train_backbone(config: dict, device: torch.device) -> None:
    """
    Train AlexNet_BN_LSTM on synthetic RDM videos for *config['num_epochs']* epochs.

    Curriculum: coherence=99.9% only for the first 10 epochs, then all levels.
    Saves one checkpoint per epoch to a scratch directory and keeps the last
    one as config['backbone_path'].
    """
    ckpt_dir    = Path(config["scratch_dir"]) / BACKBONE_NAME
    final_ckpt  = Path(config["backbone_path"])

    if final_ckpt.exists() and not config.get("force_retrain", False):
        print(f"[Backbone] Already trained — {display_path(final_ckpt)}")
        print("  Set force_retrain=True to retrain from scratch.")
        return

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(config.get("seed", 42))

    model     = AlexNet_BN_LSTM(time_steps=TIME_STEPS).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: 1.0 if ep < 10 else 0.1
    )

    all_coherences = config.get(
        "coherence_list", [99.9, 51.2, 25.6, 12.8, 6.4, 3.2, 1.6, 0.8]
    )
    dataset_size = config.get("dataset_size", 5000)
    batch_size   = config.get("batch_size", 50)
    num_epochs   = config["num_epochs"]

    print(f"\n{'='*70}")
    print(f"PHASE 1 — Backbone training  ({num_epochs} epochs)")
    print(f"{'='*70}")

    epoch_acc_list  = []
    epoch_loss_list = []

    for epoch in tqdm(range(num_epochs), desc="Backbone"):
        model.train()
        active_coh = [99.9] if epoch < 10 else all_coherences
        total_loss = total_acc = total_n = 0.0

        for coh in active_coh:
            dataset    = SyntheticRDMDataset(
                coh, RDM_generator,
                size=dataset_size, img_size=STIM_SIZE, num_frames=TIME_STEPS,
                speed=SPEED, dot_size=DOT_SIZE, groups=GROUPS, nDot=N_DOT,
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for seqs, directions in dataloader:
                seqs       = (seqs.to(device) / 255.0)
                directions = directions.to(device)

                optimizer.zero_grad()
                outputs = model(seqs)
                loss    = criterion(outputs, directions)
                loss.backward()
                optimizer.step()

                bs          = seqs.size(0)
                total_loss += loss.item() * bs
                total_acc  += ((outputs > 0) == (directions > 0)).float().sum().item()
                total_n    += bs

        scheduler.step()
        epoch_loss_list.append(total_loss / max(total_n, 1))
        epoch_acc_list.append(total_acc  / max(total_n, 1))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{num_epochs}"
                f"  loss={epoch_loss_list[-1]:.4f}"
                f"  acc={epoch_acc_list[-1]:.4f}"
            )

        torch.save(
            {
                "epoch":           epoch,
                "state_dict":      model.state_dict(),
                "optimizer":       optimizer.state_dict(),
                "epoch_acc_list":  epoch_acc_list,
                "epoch_loss_list": epoch_loss_list,
            },
            ckpt_dir / f"epoch_{epoch}.pth",
        )

    shutil.copyfile(ckpt_dir / f"epoch_{num_epochs - 1}.pth", final_ckpt)
    print(f"Backbone training complete.  Final checkpoint: {display_path(final_ckpt)}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 – Precompute backbone outputs (LSTM hidden states)
# ══════════════════════════════════════════════════════════════════════════════

def precompute_backbone_outputs(config: dict, device: torch.device) -> None:
    """
    Load trained backbone, generate synthetic RDM videos, store LSTM states.

    Output file stores:
        (backbone_outputs_list, direction_list)
    where backbone_outputs_list[i] has shape (dataset_size, T, 4096)
    and direction_list[i] has shape (dataset_size, 1).

    The ordering follows COHERENCE_LIST (descending coherence).
    """
    ckpt_path = Path(config["backbone_path"])
    out_path  = Path(config["states_path"])

    if out_path.exists() and not config.get("force_recompute", False):
        print(f"[Precompute] Already exists — {display_path(out_path)}")
        return

    print(f"\n{'='*70}")
    print(f"PHASE 2 — Precomputing backbone outputs  ({ckpt_path.name})")
    print(f"{'='*70}")

    model = AlexNet_BN_LSTM(time_steps=TIME_STEPS).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.output = True   # return LSTM hidden states

    coherence_list = config.get("coherence_list", COHERENCE_LIST)
    dataset_size   = config.get("precompute_dataset_size", 2048)
    batch_size     = config.get("precompute_batch_size", 256)

    backbone_outputs_list: list[torch.Tensor] = []
    direction_list:        list[torch.Tensor] = []

    with torch.no_grad():
        for coh in tqdm(coherence_list, desc="Coherences"):
            dataset    = SyntheticRDMDataset(
                coh, RDM_generator,
                size=dataset_size, img_size=STIM_SIZE, num_frames=TIME_STEPS,
                speed=SPEED, dot_size=DOT_SIZE, groups=GROUPS, nDot=N_DOT,
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            all_out: list[torch.Tensor] = []
            all_dir: list[torch.Tensor] = []

            for seqs, directions in dataloader:
                seqs = seqs.to(device) / 255.0
                out  = model(seqs).cpu()   # (B, T, 4096)
                all_out.append(out)
                all_dir.append(directions.cpu())

            backbone_outputs_list.append(torch.cat(all_out, dim=0))  # (N, T, 4096)
            direction_list.append(torch.cat(all_dir, dim=0))         # (N, 1)

    torch.save((backbone_outputs_list, direction_list), str(out_path))
    print(f"Backbone outputs saved → {display_path(out_path)}")
    print(f"  Per-coherence shape: {backbone_outputs_list[0].shape}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 – Human RT distributions from teddm_data.csv
# ══════════════════════════════════════════════════════════════════════════════

def build_human_rt_distributions(
    df:             pd.DataFrame,
    histogram:      SoftHistogram,
    device:         torch.device,
    coherence_list: list[float] | None = None,
) -> dict:
    """
    Build per-coherence signed-RT histograms from human behavioural data.

    signed_RT = RT * (2*correct − 1):
        positive values → correct decisions
        negative values → errors

    The histograms are computed separately for each split (train / val / test)
    using the split labels produced by NRM's build_rdm_nrm_dataset().

    Returns
    -------
    dict with keys 'train', 'val', 'test' — each a (N_COH, n_bins) tensor,
    ordered according to *coherence_list* (descending by default).
    Also adds 'mean_rt_correct' and 'accuracy' lists for reporting.
    """
    if coherence_list is None:
        coherence_list = COHERENCE_LIST

    print("\nBuilding human RT distributions ...")
    result: dict = {}

    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split]
        dists: list[torch.Tensor] = []

        for coh_pct in coherence_list:
            coh_df  = split_df[np.isclose(split_df["coherence"].values, coh_pct)]
            rt      = torch.tensor(coh_df["rt"].values,      dtype=torch.float32)
            correct = torch.tensor(coh_df["correct"].values, dtype=torch.float32)
            signed  = rt * (2.0 * correct - 1.0)
            dist    = histogram(signed.to(device)).detach()   # (n_bins,)
            dists.append(dist)

        result[split] = torch.stack(dists)   # (N_COH, n_bins)
        n_tot = len(split_df)
        print(f"  [{split:5s}]  {n_tot:6d} trials  →  histogram shape {result[split].shape}")

    # Extra summary statistics (not tensors — just for logging)
    result["mean_rt_correct"] = []
    result["accuracy"]        = []
    for coh_pct in coherence_list:
        coh_df = df[np.isclose(df["coherence"].values, coh_pct)]
        result["accuracy"].append(coh_df["correct"].mean())
        corr_rt = coh_df.loc[coh_df["correct"] == 1, "rt"]
        result["mean_rt_correct"].append(corr_rt.mean() if len(corr_rt) > 0 else float("nan"))

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 – RTify supervised training
# ══════════════════════════════════════════════════════════════════════════════

def train_rtify_supervised(
    model:           AlexNet_BN_LSTM_wrapper,
    optimizer:       optim.Optimizer,
    histogram:       SoftHistogram,
    human_rt_dists:  dict,
    config:          dict,
    device:          torch.device,
) -> dict:
    """
    Train RTify wrapper to match human signed-RT distributions via MSE loss.

    Faithful to AlexNet_BN_LSTM_sup.py::train_model(), adapted for our data.
    """
    exp_dir = Path(config["scratch_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(config["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pre-computed backbone outputs (kept on CPU to save VRAM) ────────
    outputs_path = Path(config["outputs_path"])
    print(f"\nLoading backbone outputs from {display_path(outputs_path)} ...")
    backbone_outputs_list, direction_list = torch.load(
        str(outputs_path), weights_only=False
    )
    # backbone_outputs_list[i]: (dataset_size, T, 4096) — stays on CPU
    # direction_list[i]:        (dataset_size, 1)

    coherence_list  = config.get("coherence_list", COHERENCE_LIST)
    num_epochs      = config["num_epochs"]
    batch_size      = config["batch_size"]
    dataset_size    = config["dataset_size"]

    human_train = human_rt_dists["train"].to(device)   # (N_COH, n_bins)
    human_val   = human_rt_dists["val"].to(device)

    print(f"\n{'='*70}")
    print(f"PHASE 4 — RTify supervised training  ({num_epochs} epochs)")
    print(f"{'='*70}")
    print(f"  Batch size : {batch_size}  |  Pool size : {dataset_size}")

    epoch_loss_list          : list[float] = []
    epoch_acc_list           : list[float] = []
    epoch_decision_time_list : list[float] = []
    signed_RT_coh_list       = [[] for _ in range(N_COH)]

    best_val_loss   = float("inf")
    log_interval    = max(1, num_epochs // 100)   # ~100 log lines total
    save_interval   = max(1, num_epochs // 10)    # 10 intermediate checkpoints

    for epoch in tqdm(range(num_epochs), desc="RTify training"):
        model.train()
        total_loss = total_acc = total_dt = 0.0

        for i_coh, _ in enumerate(coherence_list):
            n_pool  = min(dataset_size, backbone_outputs_list[i_coh].shape[0])
            indices = torch.randperm(n_pool)[:batch_size]

            lstm_batch = backbone_outputs_list[i_coh][indices].to(device)   # (B, T, 4096)
            dir_batch  = direction_list[i_coh][indices].to(device)          # (B, 1)

            optimizer.zero_grad()
            outputs, decision_time = model(lstm_batch)

            correct = ((outputs > 0) == (dir_batch > 0)).squeeze()         # (B,)
            signed  = decision_time * model.dt * (2.0 * correct.float() - 1.0)
            loss    = F.mse_loss(histogram(signed), human_train[i_coh])

            loss.backward()
            optimizer.step()

            # re-compute correct after graph is freed (no grad needed)
            with torch.no_grad():
                correct_det = ((outputs > 0) == (dir_batch > 0)).squeeze()
                signed_RT_coh_list[i_coh].append(
                    (decision_time * (2.0 * correct_det.float() - 1.0)).cpu().numpy()
                )

            total_loss += loss.item()
            total_acc  += correct.float().mean().item()
            total_dt   += decision_time.mean().item()

        epoch_loss_list.append(total_loss / N_COH)
        epoch_acc_list.append(total_acc  / N_COH)
        epoch_decision_time_list.append(total_dt / N_COH * DT)

        # ── Validation & logging ──────────────────────────────────────────
        if (epoch + 1) % log_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i_coh, _ in enumerate(coherence_list):
                    n_pool  = min(batch_size, backbone_outputs_list[i_coh].shape[0])
                    indices = torch.arange(n_pool)
                    out, dt = model(backbone_outputs_list[i_coh][indices].to(device))
                    corr    = ((out > 0) == (direction_list[i_coh][indices].to(device) > 0)).squeeze()
                    signed  = dt * model.dt * (2.0 * corr.float() - 1.0)
                    val_loss += F.mse_loss(histogram(signed), human_val[i_coh]).item()
                val_loss /= N_COH

            print(
                f"  Epoch {epoch+1:5d}/{num_epochs}"
                f"  train_loss={epoch_loss_list[-1]:.6f}"
                f"  val_loss={val_loss:.6f}"
                f"  acc={epoch_acc_list[-1]:.3f}"
                f"  mean_RT={epoch_decision_time_list[-1]*1000:.1f}ms"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer":  optimizer.state_dict(),
                        "epoch":      epoch,
                        "val_loss":   val_loss,
                    },
                    exp_dir / "best_model.pth",
                )

        # ── Periodic checkpoint ───────────────────────────────────────────
        if (epoch + 1) % save_interval == 0:
            torch.save(
                {
                    "state_dict":               model.state_dict(),
                    "optimizer":                optimizer.state_dict(),
                    "epoch":                    epoch,
                    "epoch_loss_list":          epoch_loss_list,
                    "epoch_acc_list":           epoch_acc_list,
                    "epoch_decision_time_list": epoch_decision_time_list,
                    "signed_RT_coh_list":       signed_RT_coh_list,
                },
                exp_dir / f"epoch_{epoch + 1}.pth",
            )

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final_state = {
        "state_dict":               model.state_dict(),
        "optimizer":                optimizer.state_dict(),
        "epoch":                    num_epochs,
        "epoch_loss_list":          epoch_loss_list,
        "epoch_acc_list":           epoch_acc_list,
        "epoch_decision_time_list": epoch_decision_time_list,
        "signed_RT_coh_list":       signed_RT_coh_list,
    }
    torch.save(final_state, exp_dir / "epoch_final.pth")

    # Keep the best-validation head (the final one if validation never ran).
    best_ckpt = exp_dir / "best_model.pth"
    shutil.copyfile(best_ckpt if best_ckpt.exists() else exp_dir / "epoch_final.pth",
                    config["head_path"])

    # ── Training curves plot ──────────────────────────────────────────────────
    _plot_training_curves(
        epoch_loss_list, epoch_acc_list, epoch_decision_time_list,
        save_path=plots_dir / "training_curves.png",
    )

    print(f"\nRTify training complete.  Head: {display_path(config['head_path'])}")
    return final_state


def _plot_training_curves(losses, accs, mean_rts, save_path: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(losses, color="steelblue")
    axs[0].set_title("Training loss (MSE)")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss")

    axs[1].plot(accs, color="firebrick")
    axs[1].set_title("Training accuracy")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Acc")
    axs[1].set_ylim(0, 1)

    axs[2].plot([r * 1000 for r in mean_rts], color="seagreen")
    axs[2].set_title("Mean decision time")
    axs[2].set_xlabel("Epoch"); axs[2].set_ylabel("ms")

    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Training curves → {display_path(save_path)}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 – Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_rtify(
    model:                 AlexNet_BN_LSTM_wrapper,
    histogram:             SoftHistogram,
    human_rt_dists:        dict,
    backbone_outputs_list: list[torch.Tensor],
    direction_list:        list[torch.Tensor],
    config:                dict,
    device:                torch.device,
) -> dict:
    """
    Compare model signed-RT distributions against human test distributions.

    Reports per-coherence: accuracy, mean correct RT, KS distance.
    Saves signed-RT histogram plots.
    """
    plots_dir = Path(config["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    coherence_list = config.get("coherence_list", COHERENCE_LIST)
    n_sample       = config.get("eval_n_sample", 1024)

    model.eval()
    human_test = human_rt_dists["test"].to(device)   # (N_COH, n_bins)
    bins_np    = histogram.centers.cpu().numpy()

    print(f"\n{'='*70}")
    print("PHASE 5 — Evaluation")
    print(f"{'='*70}")
    print(f"  {'Coh%':>6}  {'Acc':>6}  {'RT_model':>9}  {'RT_human':>9}  {'KS':>6}")

    results: dict = {}
    ks_vals, acc_vals = [], []

    with torch.no_grad():
        for i_coh, coh in enumerate(coherence_list):
            n_avail = backbone_outputs_list[i_coh].shape[0]
            indices = torch.randperm(n_avail)[:min(n_sample, n_avail)]

            lstm_batch = backbone_outputs_list[i_coh][indices].to(device)
            dir_batch  = direction_list[i_coh][indices].to(device)

            out, dt = model(lstm_batch)
            correct  = ((out > 0) == (dir_batch > 0)).squeeze()   # (B,)
            signed   = dt * model.dt * (2.0 * correct.float() - 1.0)

            model_hist = histogram(signed).cpu()
            human_hist = human_test[i_coh].cpu()

            # KS distance from empirical CDFs
            ks = float((torch.cumsum(model_hist, 0) - torch.cumsum(human_hist, 0)).abs().max())

            acc    = correct.float().mean().item()
            rt_m   = (dt[correct] * model.dt).mean().item() if correct.any() else float("nan")
            rt_h   = human_rt_dists["mean_rt_correct"][i_coh]

            ks_vals.append(ks); acc_vals.append(acc)
            results[coh] = dict(
                ks=ks, acc=acc, mean_rt_model=rt_m, mean_rt_human=rt_h,
                model_hist=model_hist.numpy(), human_hist=human_hist.numpy(),
            )
            print(f"  {coh:6.1f}  {acc:6.3f}  {rt_m*1000:7.0f}ms  {rt_h*1000:7.0f}ms  {ks:6.4f}")

    print(f"\n  Mean KS     : {np.mean(ks_vals):.4f}")
    print(f"  Mean acc    : {np.mean(acc_vals):.3f}")

    # ── Signed-RT histogram plot ──────────────────────────────────────────────
    n_cols  = 4
    n_rows  = (N_COH + n_cols - 1) // n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    ax      = ax.flatten()

    for i_coh, coh in enumerate(coherence_list):
        r = results[coh]
        ax[i_coh].plot(bins_np, r["model_hist"], color="steelblue", label="RTify model")
        ax[i_coh].plot(bins_np, r["human_hist"],  color="firebrick", ls="--", label="Human data")
        ax[i_coh].set_title(
            f"Coh={coh}%  acc={r['acc']:.2f}  KS={r['ks']:.4f}",
            fontsize=9,
        )
        ax[i_coh].set_xlabel("Signed RT (s)"); ax[i_coh].set_ylabel("Freq.")
        ax[i_coh].set_xlim(-2.5, 2.5)
        ax[i_coh].legend(fontsize=7)

    for j in range(N_COH, len(ax)):
        ax[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(plots_dir / "signed_rt_histograms.png", dpi=150)
    plt.close(fig)
    print(f"\n  Histogram plot → {display_path(plots_dir / 'signed_rt_histograms.png')}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def setup_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_or_build_df(force_rebuild: bool = False) -> pd.DataFrame:
    """Load the human behavioural DataFrame, build from CSV if missing."""
    if PKL_OUT.exists() and not force_rebuild:
        df = pd.read_pickle(str(PKL_OUT))
        print(f"Loaded RDM metadata from {display_path(PKL_OUT)}  ({len(df)} rows)")
    else:
        print("Building RDM metadata from teddm_data.csv ...")
        ds = build_rdm_nrm_dataset()
        ds.remove_rt_outliers_mad(mad_thresh=3.0, min_rt=0.05, max_rt=3.0)
        df = ds.df
    return df


def build_rtify_model(
    device:        torch.device,
    backbone_path: Path = BACKBONE_PATH,
) -> tuple[AlexNet_BN_LSTM_wrapper, optim.Optimizer]:
    """
    Instantiate wrapper, load fc1/fc2 from backbone checkpoint, freeze them,
    and build the Adam optimiser with per-group learning rates.
    """
    model     = AlexNet_BN_LSTM_wrapper(time_steps=TIME_STEPS, dt=DT).to(device)
    ckpt_path = Path(backbone_path)

    print(f"Loading backbone weights from {display_path(ckpt_path)} ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    # strict=False: fc1/fc2 are loaded; linear/w/b/threshold are freshly initialised
    model.load_state_dict(ckpt["state_dict"], strict=False)

    for param in model.fc1.parameters():
        param.requires_grad = False
    for param in model.fc2.parameters():
        param.requires_grad = False

    optimizer = optim.Adam([
        {"params": model.linear.parameters(), "lr": 1e-5},
        {"params": [model.w, model.b],        "lr": 1e-4},
        {"params": [model.threshold],         "lr": 2e-4},
    ])

    return model, optimizer


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTify RDM training pipeline")
    p.add_argument("--skip-backbone",   action="store_true",
                   help="Skip Phase 1 (backbone training)")
    p.add_argument("--skip-precompute", action="store_true",
                   help="Skip Phase 2 (backbone output precomputation)")
    p.add_argument("--skip-rtify",      action="store_true",
                   help="Skip Phase 4 (RTify supervised training)")
    p.add_argument("--only-eval",       action="store_true",
                   help="Run only Phase 5 (evaluation) with the trained head")
    p.add_argument("--refresh-weights", action="store_true",
                   help="Consume the _REFRESH backbone / states / head of a "
                        "previous run instead of the released ones")
    p.add_argument("--rtify-epochs",    type=int, default=10000,
                   help="Number of RTify training epochs (default: 10000)")
    p.add_argument("--force-retrain",   action="store_true",
                   help="Force backbone retraining even if rtify_backbone_REFRESH.pth exists")
    p.add_argument("--force-recompute", action="store_true",
                   help="Force backbone output recomputation even if rtify_backbone_states_REFRESH.pth exists")
    p.add_argument("--rebuild-data",    action="store_true",
                   help="Rebuild dataset from CSV even if pkl exists")
    p.add_argument("--device",          type=str, default=None,
                   help="Torch device (default: cuda:0 if available)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    set_seeds(42)

    if args.device:
        device = torch.device(args.device)
    else:
        device = setup_device()

    # ── Which files each phase reads and writes ───────────────────────────────
    # A phase that runs writes the _REFRESH twin of its output; a later phase
    # consumes it. Phases that are skipped leave the released files in use,
    # unless --refresh-weights asks for those of a previous run.
    train_bb   = not (args.skip_backbone or args.only_eval)
    precompute = not (args.skip_precompute or args.only_eval)
    train_head = not (args.skip_rtify or args.only_eval)
    bb_refresh     = args.refresh_weights or train_bb
    states_refresh = bb_refresh or precompute
    head_refresh   = states_refresh or train_head

    backbone_path = refresh_path(BACKBONE_PATH, bb_refresh)
    states_path   = refresh_path(STATES_PATH, states_refresh)
    head_path     = refresh_path(HEAD_PATH, head_refresh)
    plots_dir     = benchmark_dir("rdm") / f"rtify{refresh_tag(head_refresh)}"
    scratch_dir   = Path(tempfile.mkdtemp(prefix="rtify_rdm_"))

    # ── Configs ───────────────────────────────────────────────────────────────
    backbone_config = {
        "num_epochs":    101,   # epoch_0 … epoch_100; the last one is kept
        "batch_size":    50,
        "dataset_size":  5000,
        "coherence_list": [99.9, 51.2, 25.6, 12.8, 6.4, 3.2, 1.6, 0.8],
        "seed":          42,
        "force_retrain": args.force_retrain,
        "scratch_dir":   str(scratch_dir),
        "backbone_path": str(backbone_path),
    }

    precompute_config = {
        "backbone_path":           str(backbone_path),
        "states_path":             str(states_path),
        "coherence_list":          COHERENCE_LIST,
        "precompute_dataset_size": 2048,
        "precompute_batch_size":   256,
        "force_recompute":         args.force_recompute,
    }

    rtify_config = {
        "num_epochs":      args.rtify_epochs,
        "batch_size":      1024,
        "dataset_size":    2048,
        "coherence_list":  COHERENCE_LIST,
        "outputs_path":    str(states_path),
        "scratch_dir":     str(scratch_dir / "head"),
        "plots_dir":       str(plots_dir),
        "head_path":       str(head_path),
    }

    eval_config = {
        "coherence_list":  COHERENCE_LIST,
        "eval_n_sample":   1024,
        "plots_dir":       str(plots_dir),
    }

    # ── Data: load/build human behavioural DataFrame ──────────────────────────
    print(f"\n{'='*70}")
    print("DATA — Human behavioural data (teddm_data.csv / NRM splits)")
    print(f"{'='*70}")
    df = load_or_build_df(force_rebuild=args.rebuild_data)

    histogram = SoftHistogram(500, -2.5, 2.5, 40.0).to(device)

    human_rt_dists = build_human_rt_distributions(df, histogram, device, COHERENCE_LIST)

    # ── Phase 1: Backbone training ────────────────────────────────────────────
    if train_bb:
        train_backbone(backbone_config, device)

    # ── Phase 2: Precompute backbone outputs ──────────────────────────────────
    if precompute:
        precompute_backbone_outputs(precompute_config, device)

    # ── Phase 4: RTify supervised training ────────────────────────────────────
    if train_head:
        model, optimizer = build_rtify_model(device, backbone_path=backbone_path)
        train_rtify_supervised(
            model, optimizer, histogram, human_rt_dists, rtify_config, device
        )

    # ── Phase 5: Evaluation ───────────────────────────────────────────────────
    # Load backbone outputs for evaluation (needed regardless of training path)
    print(f"\nLoading backbone outputs for evaluation from {display_path(states_path)} ...")
    backbone_outputs_list, direction_list = torch.load(
        str(states_path), weights_only=False
    )

    if not train_head:
        model, _ = build_rtify_model(device, backbone_path=backbone_path)
        ckpt     = torch.load(head_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded RTify weights from {display_path(head_path)}")

    evaluate_rtify(
        model, histogram, human_rt_dists,
        backbone_outputs_list, direction_list,
        eval_config, device,
    )
    shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
