#!/usr/bin/env python3
"""
competitors/crnn/train_sc.py
============================
Train hGRU (C-RBP) on SUN_filtered, map ξ_cRNN → RT on experiment-1 val split,
evaluate on experiment-1 test split.

Sections
--------
1. Data — SUN_filtered (train/val ImageFolder) + experiment-1 splits
2. hGRU training with C-RBP + EDL + LCP penalty
3. ξ_cRNN computation and RT mapper training (4 seeds)
4. Evaluation on experiment-1 test split → crnn_metrics_*.csv
"""

import os
import sys
import warnings
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Vendored cRNN modules ─────────────────────────────────────────────────────
# Loaded by file path rather than by name. The upstream tree uses the generic
# top-level names `models` and `utils`, which collide with other vendored
# competitors (competitors/rtify/utils.py) and with any `utils` already in
# sys.modules — whichever was imported first would win, and `from utils.loss
# import ...` would then fail with "'utils' is not a package". These three
# files have no intra-package imports, so loading them directly is exact.
import importlib.util as _ilu

_RNN_RTS = os.path.dirname(os.path.abspath(__file__))


def _load_vendored(mod_name, relative_path):
    spec = _ilu.spec_from_file_location(
        f"competitors.crnn._vendored.{mod_name}",
        os.path.join(_RNN_RTS, relative_path),
    )
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hGRU = _load_vendored("hgru", os.path.join("models", "hgru.py")).hGRU
_crnn_loss = _load_vendored("loss", os.path.join("utils", "loss.py"))
EDLLoss, get_edl_vars, exp_evidence = (
    _crnn_loss.EDLLoss, _crnn_loss.get_edl_vars, _crnn_loss.exp_evidence)
AverageMeter = _load_vendored("misc", os.path.join("utils", "misc.py")).AverageMeter

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# First on sys.path even if already present further down: this directory holds
# a vendored `datasets` package that would otherwise shadow the repository's.
sys.path.insert(0, _REPO_ROOT)

# ── Project imports ────────────────────────────────────────────────────────────
from datasets.sc.loader import SCDataset, sc_collate_fn      # noqa: E402
from scripts.eval_sc import (                                           # noqa: E402
    PROB_LIST,
    compute_human_nc_global, compute_human_nc_per_prob,
)
from scripts.results_io import display_path, refresh_path, refresh_tag  # noqa: E402

try:
    from scipy.stats import wasserstein_distance
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — DATASETS
# ══════════════════════════════════════════════════════════════════════════════

HGRU_TRANSFORM = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),                        # [0,1], shape [1, 150, 150]
    transforms.Normalize(mean=[0.5], std=[0.5]),  # → [-1, 1]
])


class SUNFilteredDataset(Dataset):
    """
    SUN_filtered train or val split.
    Subfolder convention: natural/ → label 0,  manmade/ → label 1.
    Images are 150×150 greyscale.
    """

    def __init__(self, root: str, transform=None):
        self.transform = transform or HGRU_TRANSFORM
        self.class_to_idx = {"natural": 0, "manmade": 1}
        self.samples: list[tuple[str, int]] = []
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Missing subfolder: {cls_dir}")
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))
        print(f"  SUNFiltered ({display_path(root)}): {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


class SCImageDataset(Dataset):
    """
    Wraps a SCDataset split and loads actual images for hGRU inference.
    All fields from the SCDataset split are preserved.
    """

    def __init__(self, sc_ds: SCDataset, transform=None):
        self.ds = sc_ds
        self.transform = transform or HGRU_TRANSFORM

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        img = Image.open(item["image_path"]).convert("L")
        img = self.transform(img)
        item["image"] = img
        return item


def _sc_image_collate(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])
    base = sc_collate_fn([{k: v for k, v in b.items() if k != "image"} for b in batch])
    base["images"] = images
    return base


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — hGRU TRAINING
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _val_hgru(model, loader, edl, global_step, tau, device):
    model.eval()
    losses = AverageMeter()
    crit = partial(edl, global_step=global_step,
                   annealing_step=tau * max(1, len(loader)))
    for batch in loader:
        imgs   = batch["image"].to(device)
        target = batch["label"].to(device)
        od = model.forward(imgs, 0, 0, target, crit)
        losses.update(od["loss"].mean().item())
    model.train()
    return losses.avg


def train_hgru(
    model: hGRU,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    n_epochs: int,
    gamma: float,
    tau: int,
    device: torch.device,
    save_path: str,
) -> hGRU:
    """
    Train hGRU with C-RBP + EDL + LCP (Jacobian) penalty.
    Follows rnn_rts/train.py logic exactly; C-RBP backward is inside hGRU.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    edl = EDLLoss(num_classes=2).to(device)
    global_step = 1
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        losses = AverageMeter()

        for i, batch in enumerate(train_loader):
            imgs   = batch["image"].to(device)
            target = batch["label"].to(device)

            print(i)

            # Annealing: rho = global_step / (tau * n_batches)  (capped at 1)
            crit = partial(
                edl,
                global_step=global_step,
                annealing_step=tau * len(train_loader),
            )

            od = model.forward(imgs, epoch, i, target, crit)
            loss   = od["loss"].mean()
            jv_pen = od["jv_penalty"].mean()
            total  = loss + gamma * jv_pen

            total.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.update(total.item())
            global_step += 1

        val_loss = _val_hgru(model, val_loader, edl, global_step, tau, device)
        print(f"  [hGRU] epoch {epoch+1:3d}/{n_epochs}  "
              f"train={losses.avg:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"    → best saved (val={val_loss:.4f})")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ξ_cRNN COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_xi_and_choices(
    model: hGRU,
    images: torch.Tensor,   # [B, 1, 150, 150]
    device: torch.device,
    n_classes: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run hGRU in bptt testmode to collect all T internal states.

    ξ_cRNN = Σ_{t=1}^{T} ε_t,   ε_t = K / Σ_k α_k(t)
    alpha at final step → predicted choice and soft probability.

    Returns
    -------
    xi       : [B]  cumulative uncertainty
    choices  : [B]  argmax choice (0=natural, 1=manmade)
    probs1   : [B]  P(manmade) from final alpha
    """
    model.eval()
    # Use bptt for inference — C-RBP is a training-time approximation only
    original_grad = model.grad_method
    model.grad_method = "bptt"

    images = images.to(device)
    out = model.forward(images, 0, 0, testmode=True)
    states = out["states"]    # list of T+1 tensors [B, C, H, W] (includes t=0)
    output = out["output"]    # [B, n_classes] — final step readout

    # ξ_cRNN: accumulate uncertainty at each step AFTER the recurrent update
    xi = torch.zeros(images.shape[0], device=device)
    for state in states[1:]:        # skip t=0 (initial zeros state)
        logits = model.readout(state)                      # [B, 2]
        evidence = exp_evidence(logits)                    # [B, 2]
        alpha = evidence + 1.0                             # [B, 2]
        uncertainty = n_classes / alpha.sum(dim=1)         # [B]
        xi = xi + uncertainty

    # Final-step choices and soft probabilities
    evidence_f = exp_evidence(output)
    alpha_f    = evidence_f + 1.0
    probs_f    = alpha_f / alpha_f.sum(dim=1, keepdim=True)   # [B, 2]
    choices    = alpha_f.argmax(dim=1)                         # [B]
    probs1     = probs_f[:, 1]                                 # P(manmade)

    model.grad_method = original_grad
    return xi.cpu(), choices.cpu(), probs1.cpu()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RT MAPPER
# ══════════════════════════════════════════════════════════════════════════════

class RTMapper(nn.Module):
    """
    ξ_cRNN (scalar) → RT (seconds, scalar).
    Non-linear: one hidden layer with 16 units and ReLU.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1] or [B]
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.net(x).squeeze(1)   # [B]


def collect_xi_on_split(
    model: hGRU,
    sc_image_ds: SCImageDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Forward frozen hGRU on all images in `sc_image_ds`.

    Returns
    -------
    xi_all      : [N] ξ_cRNN for each trial
    human_rts   : [N] human RT in seconds (NaN excluded kept as 0.0)
    valid_mask  : [N] bool — True where rt is finite
    """
    loader = DataLoader(
        sc_image_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=_sc_image_collate,
    )

    xi_list, rt_list = [], []
    for batch in tqdm(loader, desc="  ξ_cRNN forward"):
        imgs = batch["images"]
        xi, _, _ = compute_xi_and_choices(model, imgs, device)
        xi_list.append(xi)
        rt_list.append(batch["rts"])

    xi_all    = torch.cat(xi_list)           # [N]
    human_rts = torch.cat(rt_list)           # [N]
    valid     = human_rts.isfinite().numpy() & (human_rts.numpy() > 0)
    return xi_all, human_rts, valid


def train_rt_mapper(
    xi_all: torch.Tensor,
    human_rts: torch.Tensor,
    valid_mask: np.ndarray,
    seed: int,
    n_epochs: int,
    lr: float,
    device: torch.device,
    save_path: str,
) -> RTMapper:
    """Train one RT mapping layer on (ξ, human_rt) pairs (val split only)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mapper = RTMapper().to(device)
    optimizer = optim.Adam(mapper.parameters(), lr=lr)
    mse = nn.MSELoss()

    xi_v  = xi_all[valid_mask].to(device)
    rt_v  = human_rts[valid_mask].to(device)

    mapper.train()
    for epoch in range(n_epochs):
        pred = mapper(xi_v)
        loss = mse(pred, rt_v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"    [RT mapper seed={seed}] epoch {epoch+1}/{n_epochs} "
                  f"MSE={loss.item():.6f}")

    torch.save(mapper.state_dict(), save_path)
    print(f"    → saved: {display_path(save_path)}")
    return mapper


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_eval_data(
    model: hGRU,
    mappers: list[RTMapper],
    sc_image_ds: SCImageDataset,
    batch_size: int,
    device: torch.device,
) -> dict:
    """
    Forward hGRU + 4 RT mappers on test split.
    Assembles the `data` dict expected by SCEvaluator.compute_metrics().
    """
    loader = DataLoader(
        sc_image_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=_sc_image_collate,
    )

    img_idxs, true_labs, hum_ch, pred_ch = [], [], [], []
    true_rt, pred_rt = [], []
    prob_vals: list[float] = []

    rt_samples_by_prob:     dict[float, list] = defaultdict(list)
    human_rts_by_prob:      dict[float, list] = defaultdict(list)
    trial_pred_rts_by_prob: dict[float, list] = defaultdict(list)
    pred_probs_by_prob:     dict[float, list] = defaultdict(list)
    model_info_by_prob: dict[float, dict] = defaultdict(
        lambda: {"pred_choices": [], "true_labels": [], "human_choices": []}
    )

    print("Collecting cRNN predictions on test set...")
    for batch in tqdm(loader, desc="  eval forward"):
        imgs    = batch["images"]
        b_probs = batch["probs"]                        # list[float]
        h_rts   = batch["rts"]                          # [B] in seconds

        xi, choices, probs1 = compute_xi_and_choices(model, imgs, device)

        # RT mapper ensemble: 4 predictions per trial
        xi_d = xi.to(device).unsqueeze(1)               # [B, 1]
        mapper_rts = torch.stack(
            [m(xi_d).cpu() for m in mappers], dim=1     # [B, 4]
        )
        med_rts = mapper_rts.median(dim=1).values        # [B]

        img_idxs.append(batch["image_index"])
        true_labs.append(batch["choices"])
        hum_ch.append(batch["human_choices"])
        pred_ch.append(choices)
        true_rt.append(h_rts)
        pred_rt.append(med_rts)
        prob_vals.extend(b_probs)

        for i, prob in enumerate(b_probs):
            rt4 = mapper_rts[i].tolist()                # 4 RT samples
            rt_samples_by_prob[prob].extend(rt4)
            human_rts_by_prob[prob].append(float(h_rts[i]))
            trial_pred_rts_by_prob[prob].append(float(med_rts[i]))
            pred_probs_by_prob[prob].append(float(probs1[i]))
            model_info_by_prob[prob]["pred_choices"].append(int(choices[i]))
            model_info_by_prob[prob]["true_labels"].append(int(batch["choices"][i]))
            model_info_by_prob[prob]["human_choices"].append(int(batch["human_choices"][i]))

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
    """Save crnn_metrics_per_prob{tag}.csv and crnn_metrics_global{tag}.csv."""
    os.makedirs(output_dir, exist_ok=True)

    # ── per-prob ──────────────────────────────────────────────────────────────
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
        os.path.join(output_dir, f"crnn_metrics_per_prob{tag}.csv"), index=False
    )

    # ── global ────────────────────────────────────────────────────────────────
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
    }]).to_csv(os.path.join(output_dir, f"crnn_metrics_global{tag}.csv"), index=False)

    print(f"\nCSVs saved to {display_path(output_dir)}/")


def print_paper_table(metrics: dict) -> None:
    glb = metrics["global"]
    nc  = glb.get("nc", {})

    def _f(v, fmt=".3f"):
        return "—" if not np.isfinite(float(v)) else format(float(v), fmt)

    W = 100
    print("\n" + "=" * W)
    print("PAPER TABLE — cRNN (hGRU, C-RBP) on SC task")
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
        f"{'cRNN (hGRU, C-RBP)':22s}  "
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
        p   = metrics["per_prob"][prob]
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
    TRAIN_DIR  = os.environ.get("SC_SUN_TRAIN_DIR", os.path.join(_REPO_ROOT, "datasets", "sc", "sun_filtered", "train"))
    VAL_DIR    = os.environ.get("SC_SUN_VAL_DIR", os.path.join(_REPO_ROOT, "datasets", "sc", "sun_filtered", "val"))
    BATCH_SIZE = 128

    # hGRU training
    T_STEPS   = 40      # recurrent steps
    LR        = 1e-3
    GAMMA     = 100     # LCP penalty weight
    TAU       = 16      # EDL annealing factor (epochs)
    N_EPOCHS  = 100

    # RT mapper training
    RT_MAP_EPOCHS = 50
    SEEDS         = [0, 1, 2, 3]
    RT_MAP_LR     = 1e-3

    # Misc
    RANDOM_SEED = 42
    OUTPUT_DIR  = os.path.join(_REPO_ROOT, "outputs", "sc")
    MODEL_DIR   = os.path.join(_REPO_ROOT, "competitors", "weights", "sc")

    TRAIN_HGRU       = True    # set False to skip to RT mapper / eval
    TRAIN_RT_MAPPERS = True   # set False to skip RT mapper training

    # Trained weights are saved with the _REFRESH suffix next to the released
    # ones (crnn_hgru_REFRESH.pt, crnn_mapper_seed<k>_REFRESH.pt), which are
    # never overwritten; the metrics CSVs of such a run carry the suffix too.
    # With TRAIN_HGRU = False the released hGRU is used.

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hgru_save_path = str(refresh_path(os.path.join(MODEL_DIR, "crnn_hgru.pt"),
                                      TRAIN_HGRU))

    # ── 1. SUN_filtered data ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1 — SUN_filtered data loaders")
    print("=" * 70)

    sun_train = SUNFilteredDataset(TRAIN_DIR)
    sun_val   = SUNFilteredDataset(VAL_DIR)

    sun_train_loader = DataLoader(
        sun_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=device.type == "cuda", drop_last=True,
    )
    sun_val_loader = DataLoader(
        sun_val, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda", drop_last=False,
    )

    # ── 2. hGRU training ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 — hGRU model")
    print("=" * 70)

    model = hGRU(
        timesteps=T_STEPS,
        filt_size=15,
        num_iter=15,
        exp_name="sc_crnn",
        jacobian_penalty=True,
        grad_method="rbp",          # C-RBP
        activ="softplus",
        xavier_gain=1.0,
        num_channels=25,
        n_in=1,                     # greyscale
        n_classes=2,
    ).to(device)

    if TRAIN_HGRU:
        print("\nTraining hGRU with C-RBP + EDL + LCP penalty...")
        optimizer = optim.Adam(model.parameters(), lr=LR)
        train_hgru(
            model=model,
            train_loader=sun_train_loader,
            val_loader=sun_val_loader,
            optimizer=optimizer,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            tau=TAU,
            device=device,
            save_path=hgru_save_path,
        )

    if not os.path.exists(hgru_save_path):
        raise FileNotFoundError(
            f"{display_path(hgru_save_path)} not found. The released hGRU is not "
            "distributed: set TRAIN_HGRU = True to train it.")
    print(f"\nLoading best hGRU from {display_path(hgru_save_path)}")
    model.load_state_dict(torch.load(hgru_save_path, map_location=device))
    model.eval()

    # ── 3. experiment-1 splits (identical to NRM) ─────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 — experiment-1 splits (same seed as NRM)")
    print("=" * 70)

    sc_dataset = SCDataset(
        csv_path=os.path.join(_REPO_ROOT, "datasets", "sc", "experiment_1.csv"),
        image_dir=os.path.join(_REPO_ROOT, "datasets", "sc", "stimuli"),
    )
    _, val_ds, test_ds = sc_dataset.train_val_test_split(
        val_size=0.10, test_size=0.20, seed=RANDOM_SEED,
    )

    val_img_ds  = SCImageDataset(val_ds)
    test_img_ds = SCImageDataset(test_ds)

    # ── 4. ξ_cRNN on val split + RT mapper training ───────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4 — RT mapper training on experiment-1 val split")
    print("=" * 70)

    mappers: list[RTMapper] = []

    for seed in SEEDS:
        mapper_path = str(refresh_path(
            os.path.join(MODEL_DIR, f"crnn_mapper_seed{seed}.pt"),
            TRAIN_HGRU or TRAIN_RT_MAPPERS))
        mapper = RTMapper().to(device)

        if TRAIN_RT_MAPPERS:
            print(f"\n  Seed {seed} — collecting ξ_cRNN on val split...")
            xi_val, rt_val, valid = collect_xi_on_split(
                model, val_img_ds, BATCH_SIZE, device
            )
            train_rt_mapper(
                xi_all=xi_val,
                human_rts=rt_val,
                valid_mask=valid,
                seed=seed,
                n_epochs=RT_MAP_EPOCHS,
                lr=RT_MAP_LR,
                device=device,
                save_path=mapper_path,
            )

        mapper.load_state_dict(torch.load(mapper_path, map_location=device))
        mapper.eval()
        mappers.append(mapper)

    # ── 5. Evaluation on test split ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5 — Evaluation on experiment-1 test split")
    print("=" * 70)

    data = build_eval_data(
        model=model,
        mappers=mappers,
        sc_image_ds=test_img_ds,
        batch_size=BATCH_SIZE,
        device=device,
    )

    # Compute metrics with SCEvaluator.compute_metrics directly
    # (bypasses SCEvaluator.collect_predictions which is NRM-specific)
    from scripts.eval_sc import SCEvaluator  # noqa
    evaluator = SCEvaluator(model=None, device=device)
    metrics   = evaluator.compute_metrics(data)

    save_csvs(metrics, OUTPUT_DIR, tag=refresh_tag(TRAIN_HGRU or TRAIN_RT_MAPPERS))
    print_paper_table(metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
