"""
competitors/cnet/train_sat.py

Adapts the original SATBench CNet code to train and evaluate
CNet on the same data (identical 70/10/20 split) used for NRM,
instead of the standard ImageNet dataset.

Identical workflow to the original CNet code:
  1. train.py  → train the model, save checkpoint in the expected directory structure
  2. eval.py   → load checkpoint, save output_representations__test__OSD.pt
  3. Analysis  → loads the .pt files, produces the SAT curve (accuracy per timestep)

The only difference from the original code is the data source:
  - SATDataset (NRM pipeline) instead of DataHandler/ImageNet
  - same remove_rt_outliers([-100,+100]), tutorial block removal, split seed=42

Technical note: CascadedTrainingScheme in train_handler.py has a device bug
  (torch.eye(n)[cuda_targets] fails because torch.eye is on CPU and targets on GPU).
  Fixed with a minimal subclass that keeps targets on CPU for
  indexing, exactly as already done correctly in the eval handler.

Usage:
  python competitors/cnet/train_sat.py --lambda_val 1.0 --perturb color-gray [--eval_only]   # parallel
  python competitors/cnet/train_sat.py --lambda_val 0.0 --perturb color-blur [--eval_only]   # serial
  python competitors/cnet/train_sat.py --lambda_val 1.0 --perturb gray-noise [--eval_only]
"""

import argparse
import shutil
import tempfile
import collections
import contextlib
import importlib.util
import os
import sys
import types
import json
import csv
import numpy as np
from pathlib import Path
import torch
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
CNET_ROOT  = _SCRIPT_DIR
# IMAGE_DIR can also be set via SAT_IMAGE_DIR environment variable
IMAGE_DIR = os.environ.get("SAT_IMAGE_DIR", os.path.join(REPO_ROOT, "datasets", "sat", "data"))


PERTURB_TO_FILE = {
    "color-gray": os.path.join(REPO_ROOT, "datasets/sat", "color-gray.txt"),
    "color-blur": os.path.join(REPO_ROOT, "datasets/sat", "color-blur.txt"),
    "gray-noise": os.path.join(REPO_ROOT, "datasets/sat", "gray-noise.txt"),
}

# ── released checkpoints ──────────────────────────────────────────────────────
# competitors/released_weights.py is loaded by file path instead of imported as
# `competitors.released_weights`: the block below puts this script's own
# vendored `models`/`modules`/`datasets` directories at the front of sys.path,
# so a package import would depend on the order that happened in.
_rw_spec = importlib.util.spec_from_file_location(
    "competitors_released_weights",
    os.path.join(REPO_ROOT, "competitors", "released_weights.py"),
)
released_weights = importlib.util.module_from_spec(_rw_spec)
_rw_spec.loader.exec_module(released_weights)
resolve_sat_checkpoint = released_weights.resolve_sat_checkpoint
keep_final_checkpoint = released_weights.keep_final_checkpoint

# ── sys.path for cnet submodules and project root ──────────────────────────────────
for _sub in ("models", "modules", "datasets"):
    _p = os.path.join(CNET_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if CNET_ROOT not in sys.path:
    sys.path.insert(0, CNET_ROOT)


# Per-model plots (the seven PNGs) and the per-model .tex/.csv are off by
# default: only sat_level_curves.json and satbench_metrics_full.json feed
# pipelines/sat.ipynb. `--diagnostics` turns them on.
WRITE_DIAGNOSTICS = False


@contextlib.contextmanager
def _vendored_datasets():
    """Make ``datasets`` mean ``competitors/cnet/datasets`` for the duration.

    CNet's vendored handlers import each other by that name
    (``from datasets import cifar_handler``, ``from datasets.dataset_handler
    import DataHandler``), and this repository also has a top-level
    ``datasets`` package holding the three benchmark loaders. CNet's directory
    carries no ``__init__.py``, so it is only a *namespace portion*: the import
    system collects it, keeps searching, finds the repo's regular package, and
    that one wins — no matter where CNet's directory sits on sys.path. CNet's
    ``train.py`` then dies on ``No module named 'datasets.dataset_handler'``.

    Binding the name explicitly, and only around the CNet imports, keeps both
    packages reachable without reordering sys.path for everything else —
    ``from datasets.sat.loader import ...`` further down still resolves to the
    repo's loader.
    """
    saved = {k: v for k, v in sys.modules.items()
             if k == "datasets" or k.startswith("datasets.")}

    def _drop_datasets():
        for k in [k for k in sys.modules
                  if k == "datasets" or k.startswith("datasets.")]:
            del sys.modules[k]

    shim = types.ModuleType("datasets")
    shim.__path__ = [os.path.join(CNET_ROOT, "datasets")]
    _drop_datasets()
    sys.modules["datasets"] = shim
    try:
        yield
    finally:
        _drop_datasets()
        sys.modules.update(saved)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset — same pipeline as scripts/train_nrm_sat.py
# ─────────────────────────────────────────────────────────────────────────────

class _SATImageDataset(Dataset):
    """
    Wrap a SATDataset subset (post-split) so it yields (image_tensor, label)
    in the format expected by CNet (same transforms as imagenet2012_handler.py).
    """
    _TRAIN_TFM = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    _EVAL_TFM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mock DataHandler — compatible with setup_model() in train.py
# ─────────────────────────────────────────────────────────────────────────────

class _MockDataHandler:
    """The minimal object setup_model() in train.py expects."""
    num_classes = 16


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training — uses setup_model() and condition_model() from train.py;
#    keeps CascadedTrainingScheme's targets on the right device
# ─────────────────────────────────────────────────────────────────────────────

def _patch_cascaded_training():
    """
    Fixes the device bug of CascadedTrainingScheme.__call__:
      torch.eye(num_classes)[cuda_targets]  →  crash because eye is on CPU

    Solution: keep targets on CPU for torch.eye indexing
    (identical to how CascadedEvalLoop correctly works in eval_handler.py),
    then moves y to the correct device before computing the loss.

    Does not modify any other behaviour of the original code.
    """
    from modules import train_handler
    import torch.nn.functional as F_

    OrigClass = train_handler.CascadedTrainingScheme

    class _FixedCascadedTrainingScheme(OrigClass):
        def __call__(self, net, loader, criterion, epoch_i, optimizer, device):
            net.train()
            batch_losses = []
            batch_accs   = []

            for batch_i, (data, targets) in enumerate(loader):
                if self.flags.debug and batch_i > 1:
                    break

                data    = data.to(device)
                # targets STAY on CPU for torch.eye indexing
                targets_cpu = targets.type(torch.int64)

                optimizer.zero_grad()
                predicted_logits = []
                for t in range(self.n_timesteps):
                    predicted_logits.append(net(data, t))

                # device of the model output
                out_device = predicted_logits[-1].device

                # move the targets to the output's device
                targets_dev = targets_cpu.to(out_device)

                # one-hot: index on CPU, then move to the device
                y = torch.eye(self.num_classes)[targets_cpu].to(out_device)

                loss = 0
                timestep_losses = torch.zeros(self.n_timesteps)
                timestep_accs   = torch.zeros(self.n_timesteps)

                for t in range(len(predicted_logits)):
                    sum_term   = torch.zeros_like(predicted_logits[t])
                    t_timesteps = list(range(t + 1, self.n_timesteps))
                    for i, n in enumerate(t_timesteps, 1):
                        logit_k  = predicted_logits[n].detach().clone()
                        softmax_i = F_.softmax(logit_k, dim=1)
                        sum_term  = sum_term + self.flags.lambda_val ** (i - 1) * softmax_i
                        del logit_k

                    term_1     = (1 - self.flags.lambda_val) * sum_term
                    term_2     = self.flags.lambda_val ** (self.n_timesteps - t - 1) * y
                    softmax_j  = term_1 + term_2

                    loss_i = criterion(
                        pred_logits=predicted_logits[t],
                        y_true_softmax=softmax_j,
                    )

                    if self.flags.tau_weighted_loss and t < self.n_timesteps - 1:
                        tau_i  = self.tau_handler(t - 1, epoch_i)
                        loss_i = tau_i * loss_i

                    if self.flags.tdl_mode == "EWS":
                        loss = loss + loss_i
                    else:
                        if t > 0:
                            loss = loss + loss_i

                    timestep_losses[t] = loss_i.item()

                    softmax_i = F_.softmax(predicted_logits[t], dim=1)
                    y_pred    = torch.argmax(softmax_i, dim=1)
                    n_correct = torch.eq(targets_dev, y_pred).sum()
                    timestep_accs[t] = n_correct / float(targets_dev.shape[0])

                if self.flags.normalize_loss:
                    loss = loss / float(self.n_timesteps)

                loss.backward()
                optimizer.step()

                batch_losses.append(timestep_losses)
                batch_accs.append(timestep_accs)

                mean_loss = torch.stack(batch_losses).mean().item()
                mean_acc  = torch.stack(batch_accs).mean().item() * 100
                sys.stdout.write(
                    f"\rTraining Batch {batch_i+1}/{len(loader)} — "
                    f"Loss: {mean_loss:.6f}  Acc: {mean_acc:.2f}%"
                )
                sys.stdout.flush()

            batch_losses = torch.stack(batch_losses).detach().numpy()
            batch_accs   = torch.stack(batch_accs).detach().numpy()
            return batch_losses, batch_accs

    # replacement in the module class (only modification to the original code)
    train_handler.CascadedTrainingScheme = _FixedCascadedTrainingScheme
    # also update get_train_loop which uses CascadedTrainingScheme
    original_get_train_loop = train_handler.get_train_loop
    def _fixed_get_train_loop(n_timesteps, num_classes, flags, tau_handler=None):
        if flags.train_mode == "baseline":
            return train_handler.SequentialTrainingScheme(num_classes, flags)
        elif flags.train_mode == "cascaded":
            return _FixedCascadedTrainingScheme(n_timesteps, num_classes, flags, tau_handler)
    train_handler.get_train_loop = _fixed_get_train_loop


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAT benchmark analysis — produces the same figures/metrics as NRM
#    but using CNet predictions instead of NRM forward passes.
#
#    Mapping exit ← beep (from split_exit_accuracies.py):
#      beep 500ms  → exit 0  (timestep 0)
#      beep 900ms  → exit 2  (timestep 2)
#      beep 1100ms → exit 4  (timestep 4)
#      beep 1300ms → exit 6  (timestep 6)
#      beep 1500ms → exit 8  (timestep 8)
# ─────────────────────────────────────────────────────────────────────────────

_BEEP_MS_TO_EXIT = {500: 0, 900: 2, 1100: 4, 1300: 6, 1500: 8}
_BEEP_TIMES_S    = sorted(ms / 1000.0 for ms in _BEEP_MS_TO_EXIT)
_N_CHOICES       = 16
_CHANCE          = 1.0 / _N_CHOICES


def _build_human_data_dict(test_ds) -> dict:
    """
    Extract from the test-set DataFrame the 'data' dict
    in the format expected by scripts/eval_sat.py.
    """
    from datasets.sat.loader import CATEGORIES as _CATS

    df        = test_ds.df.reset_index(drop=True)
    cat_to_idx = {cat: i for i, cat in enumerate(_CATS)}

    def _resp_to_idx(r):
        if isinstance(r, str):
            return cat_to_idx.get(r.strip().lower(), -1)
        return -1

    return {
        'levels':        df['level_str'].tolist(),
        'image_indices': df['image_index'].values.astype(int),
        'human_choices': np.array([_resp_to_idx(r) for r in df['response']], dtype=int),
        'true_labels':   df['choice'].values.astype(int),    # TRUE label (from the filename)
        'true_rts':      df['duration'].values.astype(float) / 1000.0,  # ms → s
        'time_limits':   df['time_limit'].values.astype(float) / 1000.0, # ms → s
        'srids':         df['srid'].tolist(),
    }


def _make_per_beep_entry(beep_s: float, per_img_acc: np.ndarray,
                          per_img_rt: np.ndarray | None = None) -> dict:
    """Helper: build the per_beep entry in the curves_dict format."""
    n    = len(per_img_acc)
    rt   = per_img_rt if per_img_rt is not None else np.full(n, beep_s)
    return {
        'beep_s':        beep_s,
        'rt_mean':       float(rt.mean()),
        'rt_std':        float(rt.std())          if n > 1 else 0.0,
        'rt_sem':        float(rt.std() / n**0.5) if n > 1 else 0.0,
        'acc_mean':      float(per_img_acc.mean()),
        'acc_std':       float(per_img_acc.std())          if n > 1 else 0.0,
        'acc_sem':       float(per_img_acc.std() / n**0.5) if n > 1 else 0.0,
        'n_images':      n,
        'n_trials':      n,
        'per_image_rt':  rt.tolist(),
        'per_image_acc': per_img_acc.tolist(),
    }


def _build_cnet_model_curves(test_ds, output_reps: dict) -> dict:
    """
    Build model_curves in the format scripts/eval_sat.py expects,
    using the CNet predictions already computed in output_reps.

    Same structure as the output of calculate_model_curves_by_level:
        {scenario, ordered_levels, display_labels, colors,
         overall: {lv: {per_beep, beeps_s, n_beeps}},
         by_category: {lv: {cat: {per_beep, beeps_s, n_beeps}}}}
    """
    from scripts.eval_sat import infer_scenario, _make_curves_dict

    df            = test_ds.df.reset_index(drop=True)
    N             = len(df)
    correct       = output_reps['correct']              # (T, N) float tensor, 0 o 1

    time_limits_ms = df['time_limit'].values.astype(int)
    true_labels    = df['choice'].values.astype(int)
    levels_list    = df['level_str'].tolist()
    image_indices  = df['image_index'].values.astype(int)
    sat_mask       = time_limits_ms > 0

    scenario, ordered_levels, display, colors = infer_scenario(levels_list)

    assert correct.shape[1] == N, (
        f"output_reps.correct has {correct.shape[1]} columns, test_ds has {N} trials"
    )

    # raw[lv][cat][beep_s][img_id] = list of correct values (0/1)
    from collections import defaultdict as _dd
    raw = {
        lv: {cat: _dd(lambda: _dd(list)) for cat in range(_N_CHOICES)}
        for lv in ordered_levels
    }

    for i in range(N):
        if not sat_mask[i]:
            continue
        lv    = str(levels_list[i])
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
                cat_beep[beep_s]      = _make_per_beep_entry(beep_s, per_img)
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
        'scenario':       scenario,
        'ordered_levels': ordered_levels,
        'display_labels': display,
        'colors':         colors,
        'overall':        overall_by_level,
        'by_category':    by_cat_by_level,
    }


def _build_pooled_curves(test_ds, output_reps: dict) -> tuple:
    """
    Human and CNet curves pooled over every level (no grouping by level).
    Used for sat_curves_beep_vs_acc.png, like SATPlotter in metrics/plots_sat.py.

    Returns (human_by_cat, human_overall, model_by_cat, model_overall).
    """
    from scripts.eval_sat import _make_curves_dict
    from collections import defaultdict as _dd

    df            = test_ds.df.reset_index(drop=True)
    N             = len(df)
    correct       = output_reps['correct']   # (T, N)
    predictions   = output_reps['predictions']  # (T, N)
    targets_t     = output_reps['target']       # (N,)

    time_limits_ms = df['time_limit'].values.astype(int)
    true_labels    = df['choice'].values.astype(int)
    image_indices  = df['image_index'].values.astype(int)
    sat_mask       = time_limits_ms > 0

    # Human responses
    from datasets.sat.loader import CATEGORIES as _CATS
    cat_to_idx = {c: i for i, c in enumerate(_CATS)}
    human_choices = np.array(
        [cat_to_idx.get(str(r).strip().lower(), -1) if isinstance(r, str) else -1
         for r in df['response']],
        dtype=int,
    )
    true_rts_ms = df['duration'].values.astype(float)

    # ── Human: nested[cat][beep_s][img_id] = {rts, correct} ─────────────
    h_raw: dict = {cat: _dd(lambda: _dd(lambda: {'rts': [], 'correct': []}))
                   for cat in range(_N_CHOICES)}
    # ── CNet: nested[cat][beep_s][img_id] = list[correct] ────────────────
    c_raw: dict = {cat: _dd(lambda: _dd(list)) for cat in range(_N_CHOICES)}

    for i in range(N):
        if not sat_mask[i]:
            continue
        tl_ms  = int(time_limits_ms[i])
        if tl_ms not in _BEEP_MS_TO_EXIT:
            continue
        beep_s = tl_ms / 1000.0
        img_id = int(image_indices[i])
        cat    = int(true_labels[i])
        h_ch   = int(human_choices[i])
        rt_s   = float(true_rts_ms[i]) / 1000.0

        # Human
        h_raw[cat][beep_s][img_id]['rts'].append(rt_s)
        h_raw[cat][beep_s][img_id]['correct'].append(int(h_ch == cat))

        # CNet
        exit_t = _BEEP_MS_TO_EXIT[tl_ms]
        c_raw[cat][beep_s][img_id].append(float(correct[exit_t, i].item()))

    def _to_curves(raw_dict, use_rts=False):
        """raw_dict[cat][beep][img] -> {acc, rt}; rts are used only if use_rts=True."""
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
                    per_img_acc = np.array([np.mean(v['correct']) for v in img_dict.values()])
                    per_img_rt  = np.array([np.mean(v['rts'])     for v in img_dict.values()])
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
            rt_arr = (np.array(all_rt[beep_s]) if all_rt.get(beep_s) else None)
            overall_beep[beep_s] = _make_per_beep_entry(
                beep_s, np.array(acc_list), rt_arr
            )
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
        perturb: str = '',
) -> None:
    """
    17-subplot figure (overall + 16 categories), like
    SATPlotter.plot_sat_by_category() in metrics/plots_sat.py.
    X-axis = nominal beep (s).
    """
    from datasets.sat.loader import CATEGORIES as _CATS

    n_choices = _N_CHOICES
    err_key   = 'sem'
    n_cols    = 4
    n_panels  = n_choices + 1
    n_rows    = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.0, n_rows * 3.2),
                             squeeze=False)
    axes_flat = axes.flatten()

    def _plot_one(ax, h_data, m_data, title):
        chance_pct = _CHANCE * 100.0
        for data, color, lbl, ls, mk in [
            (h_data, '#4472C4', 'Human',  '-',  'o'),
            (m_data, '#ED7D31', 'CNet',   '--', 's'),
        ]:
            if data is None:
                continue
            beeps = data['beeps_s']
            pd_   = data['per_beep']
            xs    = np.array([b for b in beeps])
            acc   = np.array([pd_[b]['acc_mean'] for b in beeps]) * 100.0
            yerr  = np.array([pd_[b][f'acc_{err_key}'] for b in beeps]) * 100.0
            xs_f   = np.concatenate([[0.0], xs])
            acc_f  = np.concatenate([[chance_pct], acc])
            yerr_f = np.concatenate([[0.0], yerr])
            ax.errorbar(xs_f, acc_f, yerr=yerr_f, fmt=f'{mk}{ls}',
                        color=color, label=lbl,
                        capsize=2, linewidth=1.6, markersize=4)
        ax.axhline(chance_pct, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax.set_ylim([0, 105])
        ax.set_title(title, fontsize=8, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    _plot_one(axes_flat[0], human_overall, model_overall,
              f'Overall [{perturb}]')
    axes_flat[0].legend(fontsize=7, loc='lower right')

    for cat_idx in range(n_choices):
        ax  = axes_flat[cat_idx + 1]
        nm  = _CATS[cat_idx] if cat_idx < len(_CATS) else str(cat_idx)
        _plot_one(ax, human_by_cat.get(cat_idx), model_by_cat.get(cat_idx), nm)

    for i in range(n_choices + 1, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.supxlabel('Beep time (s)', fontsize=9, y=0.01)
    fig.supylabel('Accuracy (%)',  fontsize=9, x=0.01)
    fig.suptitle(f'SAT Curves — CNet vs Human  [{perturb}]',
                 fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [CNet] Saved {display_path(save_path)}")


def run_cnet_sat_analysis(
        test_ds,
        output_reps: dict,
        perturb: str,
        save_dir: str | os.PathLike,
        ranking_key: str = "cnet_par",
        refresh: bool = False,
) -> dict:
    """
    SAT benchmark pipeline for CNet — produces the same figures and metrics
    as scripts/eval_sat.py for NRM, but based on already-computed output_reps.

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
    import sys as _sys
    _sys.path.insert(0, REPO_ROOT)

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
    print("SAT BENCHMARK ANALYSIS  (CNet — compatible with NRM)")
    print("=" * 70)

    # ── 1. Human data dict ───────────────────────────────────────────────
    print("  Extracting the human data from the test set...")
    data = _build_human_data_dict(test_ds)

    # ── 2. Human curves grouped by level ─────────────────────────────────
    print("  Computing human curves per level...")
    human_curves = calculate_human_curves_by_level(data, n_choices=_N_CHOICES)
    scenario     = human_curves['scenario']
    ordered_lvls = human_curves['ordered_levels']
    print(f"  Scenario: {scenario}  |  Levels: {ordered_lvls}")

    if not ordered_lvls:
        print("  [WARN] No level found - SAT analysis skipped.")
        return {}

    # ── 3. CNet curves per level ─────────────────────────────────────────
    print("  Building CNet curves per level (from output_reps)...")
    model_curves = _build_cnet_model_curves(test_ds, output_reps)
    model_curves['scenario'] = scenario

    # ── 4. Per-observer curves (for the eRMSE / Spearman boxplots) ─────────
    print("  Computing per-observer curves (srid)...")
    per_obs_curves = _compute_per_observer_curves(data, ordered_lvls,
                                                   n_choices=_N_CHOICES)
    n_obs = len(per_obs_curves) if per_obs_curves else 0
    print(f"  {n_obs} observers in the test set.")

    # ── 5. Pooled curves (all levels) for sat_curves_beep_vs_acc ──────
    print("  Building pooled curves (human + CNet)...")
    h_by_cat, h_overall, m_by_cat, m_overall = _build_pooled_curves(
        test_ds, output_reps
    )

    # ── 6. JSON curve ────────────────────────────────────────────────────
    json_path = save_dir / 'sat_level_curves.json'
    save_curves_to_json(human_curves, model_curves, json_path)

    # ── 7. sat_curves_beep_vs_acc.png (pooled, 17 subplot) ───────────────
    if WRITE_DIAGNOSTICS:
        _plot_sat_curves_beep_vs_acc(
            h_by_cat, m_by_cat, h_overall, m_overall,
            save_path=save_dir / 'sat_curves_beep_vs_acc.png',
            perturb=perturb,
        )

    # ── 8. sat_level_beep_vs_acc.png ─────────────────────────────────────
    if WRITE_DIAGNOSTICS:
        plot_sat_by_level(
            human_curves, model_curves,
            x_mode='beep', use_sem=True,
            save_path=save_dir / 'sat_level_beep_vs_acc.png',
        )

    # ── 9. sat_level_by_cat_beep_vs_acc.png ──────────────────────────────
    if WRITE_DIAGNOSTICS:
        plot_sat_by_level_and_category(
            human_curves, model_curves,
            x_mode='beep', use_sem=True, n_choices=_N_CHOICES,
            save_path=save_dir / 'sat_level_by_cat_beep_vs_acc.png',
        )

    # ── 10. SATBench-equivalent metrics ───────────────────────────────────
    print("  Computing metrics (eRMSE, Spearman, steepness)...")
    metrics = compute_satbench_metrics(
        human_curves, model_curves, per_obs_curves, n_choices=_N_CHOICES
    )
    metrics['colors'] = human_curves['colors']

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
            prefix='satbench',
            n_choices=_N_CHOICES,
        )

    # ── 13. LaTeX table ───────────────────────────────────────────────────
    if WRITE_DIAGNOSTICS:
        latex_str  = generate_latex_table(metrics, scenario_name=f'{scenario} CNet')
        latex_path = save_dir / 'sat_level_metrics_table.tex'
        latex_path.write_text(latex_str)
        print(f"  [CNet] Saved {display_path(latex_path)}")

        # ── 13. CSV metrics table ─────────────────────────────────────────────
        csv_path = save_dir / 'sat_level_metrics_table.csv'
        disp     = metrics.get('display_labels', {})
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['level', 'display', 'rmse', 'spearman_r',
                             'steepness_human', 'steepness_model'])
            for lv in ordered_lvls:
                row = metrics['overall'].get(lv, {})
                writer.writerow([
                    lv,
                    disp.get(lv, lv),
                    row.get('rmse',            float('nan')),
                    row.get('spearman_r',      float('nan')),
                    row.get('steepness_human', float('nan')),
                    row.get('steepness_model', float('nan')),
                ])
            avg = metrics.get('mean_over_levels', {})
            writer.writerow([
                'Average', 'Average',
                avg.get('rmse',            float('nan')),
                avg.get('spearman_r',      float('nan')),
                avg.get('steepness_human', float('nan')),
                avg.get('steepness_model', float('nan')),
            ])
        print(f"  [CNet] Saved {display_path(csv_path)}")

        # ── 14. Print accuracy per beep/level ────────────────────────────────
        print("\n  CNet accuracy per level/deadline:")
        for lv in ordered_lvls:
            lbl = human_curves['display_labels'].get(lv, lv)
            ov  = model_curves['overall'].get(lv)
            if ov is None:
                continue
            beep_str = '  '.join(
                f"{b:.1f}s→{ov['per_beep'][b]['acc_mean']*100:.1f}%"
                for b in ov['beeps_s']
            )
            print(f"    [{lbl}]  {beep_str}")

        print("\nSAT benchmark CNet completed.")
    return {
        'metrics':    metrics,
        'save_dir':   str(save_dir),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main script — identical to train.py + eval.py + Analysis.py
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CNet on the SAT data (NRM pipeline) - train + eval + analysis"
    )
    p.add_argument("--perturb", required=True,
                   choices=["color-gray", "color-blur", "gray-noise"])
    p.add_argument("--diagnostics", action="store_true",
                   help="Also write the per-model PNGs and the per-model "
                        ".tex/.csv. Off by default: pipelines/sat.ipynb "
                        "only reads sat_level_curves.json and "
                        "satbench_metrics_full.json.")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training and load the saved weights.")
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
    # hyperparameters — as in the paper (split_exit_accuracies.py: lr_0.01, wd_0.0005)
    p.add_argument("--n_epochs",      type=int,   default=120)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--weight_decay",  type=float, default=5e-4)
    p.add_argument("--lr_milestones", nargs="+",  type=float, default=[30, 60, 90])
    p.add_argument("--lr_schedule_gamma", type=float, default=0.2)
    p.add_argument("--lambda_val",    type=float, default=1.0)
    p.add_argument("--momentum",      type=float, default=0.9)
    p.add_argument("--nesterov",      action="store_true", default=True)
    p.add_argument("--normalize_loss",action="store_true", default=False)
    p.add_argument("--eval_freq",     type=int,   default=10)
    p.add_argument("--save_freq",     type=int,   default=10)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--device",        type=int,   default=0)
    p.add_argument("--random_seed",   type=int,   default=42)
    p.add_argument("--debug",         action="store_true", default=False)
    # for the CNet modules
    p.add_argument("--model_key",         default="resnet18")
    p.add_argument("--train_mode",        default="cascaded")
    p.add_argument("--cascaded_scheme",   default="parallel")
    p.add_argument("--tdl_mode",          default="OSD")
    p.add_argument("--tdl_alpha",         type=float, default=0.0)
    p.add_argument("--noise_var",         type=float, default=0.0)
    p.add_argument("--bn_time_affine",    action="store_true", default=False)
    p.add_argument("--bn_time_stats",     action="store_true", default=False)
    p.add_argument("--tau_weighted_loss", action="store_true", default=False)
    p.add_argument("--init_tau",          type=float, default=0.01)
    p.add_argument("--target_IC_inference_costs", nargs="+", type=float,
                   default=[0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90])
    p.add_argument("--keep_logits",    action="store_true", default=True)
    p.add_argument("--keep_embeddings",action="store_true", default=False)
    p.add_argument("--drop_last",      action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    global WRITE_DIAGNOSTICS
    WRITE_DIAGNOSTICS = bool(getattr(args, 'diagnostics', False))

    # enable the device fix in the training handler (BEFORE importing anything from cnet)
    _patch_cascaded_training()

    # train.py / eval.py pull in CNet's vendored dataset handlers, which import
    # each other as `datasets.*` — the name this repo's own loader package also
    # claims. See _vendored_datasets().
    with _vendored_datasets():
        from models   import resnet as cnet_resnet
        from modules  import eval_handler, losses, train_handler, utils
        from models   import sdn_utils
        from train    import setup_model, condition_model
        from eval     import compute_output_representations

    # ── device ───────────────────────────────────────────────────────────────
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device : {device}")
    print(f"Perturb: {args.perturb}")

    utils.make_reproducible(args.random_seed)

    exp_name  = f"cnet_resnet18_SAT_{args.perturb}"
    # td(1.0) = parallel cascade, td(0.0) = serial; they are ranked separately.
    ranking_key = "cnet_par" if float(args.lambda_val) >= 0.5 else "cnet_ser"
    weight_stem = "cnet_parallel" if float(args.lambda_val) >= 0.5 else "cnet_serial"

    # ── output and scratch directories ────────────────────────────────────────
    # A training run, --refresh or --refresh_weights writes every output with
    # the _REFRESH suffix, so the reference outputs are never overwritten.
    sys.path.insert(0, REPO_ROOT)
    from scripts.results_io import sat_model_dir
    refresh = args.refresh or args.refresh_weights or not args.eval_only
    sat_save_dir = str(sat_model_dir(args.perturb, ranking_key, refresh=refresh))
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

    # ── extra flags CNet requires ───────────────────────────────────────��──
    args.cascaded       = True
    args.multiple_fcs   = False
    args.grayscale      = False
    args.gauss_noise    = False
    args.gauss_noise_std= 0.0
    args.blur           = False
    args.blur_std       = 0.0
    # "ImageNet2012" must be in the name because setup_model() uses
    # str.find(args.dataset_name, "ImageNet2012")>-1 to pick the 7×7 head
    args.dataset_name   = "ImageNet2012_SAT"
    args.imagenet_pretrained = False
    args.n_channels     = 3
    args.use_pretrained_weights = False

    # ── model (setup_model() from train.py) ──────────────────────────────────
    # setup_model uses only data_handler.num_classes and args
    data_handler = _MockDataHandler()

    # CUDA_VISIBLE_DEVICES trick to avoid the multi-GPU DataParallel + TDL bug
    old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    try:
        net = setup_model(data_handler, device, args, save_root=exp_root)
    finally:
        if old_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd

    n_timesteps = net.module.timesteps
    print(f"Timesteps CNet: {n_timesteps}")

    # ── training ─────────────────────────────────────────────────────────────
    if not args.eval_only:
        train_loader = _build_loader(train_ds, train=True,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers)
        val_loader   = _build_loader(val_ds, train=False,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers)

        # condition_model() from train.py sets up the optimiser/scheduler
        opts = condition_model(exp_root, args)

        tau_handler = sdn_utils.IC_tau_handler(
            init_tau=args.init_tau,
            tau_targets=args.target_IC_inference_costs,
            epoch_asymptote=opts["tau_epoch_asymptote"],
            active=opts["tau_scheduling_active"],
        )
        optimizer = opts["optimizer_init_op"](net.parameters(),
                                              **opts["optimizer_dict"])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=opts["lr_schedule_milestones"],
            gamma=opts["lr_schedule_gamma"],
        )

        criterion = losses.categorical_cross_entropy
        train_fxn = train_handler.get_train_loop(
            n_timesteps, data_handler.num_classes, args, tau_handler,
        )
        eval_fxn = eval_handler.get_eval_loop(
            n_timesteps, data_handler.num_classes,
            cascaded=args.cascaded, flags=args,
            keep_logits=False, keep_embeddings=False,
            tau_handler=tau_handler,
        )

        # save args.json (as train.py does)
        utils.save_args(args, exp_root)

        metrics = {
            "train": collections.defaultdict(list),
            "val":   collections.defaultdict(list),
            "test":  collections.defaultdict(float),
        }

        print(f"\n{'='*70}\nTRAINING  ({args.n_epochs} epochs)\n{'='*70}")
        for epoch_i in range(args.n_epochs):
            print(f"\nEpoch {epoch_i+1}/{args.n_epochs}")
            train_loss, train_acc = train_fxn(
                net, train_loader, criterion, epoch_i, optimizer, device,
            )
            lr_scheduler.step()

            train_loss_val = np.mean(train_loss, axis=0)[-1]
            train_acc_val  = np.mean(train_acc,  axis=0)[-1] * 100
            print(f"\n  Train loss: {train_loss_val:.6f}  acc: {train_acc_val:.2f}%")

            metrics["train"]["loss"].append((epoch_i, train_loss))
            metrics["train"]["acc"].append((epoch_i, train_acc))

            if epoch_i % args.eval_freq == 0 or epoch_i == args.n_epochs - 1:
                val_loss, val_acc, _ = eval_fxn(
                    net, val_loader, criterion, epoch_i, device,
                )
                print(f"  Val   acc: {np.mean(val_acc)*100:.2f}%")
                metrics["val"]["loss"].append((epoch_i, val_loss))
                metrics["val"]["acc"].append((epoch_i, val_acc))

            if epoch_i % args.save_freq == 0 or epoch_i == args.n_epochs - 1:
                utils.save_model(net, optimizer, exp_root, epoch_i, args.debug)

            utils.save_metrics(metrics, exp_root, args.debug)

        print("\nTraining completed.")
        keep_final_checkpoint(ckpt_dir, weight_stem, args.perturb)

    # ── eval (eval.py) ────────────────────────────────────────────────────────
    # The checkpoint a training run just saved (or, with --refresh_weights, a
    # previous one), else the released weights in competitors/weights/sat/ —
    # td(1.0) is the parallel cascade,
    # td(0.0) the serial one, and they ship as separate files. Either layout is
    # `{"model": state_dict, ...}`, which is what models/model_utils.load_model
    # reads out of `pretrained_path` below.
    selected_ckpt = resolve_sat_checkpoint(
        model=weight_stem,
        perturb=args.perturb,
        explicit=args.weights,
        refresh_weights=args.refresh_weights or not args.eval_only,
    )

    model_config = torch.load(os.path.join(exp_root, "model_config.pt"),
                              weights_only=False)
    model_config["pretrained"]      = True
    model_config["pretrained_path"] = selected_ckpt

    old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    try:
        net_eval = cnet_resnet.__dict__[args.model_key](**model_config).to(device)
    finally:
        if old_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd

    n_timesteps = net_eval.module.timesteps

    test_loader = _build_loader(test_ds, train=False,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    tau_handler_eval = sdn_utils.IC_tau_handler(
        init_tau=args.init_tau,
        tau_targets=args.target_IC_inference_costs,
        epoch_asymptote=100,
        active=False,
    )
    eval_fxn_full = eval_handler.get_eval_loop(
        n_timesteps, data_handler.num_classes,
        cascaded=True, flags=args,
        keep_logits=True, keep_embeddings=False,
        tau_handler=tau_handler_eval,
    )
    criterion = losses.categorical_cross_entropy

    print(f"\n{'='*70}\nEVALUATION  (test set)\n{'='*70}")
    test_loss, test_acc, logged_data = eval_fxn_full(
        net_eval, test_loader, criterion, 0, device,
    )
    final_acc = test_acc[-1] * 100
    print(f"\nTest accuracy (exit finale): {final_acc:.2f}%")

    # output_representations in the format expected by Analysis.py
    logits = logged_data["logits"]             # (9, N, 16)
    y      = logged_data["y"].to(logits.device)
    output_reps = compute_output_representations(logits, y)

    out_rep_path = os.path.join(out_dir, "output_representations__test__OSD.pt")
    torch.save(output_reps, out_rep_path)
    print(f"Saved: {display_path(out_rep_path)}")

    # ── Analysis (SAT curve) — logic of Analysis.py ──────────────────���───────
    print(f"\n{'='*70}\nSAT CURVE  (Analysis)\n{'='*70}")

    correct = output_reps["correct"]           # (9, N)
    accs    = correct.float().mean(dim=1)      # (9,) — accuracy per timestep

    print("Accuracy per timestep:")
    for t, a in enumerate(accs):
        print(f"  t={t}: {a.item()*100:.2f}%")

    # plot SAT curve (identical to Analysis.py)
    t_steps = list(range(n_timesteps))
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t_steps, accs.numpy(), marker="o", linewidth=2,
             label=f"CNet parallel λ={args.lambda_val}")
    plt.axhline(1/16 * 100, ls="--", color="grey", linewidth=1, label="chance (6.25%)")
    plt.xlabel("Timestep")
    plt.ylabel("Accuracy (%)")
    plt.title(f"SAT Curve — CNet ResNet-18 [{args.perturb}]")
    plt.xticks(t_steps)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(figs_dir, f"SAT_curve_{args.perturb}.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved: {display_path(curve_path)}")

    # plot per category (as in Analysis.py accuracy_by_label)
    fig = plt.figure(figsize=(12, 6))
    target_tensor = output_reps["target"]
    for t_lbl in target_tensor.unique():
        t_idx     = (target_tensor == t_lbl).nonzero(as_tuple=True)[0]
        t_correct = torch.index_select(correct, 1, t_idx)
        t_accs    = t_correct.float().mean(dim=1)
        plt.plot(t_steps, t_accs.numpy(), label=f"label {t_lbl.item()}", alpha=0.7)
    plt.xlabel("Timestep")
    plt.ylabel("Accuracy (%)")
    plt.title(f"SAT curves per category - CNet [{args.perturb}]")
    plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=7)
    plt.tight_layout()
    cat_path = os.path.join(figs_dir, f"SAT_curve_by_label_{args.perturb}.png")
    plt.savefig(cat_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {display_path(cat_path)}")

    # ── SAT benchmark analysis ────────────────────────────────────────────────
    run_cnet_sat_analysis(
        test_ds     = test_ds,
        output_reps = output_reps,
        perturb     = args.perturb,
        save_dir    = sat_save_dir,
        ranking_key = ranking_key,
        refresh     = refresh,
    )

    shutil.rmtree(exp_root, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
