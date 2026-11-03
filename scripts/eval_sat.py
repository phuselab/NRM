"""
scripts/eval_sat.py
===================
SAT-level analysis split by perturbation level (blur / noise / color-gray).
Metrics and figures for the SAT benchmark (Section 5 of the paper).

Metrics
-------
  eRMSE          : curve-fit error averaged over levels, distribution over observers
  Spearman rho   : category-wise correlation (flattened) per observer
  Steepness kappa : mean curvature of Weibull fit (Appendix D.2)

Figures
-------
  Fig 3  : Accuracy vs Time/Timestep per level, with chance point at t=0
  Fig 4  : eRMSE boxplot (distribution over observers)
  Fig 5  : Spearman boxplot (distribution over observers)
  Fig 6  : Category barplot (median correlation per category)
  Fig 7  : Steepness line plot (log-scale) vs perturbation level

Supported datasets
------------------
  color-blur.txt  : levels '0','3','9'       (blur sigma 0, 3, 9)
  gray-noise.txt  : levels '0','0.04','0.16' (noise sigma 0, 0.04, 0.16)
  color-gray.txt  : levels 'c','g'           (Color, Gray)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metrics.point_metrics import HumanAgreementAnalyzer, MetricsCalculator
from metrics.relational_metrics import (
    SATCurvesCalculator,
    _compute_per_observer_curves,
    compute_satbench_metrics,
)
from metrics.rt_distribution import RTDistributionMetrics
from metrics.rt_likelihood import (
    RTLogLikelihoodCalculator,
    calculate_hh_and_model_ll_beep_pooled,
    calculate_hh_model_ll_beep_cross_image,
    save_pooled_beep_ll_csv,
)
from metrics.plots_sat import (
    Plotter,
    SATPlotter,
    create_distribution_comparison_plots,
    create_ll_comparison_plot,
    create_ll_comparison_plots_beep_cross_image,
    create_ll_comparison_plots_beep_pooled_separate,
    create_ll_comparison_plots_per_beep_separate,
    plot_sat_by_level,
    plot_sat_by_level_and_category,
    plot_satbench_figures,
)
from scripts.results_io import display_path

# ---------------------------------------------------------------------------
# Display labels and colours for each perturbation level
# ---------------------------------------------------------------------------

CATEGORIES = [
    "airplane", "bear", "bicycle", "bird", "boat", "bottle",
    "car", "cat", "chair", "clock", "dog", "elephant",
    "keyboard", "knife", "oven", "truck",
]

# Mapping level_str → (display label, suggested colour)
_LEVEL_META: dict[str, tuple[str, str]] = {
    # color-blur
    '0':    ('σ=0',    '#006400'),   # dark green
    '3':    ('σ=3',    '#3CB371'),   # medium green
    '9':    ('σ=9',    '#90EE90'),   # light green
    # gray-noise
    '0.04': ('σ=0.04', '#1E90FF'),   # dodger blue
    '0.16': ('σ=0.16', '#87CEEB'),   # sky blue
    # color-gray
    'c':    ('Color',  '#8B0000'),   # dark red
    'g':    ('Gray',   '#DA70D6'),   # orchid
}

# The level set of each scenario
_BLUR_LEVELS  = {'0', '3', '9'}
_NOISE_LEVELS = {'0', '0.04', '0.16'}
_CG_LEVELS    = {'c', 'g'}

# Chance level per 16-class classification
_CHANCE = 1.0 / 16.0   # 0.0625 = 6.25 %


# ---------------------------------------------------------------------------
# Helper: infer the scenario from the levels present in the data
# ---------------------------------------------------------------------------

def infer_scenario(levels: list[str]) -> tuple[str, list[str], dict[str, str], dict[str, str]]:
    """
    Returns (scenario_name, ordered_levels, display_labels, colors).

    scenario_name: 'color-blur' | 'gray-noise' | 'color-gray' | 'unknown'
    ordered_levels: levels in order (from "least degraded" to "most degraded")
    display_labels: level_str → label human-readable
    colors: level_str → matplotlib colour
    """
    unique = set(levels) - {'unknown', 'nan', 'None', ''}

    if unique <= _CG_LEVELS or unique == _CG_LEVELS:
        scenario = 'color-gray'
        ordered  = [l for l in ['c', 'g'] if l in unique]
    elif unique <= _BLUR_LEVELS:
        scenario = 'color-blur'
        ordered  = [l for l in ['0', '3', '9'] if l in unique]
    elif unique <= _NOISE_LEVELS:
        scenario = 'gray-noise'
        ordered  = [l for l in ['0', '0.04', '0.16'] if l in unique]
    else:
        scenario = 'unknown'
        ordered  = sorted(unique)

    display = {l: _LEVEL_META.get(l, (l, 'gray'))[0] for l in ordered}
    colors  = {l: _LEVEL_META.get(l, (l, 'gray'))[1] for l in ordered}
    return scenario, ordered, display, colors


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def _aggregate_over_images(image_indices, choices, true_labels, rts, time_limits):
    """
    Group trials by (deadline, image_index) and return per-deadline stats.
    """
    nested = defaultdict(lambda: defaultdict(lambda: {'rts': [], 'correct': []}))
    for i in range(len(image_indices)):
        b   = float(time_limits[i])
        img = int(image_indices[i])
        nested[b][img]['rts'].append(float(rts[i]))
        nested[b][img]['correct'].append(int(int(choices[i]) == int(true_labels[i])))

    results = {}
    for beep_s in sorted(nested.keys()):
        per_image_rt  = []
        per_image_acc = []
        n_trials_total = 0
        for img_id, vals in nested[beep_s].items():
            per_image_rt.append(np.mean(vals['rts']))
            per_image_acc.append(np.mean(vals['correct']))
            n_trials_total += len(vals['rts'])
        per_image_rt  = np.array(per_image_rt,  dtype=float)
        per_image_acc = np.array(per_image_acc, dtype=float)
        n = len(per_image_rt)
        results[beep_s] = {
            'beep_s':        beep_s,
            'rt_mean':       float(per_image_rt.mean()),
            'rt_std':        float(per_image_rt.std())          if n > 1 else 0.0,
            'rt_sem':        float(per_image_rt.std() / n**0.5) if n > 1 else 0.0,
            'acc_mean':      float(per_image_acc.mean()),
            'acc_std':       float(per_image_acc.std())          if n > 1 else 0.0,
            'acc_sem':       float(per_image_acc.std() / n**0.5) if n > 1 else 0.0,
            'n_images':      n,
            'n_trials':      n_trials_total,
            'per_image_rt':  per_image_rt.tolist(),
            'per_image_acc': per_image_acc.tolist(),
        }
    return results


def _make_curves_dict(per_beep: dict) -> dict:
    """Wrap the per_beep dict in the standard {per_beep, beeps_s, n_beeps} form."""
    beeps_s = sorted(per_beep.keys())
    return {'per_beep': per_beep, 'beeps_s': beeps_s, 'n_beeps': len(beeps_s)}


# ---------------------------------------------------------------------------
# Human SAT curves per perturbation level
# ---------------------------------------------------------------------------

def build_sat_data_dict(test_ds, include_embeddings: bool = True) -> dict:
    """Flatten a SAT dataset split into the ``data`` dict this module consumes.

    ``calculate_human_curves_by_level`` and ``calculate_model_curves_by_level``
    both take one flat dict rather than a DataLoader. Building it in one place
    keeps NRM and every competitor on the same extraction, and keeps the key
    names and unit conversions (ms -> s) from drifting apart.

    Parameters
    ----------
    test_ds
        A split returned by ``SATDataset.train_val_test_split``.
    include_embeddings
        NRM needs ``embeddings`` to run its forward pass. Competitors supply
        their own pre-computed outputs, so they can pass False.

    Returns
    -------
    dict with keys ``levels``, ``image_indices``, ``human_choices``,
    ``true_labels``, ``true_rts``, ``time_limits``, ``srids`` and (optionally)
    ``embeddings``. RTs and time limits are in **seconds**.
    """
    from datasets.sat.loader import CATEGORIES as _CATEGORIES

    df = test_ds.df.reset_index(drop=True)
    cat_to_idx = {cat: i for i, cat in enumerate(_CATEGORIES)}

    def _resp_to_idx(r):
        # Unparseable or absent responses become -1 and are filtered downstream.
        if isinstance(r, str):
            return cat_to_idx.get(r.strip().lower(), -1)
        return -1

    # float32, not float64, for the two time fields, as ``sat_collate_fn``
    # produces them (it divides a ``torch.long`` by 1000, landing in torch's
    # default float32); this function reads the frame directly and would
    # otherwise keep float64. The difference is not cosmetic: ``unique_beeps``
    # is built from ``time_limits``, and a deadline of 900 ms reads as 0.9 in
    # float64 but 0.8999999761581421 in float32, which is the key the per-beep
    # curves are filed under. ``true_rts`` likewise feeds the human curves.
    data = {
        'levels':        df['level_str'].tolist(),
        'image_indices': df['image_index'].values.astype(int),
        'human_choices': np.array([_resp_to_idx(r) for r in df['response']], dtype=int),
        'true_labels':   df['choice'].values.astype(int),
        'true_rts':      (df['duration'].values.astype(np.float32) / 1000.0
                          ).astype(np.float32),
        'time_limits':   (df['time_limit'].values.astype(np.float32) / 1000.0
                          ).astype(np.float32),
        'srids':         df['srid'].tolist() if 'srid' in df.columns else [],
    }

    if include_embeddings:
        emb = getattr(test_ds, 'embeddings', None)
        if emb is None:
            raise ValueError(
                "test_ds has no embeddings. Run "
                "SATDataset.compute_dinov2_embeddings(save_path=...) first, or "
                "pass include_embeddings=False for a competitor that supplies "
                "its own model outputs."
            )
        if len(emb) != len(df):
            raise ValueError(
                f"{len(emb)} embeddings vs {len(df)} trials — the split and the "
                "embedding cache disagree; recompute the embeddings."
            )
        data['embeddings'] = emb.cpu().numpy() if hasattr(emb, 'cpu') else np.asarray(emb)

    return data


def calculate_human_curves_by_level(data: dict, n_choices: int = 16) -> dict:
    """
    Compute human SAT curves grouped by perturbation level.

    Returns
    -------
    {
      'scenario': str,
      'ordered_levels': list[str],
      'display_labels': dict[str, str],
      'colors': dict[str, str],
      'overall': { level → curves_dict },
      'by_category': { level → { cat_idx → curves_dict } },
    }
    """
    levels_list  = data['levels']
    levels_arr   = np.array(levels_list)
    img_idx      = data['image_indices']
    human_ch     = data['human_choices']
    true_lbl     = data['true_labels']
    true_rts     = data['true_rts']
    time_lims    = data['time_limits']

    sat_mask = time_lims > 0
    scenario, ordered_levels, display, colors = infer_scenario(levels_list)

    overall_by_level = {}
    by_cat_by_level  = {}

    for level in ordered_levels:
        lev_mask = (levels_arr == level) & sat_mask
        if lev_mask.sum() == 0:
            continue

        # Overall per level
        per_beep = _aggregate_over_images(
            img_idx[lev_mask], human_ch[lev_mask], true_lbl[lev_mask],
            true_rts[lev_mask], time_lims[lev_mask],
        )
        overall_by_level[level] = _make_curves_dict(per_beep)

        # Per category per level
        by_cat_by_level[level] = {}
        for cat in range(n_choices):
            cat_mask = lev_mask & (true_lbl == cat)
            if cat_mask.sum() == 0:
                continue
            per_beep_cat = _aggregate_over_images(
                img_idx[cat_mask], human_ch[cat_mask], true_lbl[cat_mask],
                true_rts[cat_mask], time_lims[cat_mask],
            )
            by_cat_by_level[level][cat] = _make_curves_dict(per_beep_cat)

    return {
        'scenario':       scenario,
        'ordered_levels': ordered_levels,
        'display_labels': display,
        'colors':         colors,
        'overall':        overall_by_level,
        'by_category':    by_cat_by_level,
    }


# ---------------------------------------------------------------------------
# Model SAT curves per perturbation level
# ---------------------------------------------------------------------------

def calculate_model_curves_by_level(
        model,
        device,
        data: dict,
        n_rt_samples: int = 50,
        n_choices: int = 16,
) -> dict:
    """
    ComputeRows: Human | NRM SAT curves grouped by perturbation level.
    """
    levels_list  = data['levels']
    levels_arr   = np.array(levels_list)
    img_idx_arr  = data['image_indices']
    true_lbl_arr = data['true_labels']
    time_lim_arr = data['time_limits']
    embeddings_t = torch.tensor(data['embeddings'], dtype=torch.float32)

    sat_mask = time_lim_arr > 0
    scenario, ordered_levels, display, colors = infer_scenario(levels_list)

    # Map image_index → embedding and level
    img_to_emb: dict[int, torch.Tensor] = {}
    img_to_level: dict[int, str] = {}
    for i in range(len(img_idx_arr)):
        idx = int(img_idx_arr[i])
        if idx not in img_to_emb:
            img_to_emb[idx]   = embeddings_t[i]
            img_to_level[idx] = str(levels_arr[i])

    img_to_cat: dict[int, int] = {}
    for i in range(len(img_idx_arr)):
        idx = int(img_idx_arr[i])
        if idx not in img_to_cat:
            img_to_cat[idx] = int(true_lbl_arr[i])

    unique_beeps = sorted(set(float(t) for t in time_lim_arr[sat_mask]))

    results_raw: dict[str, dict[int, dict[float, dict[int, dict]]]] = {
        lv: {c: {b: {} for b in unique_beeps} for c in range(n_choices)}
        for lv in ordered_levels
    }

    model.eval()
    with torch.no_grad():
        for level in ordered_levels:
            imgs_at_level = [idx for idx, lv in img_to_level.items() if lv == level]
            cat_ids       = [img_to_cat[idx] for idx in imgs_at_level]

            print(f"  [SAT level] {display.get(level, level)}: "
                  f"{len(imgs_at_level)} images x {len(unique_beeps)} beep")

            for beep_s in unique_beeps:
                emb_batch = torch.stack([img_to_emb[idx] for idx in imgs_at_level]).to(device)
                beep_t    = torch.full((len(imgs_at_level),), beep_s,
                                       dtype=torch.float32, device=device)

                n_img    = len(imgs_at_level)
                emb_rep  = emb_batch.repeat_interleave(n_rt_samples, dim=0)
                beep_rep = beep_t.repeat_interleave(n_rt_samples)

                cov = {
                    'img_embedding': emb_rep,
                    'time_limit':    beep_rep.unsqueeze(-1) if beep_rep.dim() == 1 else beep_rep,
                }

                _, _, pred_ch_all, pred_rt_all = model.simulate_nDDM(
                    batch_size=n_img * n_rt_samples,
                    covariates=cov,
                    soft=True,
                    n_reps=1,
                    t=0.993,
                )

                pred_ch_all = pred_ch_all.view(n_img, n_rt_samples).t()
                pred_rt_all = pred_rt_all.view(n_img, n_rt_samples).t()

                for img_pos, (img_idx_val, cat) in enumerate(zip(imgs_at_level, cat_ids)):
                    rts_samples = pred_rt_all[:, img_pos].cpu().numpy().tolist()
                    chs_samples = pred_ch_all[:, img_pos].cpu().numpy().tolist()
                    mean_rt  = float(np.mean(rts_samples))
                    mean_acc = float(np.mean([int(c) == cat for c in chs_samples]))

                    results_raw[level][cat][beep_s][img_idx_val] = {
                        'rt_mean':  mean_rt,
                        'acc_mean': mean_acc,
                    }

    # Aggregate into the curves_dict format
    overall_by_level = {}
    by_cat_by_level  = {}

    for level in ordered_levels:
        by_cat_by_level[level] = {}
        all_beep_data: dict[float, dict] = {}

        for cat in range(n_choices):
            per_beep_cat = {}
            for beep_s in unique_beeps:
                img_vals = results_raw[level][cat][beep_s]
                if not img_vals:
                    continue
                per_img_rt  = np.array([v['rt_mean']  for v in img_vals.values()])
                per_img_acc = np.array([v['acc_mean'] for v in img_vals.values()])
                n = len(per_img_rt)
                per_beep_cat[beep_s] = {
                    'beep_s':        beep_s,
                    'rt_mean':       float(per_img_rt.mean()),
                    'rt_std':        float(per_img_rt.std())          if n > 1 else 0.0,
                    'rt_sem':        float(per_img_rt.std() / n**0.5) if n > 1 else 0.0,
                    'acc_mean':      float(per_img_acc.mean()),
                    'acc_std':       float(per_img_acc.std())          if n > 1 else 0.0,
                    'acc_sem':       float(per_img_acc.std() / n**0.5) if n > 1 else 0.0,
                    'n_images':      n,
                    'n_trials':      n * n_rt_samples,
                    'per_image_rt':  per_img_rt.tolist(),
                    'per_image_acc': per_img_acc.tolist(),
                }
                if beep_s not in all_beep_data:
                    all_beep_data[beep_s] = {'rt': [], 'acc': []}
                all_beep_data[beep_s]['rt'].extend(per_img_rt.tolist())
                all_beep_data[beep_s]['acc'].extend(per_img_acc.tolist())

            if per_beep_cat:
                by_cat_by_level[level][cat] = _make_curves_dict(per_beep_cat)

        per_beep_overall = {}
        for beep_s, vals in all_beep_data.items():
            rt_arr  = np.array(vals['rt'])
            acc_arr = np.array(vals['acc'])
            n = len(rt_arr)
            per_beep_overall[beep_s] = {
                'beep_s':        beep_s,
                'rt_mean':       float(rt_arr.mean()),
                'rt_std':        float(rt_arr.std())          if n > 1 else 0.0,
                'rt_sem':        float(rt_arr.std() / n**0.5) if n > 1 else 0.0,
                'acc_mean':      float(acc_arr.mean()),
                'acc_std':       float(acc_arr.std())          if n > 1 else 0.0,
                'acc_sem':       float(acc_arr.std() / n**0.5) if n > 1 else 0.0,
                'n_images':      n,
                'n_trials':      n * n_rt_samples,
                'per_image_rt':  rt_arr.tolist(),
                'per_image_acc': acc_arr.tolist(),
            }
        overall_by_level[level] = _make_curves_dict(per_beep_overall)

    return {
        'scenario':       scenario,
        'ordered_levels': ordered_levels,
        'display_labels': display,
        'colors':         colors,
        'overall':        overall_by_level,
        'by_category':    by_cat_by_level,
    }


# ---------------------------------------------------------------------------
# Save curves to JSON
# ---------------------------------------------------------------------------

def save_curves_to_json(human_curves: dict, model_curves: dict, path: str | Path):
    """Save the human and model curves to a JSON file."""
    def _jsonify(d):
        if isinstance(d, dict):
            return {str(k): _jsonify(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)):
            return [_jsonify(x) for x in d]
        if isinstance(d, np.ndarray):
            return d.tolist()
        if isinstance(d, (np.floating, np.integer)):
            return d.item()
        return d

    payload = {
        'human': _jsonify(human_curves),
        'model': _jsonify(model_curves),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  [SAT level] Curves saved to {display_path(path)}")


# ---------------------------------------------------------------------------
# Persisting the full (per-observer) metrics
# ---------------------------------------------------------------------------

def save_satbench_metrics_full(metrics: dict, save_dir) -> Path:
    """
    Write every per-observer array (what Fig 4-7 need to be replicated) to
    ``satbench_metrics_full.json`` inside save_dir.

    Saved JSON structure
    --------------------
    {
      "scenario": str,
      "ordered_levels": [...],
      "display_labels": {...},
      "ermse": {
        "model_per_obs": [...],   # eRMSE per observer (%)
        "human_per_obs": [...]    # within-human eRMSE per observer (%)
      },
      "spearman": {
        "model_per_obs": [...],
        "human_per_obs": [...],
        "by_level": {"lvl": float, ...},
        "by_level_per_obs": {"lvl": [...], ...},
        "by_cat": {"cat_idx": [...], ...}
      },
      "spearman_human": {
        "by_cat": {"cat_idx": [...], ...}
      },
      "spearman_cat_median": {"cat_idx": float, ...},
      "spearman_cat_median_human": {"cat_idx": float, ...},
      "steepness": {
        "lvl": {"human": [kappa, sem], "model": [kappa, sem]}, ...
      },
      "overall": {"lvl": {"rmse": float, "spearman_r": float, ...}, ...},
      "mean_over_levels": {...}
    }
    """
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    def _make_serializable(obj):
        """Recursively convert numpy scalars/arrays to native Python types."""
        if isinstance(obj, dict):
            return {str(k): _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(x) for x in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if np.isnan(v) or np.isinf(v) else v
        return obj

    payload = _make_serializable({
        'scenario':                   metrics.get('scenario'),
        'ordered_levels':             metrics.get('ordered_levels', []),
        'display_labels':             metrics.get('display_labels', {}),
        'ermse':                      metrics.get('ermse', {}),
        'spearman':                   metrics.get('spearman', {}),
        'spearman_human':             metrics.get('spearman_human', {}),
        'spearman_cat_median':        metrics.get('spearman_cat_median', {}),
        'spearman_cat_median_human':  metrics.get('spearman_cat_median_human', {}),
        'steepness':                  metrics.get('steepness', {}),
        'overall':                    metrics.get('overall', {}),
        'mean_over_levels':           metrics.get('mean_over_levels', {}),
    })
    # Version tag so notebook can detect old vs new JSON format
    payload['_version'] = 2

    out_path = save_dir_path / 'satbench_metrics_full.json'
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  [SAT metrics] Full per-observer metrics saved to {display_path(out_path)}")
    return out_path


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def generate_latex_table(metrics: dict, scenario_name: str | None = None) -> str:
    """
    Build a LaTeX table with RMSE, Spearman rho and steepness (human/model).
    """
    scenario     = scenario_name or metrics.get('scenario', 'SAT')
    ordered_lvls = metrics['ordered_levels']
    display      = metrics['display_labels']

    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(
        r'\caption{NRM vs Human SAT metrics — ' + scenario.replace('_', r'\_') + '}'
    )
    lines.append(r'\label{tab:sat_metrics_' + scenario.replace('-', '_') + '}')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(
        r'Level & RMSE (\%) & Spearman $\rho$ & $\kappa_{\rm H}$ & $\kappa_{\rm M}$ \\'
    )
    lines.append(r'\midrule')

    for level in ordered_lvls:
        m   = metrics['overall'].get(level)
        lbl = display.get(level, level)
        if m is None:
            lines.append(f'{lbl} & -- & -- & -- & -- \\\\')
            continue
        rmse  = f"{m['rmse']:.2f}"       if not np.isnan(m['rmse'])             else '--'
        sp    = f"{m['spearman_r']:.3f}" if not np.isnan(m['spearman_r'])       else '--'
        sh    = f"{m['steepness_human']:.4f}" if not np.isnan(m['steepness_human']) else '--'
        sm    = f"{m['steepness_model']:.4f}" if not np.isnan(m['steepness_model']) else '--'
        lines.append(f'{lbl} & {rmse} & {sp} & {sh} & {sm} \\\\')

    lines.append(r'\midrule')
    avg      = metrics['mean_over_levels']
    rmse_avg = f"{avg['rmse']:.2f}"       if not np.isnan(avg['rmse'])        else '--'
    sp_avg   = f"{avg['spearman_r']:.3f}" if not np.isnan(avg['spearman_r'])  else '--'
    sh_avg   = f"{avg['steepness_human']:.4f}" if not np.isnan(avg['steepness_human']) else '--'
    sm_avg   = f"{avg['steepness_model']:.4f}" if not np.isnan(avg['steepness_model']) else '--'
    lines.append(
        r'\textit{Average} & ' + f'{rmse_avg} & {sp_avg} & {sh_avg} & {sm_avg} \\\\'
    )
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


# ===========================================================================
# Entry point: the full pipeline
# ===========================================================================

def run_sat_level_analysis(
        model,
        device,
        data: dict,
        save_dir: str | Path,
        n_rt_samples: int = 50,
        n_choices: int = 16,
        use_sem: bool = True,
        scenario_override: str | None = None,
        make_figures: bool = True,
        write_latex: bool = True,
) -> dict:
    """
    SAT level-analysis pipeline with the SATbench-equivalent metrics.

    Steps:
      1. Human curves per level
      2. Model curves per level (forward pass)
      3. Per-observer curves (from data['srids'])
      4. Figure 3 plots (beep and rt)
      5. Per-category plots
      6. Metrics: eRMSE, Spearman, steepness
      7. Curves JSON
      8. LaTeX table
      9. Figures 4-7

    ``make_figures`` / ``write_latex`` gate steps 4-5, 9 and step 8.
    ``pipelines/sat.ipynb`` sets both to False: the SAT figure and tables are
    built across *all* models by ``scripts/figures.py`` and
    ``scripts/tables.py``, and these per-model plots are not part of the
    paper. The curve and per-observer JSONs are always written — those are
    the inputs the paper outputs are assembled from.

    Returns a dict with every saved path and the metrics.
    """
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("SAT LEVEL ANALYSIS  (SATbench-equivalent metrics)")
    print("=" * 80)

    # 1. Human curves per level
    print("  Computing human curves per level...")
    human_curves = calculate_human_curves_by_level(data, n_choices=n_choices)
    scenario = scenario_override or human_curves['scenario']
    human_curves['scenario'] = scenario
    ordered_lvls = human_curves['ordered_levels']
    if not ordered_lvls:
        print("  [SAT level] No level found — analysis skipped.")
        return {}

    print(f"  Scenario: {scenario}  |  Levels: {ordered_lvls}")
    for lv in ordered_lvls:
        lbl = human_curves['display_labels'].get(lv, lv)
        beeps = human_curves['overall'].get(lv, {}).get('beeps_s', [])
        print(f"    {lbl}: beeps = {beeps}")

    # 2. Model curves per level
    print("  Computing model curves per level (forward pass)...")
    model_curves = calculate_model_curves_by_level(
        model, device, data,
        n_rt_samples=n_rt_samples,
        n_choices=n_choices,
    )
    model_curves['scenario'] = scenario

    # 3. Per-observer curves
    has_srids = 'srids' in data and len(data['srids']) > 0
    n_obs = 0
    if has_srids:
        print("  Computing per-observer curves (srid)...")
        per_obs_curves = _compute_per_observer_curves(data, ordered_lvls, n_choices=n_choices)
        n_obs = len(per_obs_curves) if per_obs_curves else 0
        print(f"    {n_obs} observers found in the test set.")
    else:
        per_obs_curves = None
        print("  [WARN] 'srids' unavailable — eRMSE/Spearman boxplots will be empty.")

    # 4. Write the JSON
    json_path = save_dir_path / 'sat_level_curves.json'
    save_curves_to_json(human_curves, model_curves, json_path)

    paths: dict[str, Path] = {}

    if make_figures:
        # 5. Figure 3 overview plots (2 modes: beep and rt)
        for x_mode, suffix in [('beep', 'beep_vs_acc'), ('rt', 'rt_vs_acc')]:
            p = save_dir_path / f'sat_level_{suffix}.png'
            plot_sat_by_level(human_curves, model_curves,
                              x_mode=x_mode, use_sem=use_sem, save_path=p)
            paths[f'level_{suffix}'] = p

        # 6. Per-category plots
        for x_mode, suffix in [('beep', 'beep_vs_acc'), ('rt', 'rt_vs_acc')]:
            p = save_dir_path / f'sat_level_by_cat_{suffix}.png'
            plot_sat_by_level_and_category(
                human_curves, model_curves,
                x_mode=x_mode, use_sem=use_sem,
                n_choices=n_choices, save_path=p,
            )
            paths[f'level_by_cat_{suffix}'] = p

    # 7. SATBench-equivalent metrics
    print("  Computing SATbench metrics (eRMSE, Spearman, Weibull steepness)...")
    metrics = compute_satbench_metrics(
        human_curves, model_curves, per_obs_curves, n_choices=n_choices
    )
    metrics['colors'] = human_curves['colors']

    # Print a summary
    print(f"\n  {'Level':<12} {'RMSE%':>8} {'Spearman':>10} {'κ_H':>10} {'κ_M':>10}")
    print("  " + "-" * 55)
    for lv in ordered_lvls:
        m = metrics['overall'].get(lv)
        if m:
            lbl = human_curves['display_labels'].get(lv, lv)
            rmse_s = f"{m['rmse']:8.2f}"     if not np.isnan(m['rmse'])       else '       --'
            sp_s   = f"{m['spearman_r']:10.3f}" if not np.isnan(m['spearman_r']) else '         --'
            kh_s   = f"{m['steepness_human']:10.5f}" if not np.isnan(m['steepness_human']) else '         --'
            km_s   = f"{m['steepness_model']:10.5f}" if not np.isnan(m['steepness_model']) else '         --'
            print(f"  {lbl:<12} {rmse_s} {sp_s} {kh_s} {km_s}")
    avg = metrics['mean_over_levels']
    rmse_a = f"{avg['rmse']:8.2f}"     if not np.isnan(avg['rmse'])       else '       --'
    sp_a   = f"{avg['spearman_r']:10.3f}" if not np.isnan(avg['spearman_r']) else '         --'
    kh_a   = f"{avg['steepness_human']:10.5f}" if not np.isnan(avg['steepness_human']) else '         --'
    km_a   = f"{avg['steepness_model']:10.5f}" if not np.isnan(avg['steepness_model']) else '         --'
    print(f"  {'Average':<12} {rmse_a} {sp_a} {kh_a} {km_a}")
    if n_obs > 0:
        ermse_m_vals = metrics['ermse']['model_per_obs']
        sp_m_vals    = metrics['spearman']['model_per_obs']
        if ermse_m_vals:
            print(f"\n  eRMSE (NRM vs obs, N={len(ermse_m_vals)}): "
                  f"median={np.nanmedian(ermse_m_vals):.2f}%, "
                  f"IQR=[{np.nanpercentile(ermse_m_vals,25):.2f}, "
                  f"{np.nanpercentile(ermse_m_vals,75):.2f}]%")
        if sp_m_vals:
            print(f"  Spearman ρ (NRM vs obs, N={len(sp_m_vals)}): "
                  f"median={np.nanmedian(sp_m_vals):.3f}, "
                  f"IQR=[{np.nanpercentile(sp_m_vals,25):.3f}, "
                  f"{np.nanpercentile(sp_m_vals,75):.3f}]")

    # 8. Full per-observer metrics JSON (for global notebook)
    full_json_path = save_satbench_metrics_full(metrics, save_dir_path)
    paths['metrics_full_json'] = full_json_path

    # 9. LaTeX table
    if write_latex:
        latex_str = generate_latex_table(metrics, scenario_name=scenario)
        latex_path = save_dir_path / 'sat_level_metrics_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_str)
        print(f"  [SAT metrics] LaTeX table saved to {display_path(latex_path)}")
        paths['latex_table'] = latex_path

    # 10. Figure 4-7
    if make_figures:
        print("  Generating Figures 4-7 (SATbench-equivalent)...")
        fig_paths = plot_satbench_figures(
            metrics=metrics,
            human_curves=human_curves,
            model_curves=model_curves,
            save_dir=save_dir_path,
            prefix='satbench',
            n_choices=n_choices,
        )
        paths.update(fig_paths)

    return {
        'human_curves': human_curves,
        'model_curves': model_curves,
        'metrics':      metrics,
        'paths':        paths,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Test-set evaluator
# ══════════════════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """Main evaluation class that orchestrates the entire evaluation process."""
    
    def __init__(self, model, device, n_choices):
        self.model = model
        self.device = device
        self.n_choices = n_choices
        self.metrics_calc = MetricsCalculator()
        self.human_analyzer = HumanAgreementAnalyzer()
        self.ll_calc = RTLogLikelihoodCalculator()
        self.dist_metrics_calc = RTDistributionMetrics()
        self.sat_curves_calc = SATCurvesCalculator()
    
    def collect_predictions(self, test_loader, n_reps=1, collect_rt_samples=False,
                        n_rt_samples=100):
        """
        Collect all predictions from the test set.
    
        Adds rt_samples_by_image_beep : dict {(img_idx, beep) -> list[float]}
        when collect_rt_samples=True and the batch carries time_limits (SAT).
        """
        all_data = {
            'pred_choices':  [],
            'pred_rts':      [],
            'human_choices': [],
            'true_labels':   [],
            'true_rts':      [],
            'image_indices': [],
            'embeddings':    [],
            'time_limits':   [],
        }
        all_levels = []   # list of strings, collected separately (non-tensor)
        all_srids  = []   # list of subject IDs, collected separately (non-tensor)

        rt_samples_by_image      = defaultdict(list) if collect_rt_samples else None
        rt_samples_by_image_beep = defaultdict(list) if collect_rt_samples else None
    
        print("Collecting predictions from test set...")
        if collect_rt_samples:
            print(f"  (Also collecting {n_rt_samples} RT samples per observation "
                "for likelihood estimation)")
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                print(f"  Processing batch {batch_idx+1}/{len(test_loader)}...", end='\r')
    
                embeddings    = batch['embeddings'].to(self.device)
                human_choices = batch['human_choices'].to(self.device)
                true_labels   = batch['choices'].to(self.device)
                true_rts      = batch['rts'].to(self.device)
                image_indices = batch['image_index'].to(self.device)
    
                if len(true_rts) < 2:
                    continue
    
                cov = {"img_embedding": embeddings}
                if "time_limits" in batch:
                    cov["time_limit"] = batch["time_limits"].to(self.device, dtype=torch.float32)
    
                if collect_rt_samples:
                    batch_size = len(true_rts)
    
                    expanded_embeddings = embeddings.repeat_interleave(n_rt_samples, dim=0)
                    expanded_cov        = {"img_embedding": expanded_embeddings}
                    if "time_limits" in batch:
                        beeps_rep = cov["time_limit"].repeat_interleave(n_rt_samples, dim=0)
                        expanded_cov["time_limit"] = beeps_rep.to(self.device, dtype=torch.float32)
    
                    _, _, pred_choices_all, pred_rts_all = self.model.simulate_nDDM(
                        batch_size=batch_size * n_rt_samples,
                        covariates=expanded_cov,
                        soft=True,
                        n_reps=1,
                        t=0.993,
                    )
    
                    # Shape: (n_rt_samples, batch_size)
                    pred_rts_all     = pred_rts_all.view(batch_size, n_rt_samples).t()
                    pred_choices_all = pred_choices_all.view(batch_size, n_rt_samples).t()
    
                    if n_reps == 1:
                        pred_choices = pred_choices_all[0]
                        pred_rts     = pred_rts_all[0]
                    else:
                        n_to_avg     = min(n_reps, n_rt_samples)
                        pred_choices = pred_choices_all[:n_to_avg].float().mean(dim=0)
                        pred_rts     = pred_rts_all[:n_to_avg].mean(dim=0)
    
                    pred_rts_all_cpu  = pred_rts_all.cpu().numpy()   # (n_rt_samples, B)
                    image_indices_cpu = image_indices.cpu().numpy()
    
                    for obs_idx in range(batch_size):
                        img_idx = int(image_indices_cpu[obs_idx])
                        samples = pred_rts_all_cpu[:, obs_idx].tolist()
    
                        # Per image only (non-SAT)
                        rt_samples_by_image[img_idx].extend(samples)
    
                        # SAT: accumulate per (img_idx, beep)
                        if "time_limits" in batch:
                            tl_val = float(batch["time_limits"][obs_idx].item())
                            if tl_val > 0:
                                beep_key = round(tl_val, 4)
                                rt_samples_by_image_beep[(img_idx, beep_key)].extend(samples)
    
                else:
                    _, _, pred_choices, pred_rts = self.model.simulate_nDDM(
                        batch_size=len(true_rts),
                        covariates=cov,
                        soft=True,
                        n_reps=n_reps,
                        t=0.993,
                    )
    
                all_data['pred_choices'].append(pred_choices.cpu())
                all_data['pred_rts'].append(pred_rts.cpu())
                all_data['human_choices'].append(human_choices.cpu())
                all_data['true_labels'].append(true_labels.cpu())
                all_data['true_rts'].append(true_rts.cpu())
                all_data['image_indices'].append(image_indices.cpu())
                all_data['embeddings'].append(embeddings.cpu())
    
                if "time_limits" in batch:
                    all_data['time_limits'].append(batch['time_limits'].cpu())
                else:
                    all_data['time_limits'].append(torch.full((len(true_rts),), -1.0))

                if "levels" in batch:
                    all_levels.extend(batch['levels'])
                else:
                    all_levels.extend(['unknown'] * len(true_rts))

                if "srids" in batch:
                    all_srids.extend(batch['srids'])
                else:
                    all_srids.extend(['unknown'] * len(true_rts))

        print()

        for key in all_data:
            all_data[key] = torch.cat(all_data[key], dim=0).numpy()

        all_data['levels'] = all_levels   # list of strings, one per trial
        all_data['srids']  = all_srids    # list of subject IDs, one per trial

        if collect_rt_samples:
            all_data['rt_samples_by_image']      = dict(rt_samples_by_image)
            all_data['rt_samples_by_image_beep'] = dict(rt_samples_by_image_beep)
            print(f"  Collected RT samples for {len(rt_samples_by_image)} unique stimuli")
            if rt_samples_by_image_beep:
                print(f"  Collected RT samples for {len(rt_samples_by_image_beep)} "
                    "(stimulus, beep) pairs [SAT conditioned]")
    
        return all_data
    
    def calculate_predictive_likelihoods(self, data, method='gaussian'):
        """
        Compute the predictive log-likelihood of the RTs.
    
        SAT (time_limits > 0 + rt_samples_by_image_beep):
        - Uses the versions conditioned on (image, beep)
        - Balanced sampling: the model uses as many samples as there are
            human training observations (removing the artificial estimation
            advantage that put Model LL ~ 0 in the scatter)
        - Also produces the separate per-beep plot
    
        Non-SAT: unconditioned per-image likelihoods.
        """
        if 'rt_samples_by_image' not in data:
            raise ValueError(
                "RT samples not found. Run collect_predictions() with collect_rt_samples=True"
            )
    
        is_sat = (
            'time_limits' in data
            and bool((data['time_limits'] > 0).any())
            and 'rt_samples_by_image_beep' in data
            and len(data['rt_samples_by_image_beep']) > 0
        )
    
        print("\nCalculating predictive log-likelihoods...")
        if is_sat:
            print("  [SAT mode] Using (image, beep)-conditioned likelihood "
                "with balanced sample count")
    
        if is_sat:
            print("  Computing human-human LOSO likelihood (conditioned)...")
            hh_results = self.ll_calc.calculate_human_human_loso_ll_conditioned(
                image_indices=data['image_indices'],
                rts=data['true_rts'],
                time_limits=data['time_limits'],
                method=method,
            )
    
            print("  Computing model RT likelihood (conditioned, balanced)...")
            model_results = self.ll_calc.calculate_model_rt_ll_conditioned(
                image_indices=data['image_indices'],
                human_rts=data['true_rts'],
                time_limits=data['time_limits'],
                model_rts_samples_by_image_beep=data['rt_samples_by_image_beep'],
                method=method,
                balance_n_samples=True,
                hh_n_train=hh_results.get('per_n_train'),
            )
    
        else:
            print("  Computing human-human LOSO likelihood...")
            hh_results = self.ll_calc.calculate_human_human_loso_ll(
                image_indices=data['image_indices'],
                rts=data['true_rts'],
                method=method,
            )
    
            print("  Computing model RT likelihood...")
            model_results = self.ll_calc.calculate_model_rt_ll(
                image_indices=data['image_indices'],
                human_rts=data['true_rts'],
                model_rts_samples=data['rt_samples_by_image'],
                method=method,
            )
    
        print("  Comparing predictive performance...")
        comparison = self.ll_calc.compare_predictive_performance(hh_results, model_results)
    
        return {
            'human_human':     hh_results,
            'model':           model_results,
            'comparison':      comparison,
            'method':          method,
            'sat_conditioned': is_sat,
        }
    
    def calculate_distribution_metrics(self, data):
        """
        Calculate RT distribution comparison metrics (KS, quantiles, moments).
        
        Args:
            data: Data from collect_predictions (must include rt_samples_by_image)
            
        Returns:
            dict: {
                'human_human': human-human distribution comparisons,
                'model_human': model-human distribution comparisons
            }
        """
        if 'rt_samples_by_image' not in data:
            raise ValueError(
                "RT samples not found in data. Please run collect_predictions() "
                "with collect_rt_samples=True"
            )
        
        print("\nCalculating RT distribution metrics...")
        
        # Human-Human distribution comparison
        print("  Computing human-human distribution similarity...")
        #hh_dist = self.dist_metrics_calc.compare_human_human_distributions(
        hh_dist = self.dist_metrics_calc.compare_human_human_distributions_split_half(
            image_indices=data['image_indices'],
            rts=data['true_rts']
        )
        
        # Model-Human distribution comparison
        print("  Computing model-human distribution similarity...")
        mh_dist = self.dist_metrics_calc.compare_model_human_distributions(
            image_indices=data['image_indices'],
            human_rts=data['true_rts'],
            model_rt_samples=data['rt_samples_by_image']
        )
        
        return {
            'human_human': hh_dist,
            'model_human': mh_dist
        }
    
    def calculate_all_metrics(self, data):
        """Calculate all evaluation metrics."""
        metrics = {}
        
        # Human performance
        human_acc, human_correct, human_total = self.metrics_calc.calculate_accuracy(
            data['human_choices'], data['true_labels']
        )
        metrics['human_accuracy'] = human_acc
        metrics['human_correct'] = human_correct
        metrics['human_total'] = human_total
        metrics['human_auc'] = self.metrics_calc.calculate_auc(data['human_choices'], data['true_labels'], self.n_choices)
        
        # Model performance
        model_acc, model_correct, model_total = self.metrics_calc.calculate_accuracy(
            data['pred_choices'], data['true_labels']
        )
        metrics['model_accuracy'] = model_acc
        metrics['model_correct'] = model_correct
        metrics['model_total'] = model_total
        
        # Model-Human agreement
        model_human_acc, mh_match, mh_total = self.metrics_calc.calculate_accuracy(
            data['pred_choices'], data['human_choices']
        )
        metrics['model_human_agreement'] = model_human_acc
        metrics['model_human_match'] = mh_match
        
        # Model AUC (using predicted choices as probabilities)
        metrics['model_auc'] = self.metrics_calc.calculate_auc(
            data['pred_choices'], data['true_labels'], self.n_choices
        )
        
        # RT metrics
        metrics['rt_metrics'] = self.metrics_calc.calculate_rt_metrics(
            data['pred_rts'], data['true_rts']
        )
        
        # Agreement breakdown
        metrics['agreement_breakdown'] = self.metrics_calc.calculate_agreement_breakdown(
            data['pred_choices'], data['human_choices'], data['true_labels']
        )
        
        # Human-human agreement
        metrics['human_human_analysis'] = self.human_analyzer.analyze_multi_response_images(
            data['image_indices'], data['human_choices'], 
            data['true_labels'], data['true_rts']
        )
        
        return metrics
    
    def calculate_sat_benchmark(self, data, n_rt_samples=50):
        """
        Compute SAT curves for humans and NRM (per category and overall).
    
        Called by evaluate_test_set when the dataset is SAT
        (at least one trial with time_limit > 0).
    
        Args:
            data:          output of collect_predictions
                        (must contain 'time_limits', 'embeddings',
                            'true_labels', 'image_indices',
                            'human_choices', 'true_rts')
            n_rt_samples:  model samples per (image, beep)
    
        Returns:
            dict:
                'human_overall':     aggregated human SAT curve
                'human_by_category': human SAT curves per category
                'model_by_category': NRM SAT curves per category
                'model_overall':     NRM SAT curve aggregated
            or None if no SAT trial is found.
        """
        time_limits = data['time_limits']
        if not (time_limits > 0).any():
            print("  [SAT] No SAT trials found, benchmark skipped.")
            return None
    
        try:
            from datasets.sat.loader import CATEGORIES as SAT_CATEGORIES
            cat_names = (SAT_CATEGORIES
                        if len(SAT_CATEGORIES) == self.n_choices
                        else None)
        except ImportError:
            cat_names = None
    
        print("  Computing human SAT curves (overall)...")
        human_overall = self.sat_curves_calc.calculate_human_sat_curves(
            image_indices=data['image_indices'],
            true_labels=data['true_labels'],
            human_choices=data['human_choices'],
            true_rts=data['true_rts'],
            time_limits=time_limits,
        )
    
        print("  Computing human SAT curves (per category)...")
        human_by_cat = self.sat_curves_calc.calculate_human_sat_curves_by_category(
            image_indices=data['image_indices'],
            true_labels=data['true_labels'],
            human_choices=data['human_choices'],
            true_rts=data['true_rts'],
            time_limits=time_limits,
            category_names=cat_names,
        )
    
        print("  Computing NRM SAT curves (per category, fresh forward pass)...")
        model_by_cat = self.sat_curves_calc.calculate_model_sat_curves_by_category(
            model=self.model,
            device=self.device,
            data=data,
            n_rt_samples=n_rt_samples,
            category_names=cat_names,
        )
    
        print("  Aggregating the overall NRM curve...")
        model_overall = SATCurvesCalculator.aggregate_model_curves_overall(model_by_cat)
    
        return {
            'human_overall':     human_overall,
            'human_by_category': human_by_cat,
            'model_by_category': model_by_cat,
            'model_overall':     model_overall,
        }
 

    def save_detailed_results(self, data, save_dir):
        """Save detailed prediction results to CSV."""
        results_df = pd.DataFrame({
            'image_index': data['image_indices'],
            'true_label': data['true_labels'],
            'human_choice': data['human_choices'],
            'model_choice': data['pred_choices'],
            'human_correct': (data['human_choices'] == data['true_labels']).astype(int),
            'model_correct': (data['pred_choices'] == data['true_labels']).astype(int),
            'model_matches_human': (data['pred_choices'] == data['human_choices']).astype(int),
            'human_rt': data['true_rts'],
            'model_rt': data['pred_rts'],
            'rt_error': data['pred_rts'] - data['true_rts']
        })
        
        save_path = Path(save_dir) / 'detailed_predictions.csv'
        results_df.to_csv(save_path, index=False)
        return save_path
    
    def evaluate_test_set(self, test_loader, n_reps=1, save_dir='outputs', 
                          compute_rt_likelihoods=True, n_rt_samples=100, 
                          rt_ll_method='gaussian',
                          compute_sat_benchmark=False,       # <-Added
                          n_sat_rt_samples=50):             # <- Added
        """
        Run complete evaluation on test set.
        
        Args:
            test_loader: DataLoader for test data
            n_reps: Number of simulation repetitions for averaged predictions
            save_dir: Directory to save outputs
            compute_rt_likelihoods: Whether to compute RT predictive likelihoods
            n_rt_samples: Number of RT samples for likelihood estimation
            rt_ll_method: Method for RT likelihood ('gaussian' or 'kde')
        
        Returns:
            dict: Complete results including metrics, plots, and data
        """
        # Collect predictions (and optionally RT samples in the same pass)
        data = self.collect_predictions(
            test_loader,
            n_reps=n_reps,
            collect_rt_samples=compute_rt_likelihoods,
            n_rt_samples=n_rt_samples
        )
        print(f"Collected {len(data['pred_choices'])} test samples")

        # For SAT datasets, filter out cattut/tutorial trials (time_limit <= 0).
        # These trials have no beep constraint and can have arbitrary RT values,
        # which otherwise pollute error_analysis.png and metric calculations.
        if bool((data['time_limits'] > 0).any()):
            sat_mask = data['time_limits'] > 0
            n_removed = int((~sat_mask).sum())
            if n_removed > 0:
                _array_keys = ['pred_choices', 'pred_rts', 'human_choices',
                               'true_labels', 'true_rts', 'image_indices',
                               'embeddings', 'time_limits']
                for _k in _array_keys:
                    if _k in data and isinstance(data[_k], np.ndarray):
                        data[_k] = data[_k][sat_mask]
                _sat_list = sat_mask.tolist()
                if 'levels' in data and isinstance(data['levels'], list):
                    data['levels'] = [l for l, m in zip(data['levels'], _sat_list) if m]
                if 'srids' in data and isinstance(data['srids'], list):
                    data['srids'] = [s for s, m in zip(data['srids'], _sat_list) if m]
                print(f"  [SAT] Filtered out {n_removed} cattut/tutorial trials "
                      f"(time_limit <= 0) — {len(data['pred_choices'])} experimental "
                      f"trials remain.")

        # Calculate metrics
        print("\nCalculating metrics...")
        metrics = self.calculate_all_metrics(data)
        
        # RT Likelihood calculations (using samples already collected)
        rt_likelihood_results = None
        rt_distribution_results = None
        if compute_rt_likelihoods:
            print("\n" + "="*80)
            print("RT PREDICTIVE LIKELIHOOD ANALYSIS")
            print("="*80)
            
            rt_likelihood_results = self.calculate_predictive_likelihoods(
                data, method=rt_ll_method
            )
            
            print("\n" + "="*80)
            print("RT DISTRIBUTION COMPARISON")
            print("="*80)
            
            rt_distribution_results = self.calculate_distribution_metrics(data)
        
        # Create plots
        print("\nGenerating plots...")
        plotter = Plotter(save_dir)
        
        plot_paths = {}
        plot_paths['confusion_matrices'] = plotter.plot_confusion_matrices(
            data['true_labels'], data['human_choices'], data['pred_choices'],
            metrics['human_accuracy'], metrics['model_accuracy'],
            metrics['model_human_agreement'], self.n_choices
        )
        
        plot_paths['rt_correlations'] = plotter.plot_rt_correlations(
            data['true_rts'], data['pred_rts'], 
            metrics['rt_metrics'],
            metrics['human_human_analysis']
        )
        
        plot_paths['error_analysis'] = plotter.plot_error_analysis(
            data['true_labels'], data['human_choices'], data['pred_choices'],
            data['true_rts'], data['pred_rts'],
            metrics['agreement_breakdown'], metrics['rt_metrics'],
            metrics['model_human_agreement'], self.n_choices
        )
        
        # -- SAT: compute the pooled LL now; both plot sets need it --
        # Only for SAT (time_limits in data) with samples available.
        pooled_ll_results = None
        _is_sat_with_samples = (
            'time_limits' in data
            and bool((data['time_limits'] > 0).any())
            and 'rt_samples_by_image_beep' in data
            and len(data['rt_samples_by_image_beep']) > 0
            and compute_rt_likelihoods
        )
        if _is_sat_with_samples:
            print("\n" + "="*80)
            print("RT PREDICTIVE LL — POOLED PER BEEP")
            print("="*80)
            print("  Pools every RT by deadline (image identity ignored).")
            print("  LOSO over N~100-300 observations: stable estimates, no sigma->0.")
            pooled_ll_results = calculate_hh_and_model_ll_beep_pooled(
                image_indices=data['image_indices'],
                human_rts=data['true_rts'],
                time_limits=data['time_limits'],
                model_rts_samples_by_image_beep=data['rt_samples_by_image_beep'],
                method=rt_ll_method,
                min_obs=10,
            )
            for beep, res in sorted(pooled_ll_results.items()):
                ks_s = res.get('ks_stat', float('nan'))
                ks_p = res.get('ks_pval', float('nan'))
                ks_str = (f'KS={ks_s:.3f}(p={ks_p:.3f})'
                          if not (ks_s != ks_s) else 'KS=n/a')
                n_mod = res.get('n_model_samples', 0)
                print(
                    f"  Beep {beep:.3f}s: N_human={res['n_obs']} N_model={n_mod}  "
                    f"HH_LL={res['hh_mean']:.3f}  Model_LL={res['model_mean']:.3f}  "
                    f"Δ_LL={res['model_advantage']:.3f}  "
                    f"RT_human={res.get('human_mean_rt', float('nan')):.3f}s  "
                    f"RT_model={res.get('model_mean_rt', float('nan')):.3f}s  "
                    f"{ks_str}"
                )

        # Add RT likelihood plot if computed
        if rt_likelihood_results is not None:
            sat_cond = rt_likelihood_results.get('sat_conditioned', False)

            if sat_cond:
                # SAT plot 1: rt_likelihood_comparison_beep_xx - balanced per (image, deadline)
                # One scatter point = one stimulus (img, beep), a 2D scatter.
                print("  [SAT] Generating per-beep RT likelihood plots (balanced per image+beep)...")
                per_beep_paths = create_ll_comparison_plots_per_beep_separate(
                    rt_likelihood_results['human_human'],
                    rt_likelihood_results['model'],
                    save_dir=save_dir,
                    filename_prefix='rt_likelihood_comparison_beep',
                )
                for i, p in enumerate(per_beep_paths):
                    plot_paths[f'rt_likelihood_beep_{i+1}'] = p
                if per_beep_paths:
                    print(f"    Saved {len(per_beep_paths)} per-beep balanced plots.")

                # SAT plot 2: rt_likelihood_comparison_beep_cross_xx
                # HH: leave-one-image-out at beep level (a stable fit on ~195 RTs)
                # Model: per-image conditioned (different samples for each image)
                # Scatter: one point = the per-image mean, so the 2D cloud is not degenerate.
                if 'rt_samples_by_image_beep' in data:
                    print("  [SAT] Computing cross-image LL (leave-one-image-out HH + per-image model)...")
                    cross_ll_results = calculate_hh_model_ll_beep_cross_image(
                        image_indices=data['image_indices'],
                        human_rts=data['true_rts'],
                        time_limits=data['time_limits'],
                        model_rts_samples_by_image_beep=data['rt_samples_by_image_beep'],
                        method=rt_ll_method,
                    )
                    cross_paths = create_ll_comparison_plots_beep_cross_image(
                        cross_ll_results,
                        save_dir=save_dir,
                        filename_prefix='rt_likelihood_comparison_beep_cross',
                    )
                    for i, p in enumerate(cross_paths):
                        plot_paths[f'rt_likelihood_beep_cross_{i+1}'] = p
                    if cross_paths:
                        print(f"    Saved {len(cross_paths)} cross-image per-beep plots.")
            else:
                # Non-SAT: standard global plot
                print("  Generating RT likelihood comparison plot...")
                ll_plot_path = create_ll_comparison_plot(
                    rt_likelihood_results['human_human'],
                    rt_likelihood_results['model'],
                    rt_likelihood_results['comparison'],
                    save_path=Path(save_dir) / 'rt_likelihood_comparison.png'
                )
                plot_paths['rt_likelihood'] = ll_plot_path

        # Add RT distribution comparison plot if computed
        if rt_distribution_results is not None:
            print("  Generating RT distribution comparison plot...")
            dist_plot_path = create_distribution_comparison_plots(
                rt_distribution_results['human_human'],
                rt_distribution_results['model_human'],
                save_path=Path(save_dir) / 'rt_distribution_comparison.png'
            )
            plot_paths['rt_distribution'] = dist_plot_path

        # Save detailed results
        print("Saving detailed results...")
        csv_path = self.save_detailed_results(data, save_dir)

        # Save RT likelihood results if computed
        if rt_likelihood_results is not None:
            self._save_rt_likelihood_results(rt_likelihood_results, save_dir)

        # Save RT distribution results if computed
        if rt_distribution_results is not None:
            self._save_rt_distribution_results(rt_distribution_results, save_dir)

        sat_results = None
        if compute_sat_benchmark:
            print("\n" + "="*80)
            print("SAT BENCHMARK")
            print("="*80)
            sat_results = self.calculate_sat_benchmark(
                data=data,
                n_rt_samples=n_sat_rt_samples,
            )
            if sat_results is not None:
                print("  Generating SAT plots (16 categories)...")
                sat_plotter = SATPlotter(save_dir)
                sat_plot_paths = sat_plotter.plot_sat_by_category(
                    human_by_cat=sat_results['human_by_category'],
                    model_by_cat=sat_results['model_by_category'],
                    human_overall=sat_results['human_overall'],
                    model_overall=sat_results['model_overall'],
                    use_sem=True,
                    n_choices=self.n_choices,
                )
                plot_paths.update(sat_plot_paths)
                sat_plotter.save_sat_table(
                    human_by_cat=sat_results['human_by_category'],
                    model_by_cat=sat_results['model_by_category'],
                    human_overall=sat_results['human_overall'],
                    model_overall=sat_results['model_overall'],
                )

        # SAT: analysis by perturbation level (blur / noise / color-gray)
        if compute_sat_benchmark and 'time_limits' in data and data.get('levels') is not None:
            print("\n" + "="*80)
            print("SAT LEVEL ANALYSIS (degradation levels)")
            print("="*80)
            level_results = run_sat_level_analysis(
                model=self.model,
                device=self.device,
                data=data,
                save_dir=save_dir,
                n_rt_samples=n_sat_rt_samples,
                n_choices=self.n_choices,
                use_sem=True,
            )
            if level_results:
                for k, p in level_results.get('paths', {}).items():
                    plot_paths[f'sat_level_{k}'] = p

        # SAT: KDE+QQ plots and a summary CSV from the pooled analysis.
        if pooled_ll_results is not None:
            print("  Generating pooled (KDE+QQ) plots per deadline...")
            pooled_paths = create_ll_comparison_plots_beep_pooled_separate(
                pooled_ll_results,
                save_dir=save_dir,
                filename_prefix='rt_likelihood_comparison_beep_pooled',
            )
            for i, p in enumerate(pooled_paths):
                plot_paths[f'rt_likelihood_beep_pooled_{i+1}'] = p
            if pooled_paths:
                print(f"  Saved {len(pooled_paths)} pooled (KDE+QQ) plots.")

            # Pooled summary CSV
            pooled_csv_path = Path(save_dir) / 'rt_ll_beep_pooled_summary.csv'
            save_pooled_beep_ll_csv(pooled_ll_results, str(pooled_csv_path))
            print(f"  Pooled CSV saved: {pooled_csv_path.name}")

        return {
            'metrics': metrics,
            'data': data,
            'rt_likelihood': rt_likelihood_results,
            'rt_likelihood_pooled': pooled_ll_results,
            'rt_distribution': rt_distribution_results,
            'sat': sat_results,
            'plot_paths': plot_paths,
            'csv_path': csv_path,
            'save_dir': save_dir
        }
    
    def _save_rt_likelihood_results(self, rt_likelihood_results, save_dir):
        """Save RT likelihood results to CSV files."""
        save_path = Path(save_dir)
        
        # Human-human results per stimulus
        hh_data = []
        for img_idx, mean_ll in rt_likelihood_results['human_human']['per_stimulus_mean_ll'].items():
            hh_data.append({
                'image_idx': img_idx,
                'mean_ll': mean_ll,
                'n_observations': len(rt_likelihood_results['human_human']['per_stimulus_ll'][img_idx])
            })
        
        if len(hh_data) > 0:
            hh_df = pd.DataFrame(hh_data)
            hh_df.to_csv(save_path / 'human_human_rt_ll_per_stimulus.csv', index=False)
        
        # Model results per stimulus
        model_data = []
        for img_idx, mean_ll in rt_likelihood_results['model']['per_stimulus_mean_ll'].items():
            model_data.append({
                'image_idx': img_idx,
                'mean_ll': mean_ll,
                'n_observations': len(rt_likelihood_results['model']['per_stimulus_ll'][img_idx])
            })
        
        if len(model_data) > 0:
            model_df = pd.DataFrame(model_data)
            model_df.to_csv(save_path / 'model_rt_ll_per_stimulus.csv', index=False)
        
        # Comparison per stimulus
        comparison_data = []
        for img_idx, advantage in rt_likelihood_results['comparison']['per_stimulus_advantage'].items():
            comparison_data.append({
                'image_idx': img_idx,
                'model_advantage': advantage,
                'human_human_ll': rt_likelihood_results['human_human']['per_stimulus_mean_ll'].get(img_idx),
                'model_ll': rt_likelihood_results['model']['per_stimulus_mean_ll'].get(img_idx)
            })
        
        if len(comparison_data) > 0:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(save_path / 'rt_ll_comparison_per_stimulus.csv', index=False)
        
        print(f"  Saved RT likelihood CSVs to {display_path(save_dir)}/")
    
    def _save_rt_distribution_results(self, rt_distribution_results, save_dir):
        """Save RT distribution metrics to CSV files."""
        save_path = Path(save_dir)
        
        # Model-human per-stimulus metrics
        mh_data = []
        for img_idx, metrics in rt_distribution_results['model_human']['per_stimulus'].items():
            row = {'image_idx': img_idx}
            
            # KS metrics
            if 'ks' in metrics:
                row['ks_statistic'] = metrics['ks']['ks_statistic']
                row['ks_pvalue'] = metrics['ks']['ks_pvalue']
            
            # Quantile metrics
            if 'quantiles' in metrics:
                row['quantile_mae'] = metrics['quantiles']['mean_absolute_error']
                row['quantile_rel_error'] = metrics['quantiles']['mean_relative_error']
            
            # Moment metrics
            if 'moments' in metrics:
                row['mean_diff'] = metrics['moments']['differences']['mean_diff']
                row['std_diff'] = metrics['moments']['differences']['std_diff']
                row['skewness_diff'] = metrics['moments']['differences']['skewness_diff']
                row['kurtosis_diff'] = metrics['moments']['differences']['kurtosis_diff']
                
                row['human_mean'] = metrics['moments']['sample1_moments']['mean']
                row['model_mean'] = metrics['moments']['sample2_moments']['mean']
                row['human_std'] = metrics['moments']['sample1_moments']['std']
                row['model_std'] = metrics['moments']['sample2_moments']['std']
                row['human_skewness'] = metrics['moments']['sample1_moments']['skewness']
                row['model_skewness'] = metrics['moments']['sample2_moments']['skewness']
                row['human_kurtosis'] = metrics['moments']['sample1_moments']['kurtosis']
                row['model_kurtosis'] = metrics['moments']['sample2_moments']['kurtosis']
            
            mh_data.append(row)
        
        if len(mh_data) > 0:
            mh_df = pd.DataFrame(mh_data)
            mh_df.to_csv(save_path / 'rt_distribution_per_stimulus.csv', index=False)
        
        # Overall summary
        overall = rt_distribution_results['model_human']['overall']
        summary_data = {
            'metric': [],
            'value': []
        }
        
        if overall['ks_distance']['mean'] is not None:
            summary_data['metric'].extend([
                'ks_distance_mean', 'ks_distance_median', 'ks_distance_std', 'ks_pvalue_mean'
            ])
            summary_data['value'].extend([
                overall['ks_distance']['mean'],
                overall['ks_distance']['median'],
                overall['ks_distance']['std'],
                overall['ks_distance']['mean_pvalue']
            ])
        
        if overall['quantile_mae']['mean'] is not None:
            summary_data['metric'].extend(['quantile_mae_mean', 'quantile_mae_median'])
            summary_data['value'].extend([
                overall['quantile_mae']['mean'],
                overall['quantile_mae']['median']
            ])
        
        if overall['moment_differences']['mean']['mean'] is not None:
            for moment in ['mean', 'std', 'skewness', 'kurtosis']:
                summary_data['metric'].extend([
                    f'{moment}_diff_mean', f'{moment}_diff_std'
                ])
                summary_data['value'].extend([
                    overall['moment_differences'][moment]['mean'],
                    overall['moment_differences'][moment]['std']
                ])
        
        if len(summary_data['metric']) > 0:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(save_path / 'rt_distribution_summary.csv', index=False)
        
        print(f"  Saved RT distribution CSVs to {display_path(save_dir)}/")

    
    def print_summary(self, results):
        """Print formatted summary of evaluation results."""
        metrics = results['metrics']
        data = results['data']
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nDataset:")
        print(f"  Total test samples:              {len(data['pred_choices'])}")
        
        hh = metrics['human_human_analysis']
        if hh is not None:
            print(f"  Unique images:                   {hh['n_images_total']}")
            print(f"  Images with multiple responses:  {hh['n_images_multi_response']}")
        
        print(f"\nAccuracy Metrics:")
        print(f"  Human accuracy:                  {metrics['human_accuracy']:.4f}")
        print(f"  Model accuracy:                  {metrics['model_accuracy']:.4f}")
        print(f"  Model-human agreement:           {metrics['model_human_agreement']:.4f}")
        
        if hh is not None and hh.get('avg_pairwise_agreement') is not None:
            print(f"  Human-human agreement:           {hh['avg_pairwise_agreement']:.4f}")
            if hh.get('cohens_kappa') is not None:
                print(f"  Cohen's kappa (n={hh['n_images_for_kappa']}):        {hh['cohens_kappa']:.4f}")
        
        rt = metrics['rt_metrics']
        print(f"\nReaction Time Metrics:")
        print(f"  Mean human RT:                   {rt['mean_true']:.4f}s")
        print(f"  Mean model RT:                   {rt['mean_predicted']:.4f}s")
        print(f"  RT correlation (Model-Human):    {rt['pearson_r']:.4f}")
        
        if hh is not None and hh.get('rt_correlation') is not None:
            print(f"  RT correlation (Human-Human):    {hh['rt_correlation']['r']:.4f}")
        
        print(f"  RT MAE:                          {rt['mae']:.4f}s")
        
        # RT Predictive Likelihood results
        if results.get('rt_likelihood') is not None:
            rt_ll = results['rt_likelihood']
            print(f"\nRT Predictive Log-Likelihood ({rt_ll['method']}):")
            
            hh_ll = rt_ll['human_human']
            if hh_ll['overall_mean_ll'] is not None:
                print(f"  Human-Human (LOSO):")
                print(f"    Mean LL:                       {hh_ll['overall_mean_ll']:.4f}")
                print(f"    Stimuli evaluated:             {hh_ll['n_stimuli_evaluated']}")
                print(f"    Total predictions:             {hh_ll['n_total_predictions']}")
            
            model_ll = rt_ll['model']
            if model_ll['overall_mean_ll'] is not None:
                print(f"  Model:")
                print(f"    Mean LL:                       {model_ll['overall_mean_ll']:.4f}")
                print(f"    Stimuli evaluated:             {model_ll['n_stimuli_evaluated']}")
                print(f"    Total predictions:             {model_ll['n_total_predictions']}")
            
            comp = rt_ll['comparison']
            if comp['n_common_stimuli'] > 0:
                print(f"  Comparison (on {comp['n_common_stimuli']} common stimuli):")
                print(f"    Model advantage:               {comp['model_advantage']:.4f}")
                print(f"    Model win rate:                {comp['win_rate']*100:.1f}%")
                print(f"    Model wins:                    {comp['model_wins']}")
                print(f"    Human-Human wins:              {comp['human_human_wins']}")
                
                if comp['model_advantage'] > 0:
                    print(f"    → Model has BETTER RT prediction")
                elif comp['model_advantage'] < 0:
                    print(f"    → Human-Human has BETTER RT prediction")
                else:
                    print(f"    → Equal RT prediction performance")
        
        # RT Distribution Metrics
        if results.get('rt_distribution') is not None:
            rt_dist = results['rt_distribution']
            mh = rt_dist['model_human']['overall']
            
            print(f"\nRT Distribution Comparison Metrics:")
            
            if mh['ks_distance']['mean'] is not None:
                print(f"  Kolmogorov-Smirnov Distance:")
                print(f"    Mean:                          {mh['ks_distance']['mean']:.4f}")
                print(f"    Median:                        {mh['ks_distance']['median']:.4f}")
                print(f"    (lower is better, 0 = perfect match)")
            
            if mh['quantile_mae']['mean'] is not None:
                print(f"  Quantile Matching (MAE):")
                print(f"    Mean:                          {mh['quantile_mae']['mean']:.4f}s")
                print(f"    Median:                        {mh['quantile_mae']['median']:.4f}s")
            
            if mh['moment_differences']['mean']['mean'] is not None:
                print(f"  Moment Differences (Model - Human):")
                print(f"    Mean RT difference:            {mh['moment_differences']['mean']['mean']:.4f}s")
                print(f"    Std difference:                {mh['moment_differences']['std']['mean']:.4f}s")
                print(f"    Skewness difference:           {mh['moment_differences']['skewness']['mean']:.4f}")
                print(f"    Kurtosis difference:           {mh['moment_differences']['kurtosis']['mean']:.4f}")
            
            print(f"    Stimuli evaluated:             {mh['n_stimuli_evaluated']}")
        
        print(f"\nAgreement Breakdown:")
        for key, value in metrics['agreement_breakdown'].items():
            print(f"  {key:30s} {value['count']:6d} ({value['pct']*100:5.2f}%)")
        
        print(f"\nSaved outputs:")
        print(f"  Directory:        {display_path(results['save_dir'])}/")
        print(f"  Detailed CSV:     {results['csv_path'].name}")
        for plot_name, plot_path in results['plot_paths'].items():
            from pathlib import Path as _P
            _pp = _P(plot_path) if isinstance(plot_path, str) else plot_path
            print(f"  {plot_name:20s}  {_pp.name}")
        
        if results.get('rt_likelihood') is not None:
            print(f"\n  RT Likelihood CSVs:")
            print(f"    human_human_rt_ll_per_stimulus.csv")
            print(f"    model_rt_ll_per_stimulus.csv")
            print(f"    rt_ll_comparison_per_stimulus.csv")
        
        if results.get('rt_distribution') is not None:
            print(f"\n  RT Distribution CSVs:")
            print(f"    rt_distribution_per_stimulus.csv")
            print(f"    rt_distribution_summary.csv")
        
        print("="*80)
        
        # Human-Human vs Model-Human table
        self.print_comprehensive_table(results)
    
    def print_comprehensive_table(self, results):
        """
        Print the comparison table: Human-Human vs Model-Human.
        
        Shows all metrics in a side-by-side comparison format.
        """
        print("\n" + "="*100)
        print("COMPREHENSIVE PERFORMANCE COMPARISON: Human-Human Baseline vs Model-Human")
        print("="*100)
        
        # Extract metrics
        metrics = results['metrics']
        rt_likelihood = results.get('rt_likelihood')
        rt_distribution = results.get('rt_distribution')
        
        # Prepare data for table
        # Row 1: Human-Human (LOSO baseline)
        # Row 2: Model-Human
        
        # Column headers
        headers = [
            "Metric",
            "Log-Lik",
            "KS Dist",
            "Quant MAE",
            "Mean Diff",
            "SD Diff",
            "Skew Diff",
            "Kurt Diff",
            "Accuracy",
            "AUC"
        ]
        
        # Human-Human row
        hh_row = ["Human-Human"]
        
        # 1. Log-Likelihood
        if rt_likelihood and rt_likelihood['human_human']['overall_mean_ll'] is not None:
            hh_row.append(f"{rt_likelihood['human_human']['overall_mean_ll']:.3f}")
        else:
            hh_row.append("N/A")
        
        # 2. KS Distance
        if rt_distribution and rt_distribution['human_human']['overall']['mean_ks_distance'] is not None:
            hh_row.append(f"{rt_distribution['human_human']['overall']['mean_ks_distance']:.3f}")
        else:
            hh_row.append("N/A")
        
        # 3-7. Distribution metrics (N/A for human-human as it's the baseline)
        for _ in range(5):
            hh_row.append("—")  # Baseline (comparing to self)
        
        # 8. Accuracy
        if metrics['human_accuracy'] is not None:
            hh_row.append(f"{metrics['human_accuracy']:.4f}")
        else:
            hh_row.append("N/A")
        
        # 9. AUC
        if metrics['human_auc'] is not None:
            hh_row.append(f"{metrics['human_auc']:.4f}")
        else:
            hh_row.append("N/A")
        
        
        # Model-Human row
        mh_row = ["Model-Human"]
        
        # 1. Log-Likelihood
        if rt_likelihood and rt_likelihood['model']['overall_mean_ll'] is not None:
            mh_row.append(f"{rt_likelihood['model']['overall_mean_ll']:.3f}")
        else:
            mh_row.append("N/A")
        
        # 2. KS Distance
        if rt_distribution and rt_distribution['model_human']['overall']['ks_distance']['mean'] is not None:
            mh_row.append(f"{rt_distribution['model_human']['overall']['ks_distance']['mean']:.3f}")
        else:
            mh_row.append("N/A")
        
        # 3. Quantile MAE
        if rt_distribution and rt_distribution['model_human']['overall']['quantile_mae']['mean'] is not None:
            mh_row.append(f"{rt_distribution['model_human']['overall']['quantile_mae']['mean']:.3f}s")
        else:
            mh_row.append("N/A")
        
        # 4. Mean RT Difference
        if rt_distribution and rt_distribution['model_human']['overall']['moment_differences']['mean']['mean'] is not None:
            mh_row.append(f"{rt_distribution['model_human']['overall']['moment_differences']['mean']['mean']:.3f}s")
        else:
            mh_row.append("N/A")
        
        # 5. SD Difference
        if rt_distribution and rt_distribution['model_human']['overall']['moment_differences']['std']['mean'] is not None:
            mh_row.append(f"{rt_distribution['model_human']['overall']['moment_differences']['std']['mean']:.3f}s")
        else:
            mh_row.append("N/A")
        
        # 6. Skewness Difference
        if rt_distribution and rt_distribution['model_human']['overall']['moment_differences']['skewness']['mean'] is not None:
            mh_row.append(f"{rt_distribution['model_human']['overall']['moment_differences']['skewness']['mean']:.3f}")
        else:
            mh_row.append("N/A")
        
        # 7. Kurtosis Difference
        if rt_distribution and rt_distribution['model_human']['overall']['moment_differences']['kurtosis']['mean'] is not None:
            mh_row.append(f"{rt_distribution['model_human']['overall']['moment_differences']['kurtosis']['mean']:.3f}")
        else:
            mh_row.append("N/A")
        
        # 8. Accuracy
        mh_row.append(f"{metrics['model_accuracy']:.4f}")
        
        # 9. AUC
        if metrics['model_auc'] is not None:
            mh_row.append(f"{metrics['model_auc']:.4f}")
        else:
            mh_row.append("N/A")
        
        # Print table
        # Calculate column widths
        col_widths = []
        for i, header in enumerate(headers):
            max_width = max(len(header), len(hh_row[i]), len(mh_row[i]))
            col_widths.append(max_width + 2)  # Add padding
        
        # Print header
        header_line = "│".join([h.center(w) for h, w in zip(headers, col_widths)])
        separator = "─" * sum(col_widths) + "─" * (len(headers) - 1)
        
        print("┌" + separator + "┐")
        print("│" + header_line + "│")
        print("├" + separator + "┤")
        
        # Print Human-Human row
        hh_line = "│".join([val.center(w) for val, w in zip(hh_row, col_widths)])
        print("│" + hh_line + "│")
        
        print("├" + separator + "┤")
        
        # Print Model-Human row
        mh_line = "│".join([val.center(w) for val, w in zip(mh_row, col_widths)])
        print("│" + mh_line + "│")
        
        print("└" + separator + "┘")
        
        # Print interpretation notes
        print("\nNotes:")
        print("  • Log-Lik: Higher is better (less negative)")
        print("  • KS Dist: Lower is better (0 = perfect match)")
        print("  • Quant MAE: Lower is better (quantile matching error)")
        print("  • Mean/SD/Skew/Kurt Diff: Closer to 0 is better")
        print("  • Accuracy & AUC: Higher is better (0-1 scale)")
        print("  • '—' indicates baseline reference (no self-comparison)")
        print("="*100)
