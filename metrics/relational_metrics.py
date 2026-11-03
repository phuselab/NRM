"""Model-vs-human curve metrics: correlation, noise ceilings and SATBench scores (§3)."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

from datasets.sat.loader import CATEGORIES

# Chance level of the 16-way SAT categorisation.
_CHANCE = 1.0 / 16.0   # 0.0625 = 6.25 %


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Concordance Correlation Coefficient — measures both correlation and agreement."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    mx, my  = x.mean(), y.mean()
    sx2, sy2 = x.var(ddof=0), y.var(ddof=0)
    sxy     = np.mean((x - mx) * (y - my))
    denom   = sx2 + sy2 + (mx - my) ** 2
    return float(2.0 * sxy / denom) if denom > 1e-15 else float("nan")


def safe_corr(x, y):
    """Return (pearson_r, spearman_r, ccc) with NaN on failure; needs >= 3 finite pairs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    try:
        pr = float(pearsonr(x, y)[0])
    except Exception:
        pr = float("nan")
    try:
        sr = float(spearmanr(x, y)[0])
    except Exception:
        sr = float("nan")
    return pr, sr, ccc(x, y)


def split_half_rt_nc(
    level_values: np.ndarray,
    h_rts: np.ndarray,
    levels: list[float],
    n_splits: int = 100,
    seed: int = 42,
) -> tuple[float, float]:
    """
    NC for r_RT via split-half across trials (ambiguity 2B).

    Randomly splits all test trials into two halves, computes per-level
    median RT for each half, then correlates the two vectors across levels.
    Repeated n_splits times; returns (mean_pearson_r, mean_spearman_r).
    """
    rng = np.random.default_rng(seed)
    n = len(h_rts)
    if n < 6:
        return float("nan"), float("nan")

    pearson_rs:  list[float] = []
    spearman_rs: list[float] = []

    for _ in range(n_splits):
        idx    = rng.permutation(n)
        half   = n // 2
        mask_a = np.zeros(n, dtype=bool)
        mask_a[idx[:half]] = True
        mask_b = ~mask_a

        med_a, med_b = [], []
        for level in levels:
            m       = level_values == level
            rts_a   = h_rts[m & mask_a]
            rts_b   = h_rts[m & mask_b]
            if len(rts_a) >= 1 and len(rts_b) >= 1:
                med_a.append(float(np.median(rts_a)))
                med_b.append(float(np.median(rts_b)))

        if len(med_a) >= 3:
            pr, sr, _ = safe_corr(med_a, med_b)
            if np.isfinite(pr):
                pearson_rs.append(pr)
            if np.isfinite(sr):
                spearman_rs.append(sr)

    return (
        float(np.mean(pearson_rs))  if pearson_rs  else float("nan"),
        float(np.mean(spearman_rs)) if spearman_rs else float("nan"),
    )


class SATCurvesCalculator:
    """
    Compute SAT curves (Speed-Accuracy Tradeoff) for humans and NRM.
 
    Grouping logic (faithful to the original SAT paper):
        1. Keep the SAT trials (time_limit > 0).
        2. For every (category, beep) pair:
               - group the trials by image_index
               - per image: mean RT and mean accuracy
                 (several subjects may have answered the same image)
        3. Mean and std/sem *across images* -> error bars on X (RT) and Y (acc).
 
    For the NRM model, instead of reusing the aggregate predictions
    already in `data`, n_rt_samples forward passes are run per image at
    every beep value, giving a real RT distribution per image (needed for
    meaningful error bars on X).
 
    All time quantities are in seconds.
    """
 
    # -------------------------------------------------------------------------
    # Internal helper - generic aggregation per deadline
    # -------------------------------------------------------------------------
 
    @staticmethod
    def _aggregate_over_images(image_indices, choices, true_labels,
                                rts, time_limits):
        """
        Given a subset of trials (already filtered by category and/or beep),
        groups by (beep, image_index) and returns statistics per beep.
 
        Returns:
            dict  beep_s → {rt_mean, rt_std, rt_sem, acc_mean, acc_std, acc_sem,
                             n_images, n_trials,
                             per_image_rt, per_image_acc}
        """
        import numpy as np
        from collections import defaultdict
 
        # structure: beep → image_id → {rts: [], correct: []}
        nested = defaultdict(lambda: defaultdict(lambda: {'rts': [], 'correct': []}))
        for i in range(len(image_indices)):
            b   = float(time_limits[i])
            img = int(image_indices[i])
            nested[b][img]['rts'].append(float(rts[i]))
            nested[b][img]['correct'].append(
                int(choices[i] == true_labels[i])
            )
 
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
                'rt_std':        float(per_image_rt.std())         if n > 1 else 0.0,
                'rt_sem':        float(per_image_rt.std() / n**0.5) if n > 1 else 0.0,
                'acc_mean':      float(per_image_acc.mean()),
                'acc_std':       float(per_image_acc.std())         if n > 1 else 0.0,
                'acc_sem':       float(per_image_acc.std() / n**0.5) if n > 1 else 0.0,
                'n_images':      n,
                'n_trials':      n_trials_total,
                # Raw per-image arrays (useful for scatter plots or debugging)
                'per_image_rt':  per_image_rt.tolist(),
                'per_image_acc': per_image_acc.tolist(),
            }
 
        return results
 
    # -------------------------------------------------------------------------
    # Humans — overall
    # -------------------------------------------------------------------------
 
    def calculate_human_sat_curves(self, image_indices, true_labels,
                                   human_choices, true_rts, time_limits):
        """
        Human SAT curves, aggregated over every category.
 
        Returns:
            {
              'per_beep': {beep_s: {...}},
              'beeps_s':  [sorted list],
              'n_beeps':  int
            }
            or None if there are no SAT trials.
        """
        import numpy as np
        mask = time_limits > 0
        if mask.sum() == 0:
            print("  [SAT] No SAT trials (every time_limit <= 0)")
            return None
 
        per_beep = self._aggregate_over_images(
            image_indices=image_indices[mask],
            choices=human_choices[mask],
            true_labels=true_labels[mask],
            rts=true_rts[mask],
            time_limits=time_limits[mask],
        )
        return {
            'per_beep': per_beep,
            'beeps_s':  sorted(per_beep.keys()),
            'n_beeps':  len(per_beep),
        }
 
    # -------------------------------------------------------------------------
    # Humans — per category
    # -------------------------------------------------------------------------
 
    def calculate_human_sat_curves_by_category(self, image_indices, true_labels,
                                               human_choices, true_rts,
                                               time_limits, category_names=None):
        """
        Human SAT curves, split by ground-truth category (0..n_choices-1).
 
        Returns:
            dict  category_idx → {per_beep, beeps_s, n_beeps, category_name}
        """
        import numpy as np
        mask = time_limits > 0
        if mask.sum() == 0:
            return None
 
        idx   = image_indices[mask]
        labs  = true_labels[mask]
        hch   = human_choices[mask]
        rts   = true_rts[mask]
        tls   = time_limits[mask]
 
        results = {}
        for cat in np.unique(labs):
            cat_mask = labs == cat
            per_beep = self._aggregate_over_images(
                image_indices=idx[cat_mask],
                choices=hch[cat_mask],
                true_labels=labs[cat_mask],
                rts=rts[cat_mask],
                time_limits=tls[cat_mask],
            )
            cat_name = (category_names[cat]
                        if category_names is not None and cat < len(category_names)
                        else str(cat))
            results[int(cat)] = {
                'per_beep':     per_beep,
                'beeps_s':      sorted(per_beep.keys()),
                'n_beeps':      len(per_beep),
                'category_name': cat_name,
            }
 
        return results
 
    # -------------------------------------------------------------------------
    # Model — per category (new forward pass)
    # -------------------------------------------------------------------------
 
    def calculate_model_sat_curves_by_category(self, model, device, data,
                                               n_rt_samples=50, t=0.993,
                                               category_names=None):
        """
        NRM SAT curves separated by category.
 
        For each (category, deadline, image_index) it runs n_rt_samples forward passes
        with that beep as covariate, giving:
            - the RT distribution per image → mean and std RT per image
            - the choice distribution per image → accuracy per image
 
        Then mean/std across images -> error bars on X and Y.
 
        It does not reuse pred_rts from collect_predictions (1 aggregated sample,
        not suitable for error bars on X), but uses the embeddings already in `data`.
 
        Args:
            model:         NRM
            device:        torch.device
            data:          output of collect_predictions
                           (must contain 'embeddings', 'true_labels',
                            'image_indices', 'time_limits')
            n_rt_samples:  samples per (image, beep)
            t:             soft threshold for simulate_nDDM
            category_names: list of strings (optional)
 
        Returns:
            dict  category_idx → {per_beep, beeps_s, n_beeps, category_name}
            or None if there are no SAT trials.
        """
        import torch
        import numpy as np
        from collections import defaultdict
 
        time_limits = data['time_limits']
        sat_mask    = time_limits > 0
        if sat_mask.sum() == 0:
            print("  [SAT] No SAT trials for the model.")
            return None
 
        tls_sat  = time_limits[sat_mask]
        labs_sat = data['true_labels'][sat_mask]
        imgs_sat = data['image_indices'][sat_mask]
        embs_sat = data['embeddings'][sat_mask]   # np.array [N_sat, emb_dim]
 
        unique_cats  = np.unique(labs_sat)
        unique_beeps = np.unique(tls_sat)
 
        # Results: cat → beep → image_id → {rts: [], correct: []}
        # defaultdict for convenience
        cat_beep_img = {
            int(cat): {float(beep): defaultdict(lambda: {'rts': [], 'correct': []})
                       for beep in unique_beeps}
            for cat in unique_cats
        }
 
        total_cats  = len(unique_cats)
        total_beeps = len(unique_beeps)
        print(f"  [SAT model] {total_cats} categories x {total_beeps} deadlines")
 
        model.eval()
        with torch.no_grad():
            for ci, cat in enumerate(unique_cats):
                cat_mask_local = labs_sat == cat
                imgs_cat = imgs_sat[cat_mask_local]
                embs_cat = embs_sat[cat_mask_local]
                labs_cat = labs_sat[cat_mask_local]
 
                # Unique images in this category
                unique_imgs_cat = np.unique(imgs_cat)
                n_img = len(unique_imgs_cat)
 
                print(f"    cat {int(cat):2d} ({category_names[cat] if category_names else cat}): "
                      f"{n_img} images", end='')
 
                for bi, beep in enumerate(unique_beeps):
                    print(f" b{bi+1}", end='', flush=True)
 
                    # For every unique image in this category,
                    # take its embedding (the first one found)
                    img_embs = []
                    img_ids  = []
                    img_labs = []
                    for img_id in unique_imgs_cat:
                        img_mask = imgs_cat == img_id
                        img_embs.append(embs_cat[img_mask][0])   # first embedding
                        img_ids.append(img_id)
                        img_labs.append(labs_cat[img_mask][0])
 
                    n_uniq = len(img_ids)
                    embs_t = torch.tensor(
                        np.stack(img_embs), dtype=torch.float32
                    ).to(device)                              # [n_uniq, emb_dim]
                    labs_t = np.array(img_labs)
 
                    # Expand by n_rt_samples
                    embs_exp = embs_t.repeat_interleave(n_rt_samples, dim=0)
                    beep_t   = torch.full(
                        (n_uniq * n_rt_samples,), float(beep),
                        dtype=torch.float32, device=device
                    )
                    cov_exp = {
                        "img_embedding": embs_exp,
                        "time_limit":    beep_t,
                    }
 
                    _, _, pred_ch_all, pred_rt_all = model.simulate_nDDM(
                        batch_size=n_uniq * n_rt_samples,
                        covariates=cov_exp,
                        soft=True,
                        n_reps=1,
                        t=t,
                    )
 
                    # Reshape → [n_uniq, n_rt_samples]
                    pred_ch = pred_ch_all.view(n_uniq, n_rt_samples).cpu().numpy()
                    pred_rt = pred_rt_all.view(n_uniq, n_rt_samples).cpu().numpy()
 
                    beep_s = float(beep)
                    for obs_i, img_id in enumerate(img_ids):
                        true_lb = int(labs_t[obs_i])
                        for s in range(n_rt_samples):
                            cat_beep_img[int(cat)][beep_s][int(img_id)]['rts'].append(
                                float(pred_rt[obs_i, s])
                            )
                            cat_beep_img[int(cat)][beep_s][int(img_id)]['correct'].append(
                                int(pred_ch[obs_i, s] == true_lb)
                            )
 
                print()  # newline after this category's deadlines
 
        # Aggregate: per (cat, deadline), per image, mean, then std across images
        results = {}
        for cat in unique_cats:
            cat_int = int(cat)
            cat_name = (category_names[cat_int]
                        if category_names is not None and cat_int < len(category_names)
                        else str(cat_int))
            per_beep = {}
            for beep_s in sorted(cat_beep_img[cat_int].keys()):
                img_data = cat_beep_img[cat_int][beep_s]
 
                per_image_rt  = []
                per_image_acc = []
                for img_id, vals in img_data.items():
                    per_image_rt.append(np.mean(vals['rts']))
                    per_image_acc.append(np.mean(vals['correct']))
 
                per_image_rt  = np.array(per_image_rt,  dtype=float)
                per_image_acc = np.array(per_image_acc, dtype=float)
                n = len(per_image_rt)
 
                per_beep[beep_s] = {
                    'beep_s':        beep_s,
                    'rt_mean':       float(per_image_rt.mean()),
                    'rt_std':        float(per_image_rt.std())          if n > 1 else 0.0,
                    'rt_sem':        float(per_image_rt.std() / n**0.5) if n > 1 else 0.0,
                    'acc_mean':      float(per_image_acc.mean()),
                    'acc_std':       float(per_image_acc.std())          if n > 1 else 0.0,
                    'acc_sem':       float(per_image_acc.std() / n**0.5) if n > 1 else 0.0,
                    'n_images':      n,
                    'n_samples_total': int(sum(
                        len(v['rts']) for v in img_data.values()
                    )),
                    'per_image_rt':  per_image_rt.tolist(),
                    'per_image_acc': per_image_acc.tolist(),
                }
 
            results[cat_int] = {
                'per_beep':      per_beep,
                'beeps_s':       sorted(per_beep.keys()),
                'n_beeps':       len(per_beep),
                'category_name': cat_name,
            }
 
        return results
 
    @staticmethod
    def aggregate_model_curves_overall(model_curves_by_cat):
        """
        Aggregate the per-category SAT curves into one overall curve for the model.
        Uses the mean weighted by image count.
 
        Args:
            model_curves_by_cat: output of calculate_model_sat_curves_by_category
 
        Returns:
            dict with the same structure as calculate_human_sat_curves
        """
        import numpy as np
        if model_curves_by_cat is None:
            return None
 
        # Collect every unique beep
        all_beeps = set()
        for cat_data in model_curves_by_cat.values():
            all_beeps.update(cat_data['beeps_s'])
 
        per_beep = {}
        for beep_s in sorted(all_beeps):
            all_rt  = []
            all_acc = []
            for cat_data in model_curves_by_cat.values():
                if beep_s in cat_data['per_beep']:
                    per_beep_cat = cat_data['per_beep'][beep_s]
                    all_rt.extend(per_beep_cat['per_image_rt'])
                    all_acc.extend(per_beep_cat['per_image_acc'])
 
            if len(all_rt) == 0:
                continue
 
            arr_rt  = np.array(all_rt,  dtype=float)
            arr_acc = np.array(all_acc, dtype=float)
            n = len(arr_rt)
 
            per_beep[beep_s] = {
                'beep_s':   beep_s,
                'rt_mean':  float(arr_rt.mean()),
                'rt_std':   float(arr_rt.std())          if n > 1 else 0.0,
                'rt_sem':   float(arr_rt.std() / n**0.5) if n > 1 else 0.0,
                'acc_mean': float(arr_acc.mean()),
                'acc_std':  float(arr_acc.std())          if n > 1 else 0.0,
                'acc_sem':  float(arr_acc.std() / n**0.5) if n > 1 else 0.0,
                'n_images': n,
                'per_image_rt':  arr_rt.tolist(),
                'per_image_acc': arr_acc.tolist(),
            }
 
        return {
            'per_beep': per_beep,
            'beeps_s':  sorted(per_beep.keys()),
            'n_beeps':  len(per_beep),
        }


def _weibull_cdf(t: np.ndarray, lam: float, k: float) -> np.ndarray:
    """Cumulative Weibull: w(t) = 1 - exp(-(t/λ)^k)"""
    return 1.0 - np.exp(-((t / lam) ** k))


def _fit_weibull(
        times: np.ndarray,
        accs: np.ndarray,
) -> tuple[float, float] | None:
    """
    Fit a cumulative Weibull to the points (times, accs) by non-linear least squares.
    Returns (lambda, k), or None if the fit fails.
    """
    if len(times) < 3:
        return None
    try:
        accs_c = np.clip(accs, 1e-6, 1.0 - 1e-6)
        p0 = [float(np.median(times)), 1.5]
        popt, _ = curve_fit(
            _weibull_cdf, times, accs_c,
            p0=p0,
            bounds=([1e-6, 0.05], [1e6, 50.0]),
            maxfev=10000,
            # With five points on a near-step SAT curve the Weibull exponent is
            # barely identified, so loose tolerances let the solver's stopping
            # rule reach the third decimal of K. Tight tolerances make the fit
            # depend on the data rather than on the installed scipy.
            xtol=1e-12, ftol=1e-12, gtol=1e-12,
        )
        return tuple(popt)
    except Exception:
        return None


def _find_curvature(points: np.ndarray) -> tuple[float, float]:
    """
    Mean curvature of a 2D curve described by points (N×2).
    Same implementation as Appendix D.2 of the SATBench paper.

    κ = |x''·y' − x'·y''| / ((x')² + (y')²)^(3/2)
    """
    x_t  = np.gradient(points[:, 0])
    y_t  = np.gradient(points[:, 1])
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)
    denom = (x_t ** 2 + y_t ** 2) ** 1.5
    denom = np.where(denom < 1e-12, 1e-12, denom)
    curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / denom
    n = len(curvature_val)
    mean_k = float(np.mean(curvature_val))
    sem_k  = float(np.std(curvature_val) / np.sqrt(n)) if n > 1 else 0.0
    return mean_k, sem_k


def compute_weibull_steepness(
        times: np.ndarray,
        accs: np.ndarray,
        n_sample: int = 20,
) -> tuple[float, float]:
    """
    1. Fit a cumulative Weibull to the data (times, accs).
    2. Sample n_sample equally spaced points over [min(t), max(t)].
    3. Compute the mean curvature κ.

    Returns (mean_kappa, sem_kappa).
    """
    if len(times) < 3 or len(set(times)) < 3:
        return float('nan'), float('nan')
    params = _fit_weibull(times, accs)
    if params is None:
        return float('nan'), float('nan')
    lam, k = params
    t_min, t_max = float(np.min(times)), float(np.max(times))
    if t_min >= t_max:
        return float('nan'), float('nan')
    t_sample   = np.linspace(t_min, t_max, n_sample)
    acc_sample = _weibull_cdf(t_sample, lam, k)
    points = np.column_stack([t_sample, acc_sample])
    return _find_curvature(points)


def _compute_per_observer_curves(
        data: dict,
        ordered_levels: list[str],
        n_choices: int = 16,
) -> dict | None:
    """
    Compute accuracy curves per observer (srid).

    Returns
    -------
    {
      srid -> {
        level -> {
          'overall': {beep_s -> float},          # accuracy overall
          'by_cat':  {cat_idx -> {beep_s -> float}},
        }
      }
    }
    Returns None when 'srids' is not available in data.
    """
    srids = data.get('srids')
    if srids is None or len(srids) == 0:
        return None

    levels_arr   = np.array(data['levels'])
    time_lims    = data['time_limits']
    human_ch     = data['human_choices']
    true_lbl     = data['true_labels']
    sat_mask     = time_lims > 0
    srids_arr    = np.array(srids)

    unique_beeps = sorted({float(t) for t, m in zip(time_lims, sat_mask) if m})
    unique_srids = sorted({s for s, m in zip(srids, sat_mask) if m})

    result: dict[str, dict] = {}
    for sid in unique_srids:
        sid_mask = sat_mask & (srids_arr == sid)
        result[sid] = {}
        for level in ordered_levels:
            overall: dict[float, float] = {}
            by_cat:  dict[int, dict[float, float]] = {}

            lv_sid_mask = sid_mask & (levels_arr == level)
            if lv_sid_mask.sum() == 0:
                result[sid][level] = {'overall': {}, 'by_cat': {}}
                continue

            for beep in unique_beeps:
                # Use isclose for float comparison
                beep_mask = lv_sid_mask & (np.abs(time_lims - beep) < 1e-6)
                if beep_mask.sum() == 0:
                    continue
                correct = (human_ch[beep_mask] == true_lbl[beep_mask]).astype(float)
                overall[beep] = float(correct.mean())

            for cat in range(n_choices):
                cat_lv_sid_mask = lv_sid_mask & (true_lbl == cat)
                if cat_lv_sid_mask.sum() == 0:
                    continue
                cat_dict: dict[float, float] = {}
                for beep in unique_beeps:
                    beep_cat_mask = cat_lv_sid_mask & (np.abs(time_lims - beep) < 1e-6)
                    if beep_cat_mask.sum() == 0:
                        continue
                    c_corr = (human_ch[beep_cat_mask] == true_lbl[beep_cat_mask]).astype(float)
                    cat_dict[beep] = float(c_corr.mean())
                if cat_dict:
                    by_cat[cat] = cat_dict

            result[sid][level] = {'overall': overall, 'by_cat': by_cat}

    return result


def compute_satbench_metrics(
        human_curves: dict,
        model_curves: dict,
        per_obs_curves: dict | None,
        n_choices: int = 16,
) -> dict:
    """
    Compute SATbench metrics:
      - eRMSE  : per observer (distribution for boxplot)
      - Spearman ρ category-wise : per observer (distribution for boxplot)
      - Steepness κ (Weibull) : per perturbation level (line plot)

    Returns
    -------
    dict with the fields documented in the other modules.
    """
    scenario     = human_curves.get('scenario', 'unknown')
    ordered_lvls = human_curves.get('ordered_levels', [])
    display      = human_curves.get('display_labels', {})

    # ── Mean-accuracy tables (model and human mean) ──────────────────────────
    # Indexed as [level][beep] for overall and [level][cat][beep] for by_cat

    def _beep(per_beep: dict, b):
        """``per_beep`` entry for beep time ``b``.

        JSON has no float keys, so a payload that has been through
        ``save_curves_to_json`` is keyed by ``'0.5'`` while ``beeps_s`` still
        holds ``0.5``. Accept either, so these curves can be recomputed from a
        saved file and not only from memory.
        """
        if b in per_beep:
            return per_beep[b]
        return per_beep[str(b)]

    def _cat(by_level: dict, level, cat):
        """``by_category[level][cat]`` with the same str/int key tolerance.

        Returning None on a key-type mismatch would drop every category
        silently, which is worse than raising, so both spellings are tried.
        """
        entry = by_level.get(level) or by_level.get(str(level)) or {}
        return entry.get(cat, entry.get(str(cat)))

    model_acc_overall: dict[str, dict[float, float]] = {}
    model_acc_by_cat:  dict[str, dict[int, dict[float, float]]] = {}

    human_acc_overall: dict[str, dict[float, float]] = {}
    human_acc_by_cat:  dict[str, dict[int, dict[float, float]]] = {}

    for level in ordered_lvls:
        model_acc_overall[level] = {}
        m_ov = model_curves['overall'].get(level)
        if m_ov:
            for b in m_ov['beeps_s']:
                model_acc_overall[level][b] = _beep(m_ov['per_beep'], b)['acc_mean']

        model_acc_by_cat[level] = {}
        for cat in range(n_choices):
            model_acc_by_cat[level][cat] = {}
            m_cat = _cat(model_curves['by_category'], level, cat)
            if m_cat:
                for b in m_cat['beeps_s']:
                    model_acc_by_cat[level][cat][b] = _beep(m_cat['per_beep'], b)['acc_mean']

        human_acc_overall[level] = {}
        h_ov = human_curves['overall'].get(level)
        if h_ov:
            for b in h_ov['beeps_s']:
                human_acc_overall[level][b] = _beep(h_ov['per_beep'], b)['acc_mean']

        human_acc_by_cat[level] = {}
        for cat in range(n_choices):
            human_acc_by_cat[level][cat] = {}
            h_cat = _cat(human_curves['by_category'], level, cat)
            if h_cat:
                for b in h_cat['beeps_s']:
                    human_acc_by_cat[level][cat][b] = _beep(h_cat['per_beep'], b)['acc_mean']

    # ── 1. eRMSE per observer ─────────────────────────────────────────────
    # Formula: eRMSE = (1/Np) * sum_p sqrt((1/Nt) * sum_t (c1[p,t]-c2[p,t])^2)

    model_ermse_per_obs: list[float] = []
    human_ermse_per_obs: list[float] = []   # within-human: obs vs mean-human

    # Per-level distributions (for per-level table)
    ermse_model_by_level_per_obs: dict[str, list[float]] = {lvl: [] for lvl in ordered_lvls}
    ermse_human_by_level_per_obs: dict[str, list[float]] = {lvl: [] for lvl in ordered_lvls}

    if per_obs_curves is not None:
        for sid, sid_data in per_obs_curves.items():
            rmse_model_levels: list[float] = []
            rmse_human_levels: list[float] = []

            for level in ordered_lvls:
                obs_overall = sid_data.get(level, {}).get('overall', {})
                m_acc = model_acc_overall.get(level, {})
                h_avg = human_acc_overall.get(level, {})

                common_m = sorted(set(obs_overall.keys()) & set(m_acc.keys()))
                common_h = sorted(set(obs_overall.keys()) & set(h_avg.keys()))

                if len(common_m) >= 2:
                    obs_v = np.array([obs_overall[b] for b in common_m]) * 100.0
                    m_v   = np.array([m_acc[b]        for b in common_m]) * 100.0
                    rv = float(np.sqrt(np.mean((obs_v - m_v) ** 2)))
                    rmse_model_levels.append(rv)
                    ermse_model_by_level_per_obs[level].append(rv)

                if len(common_h) >= 2:
                    obs_v = np.array([obs_overall[b] for b in common_h]) * 100.0
                    h_v   = np.array([h_avg[b]        for b in common_h]) * 100.0
                    rh = float(np.sqrt(np.mean((obs_v - h_v) ** 2)))
                    rmse_human_levels.append(rh)
                    ermse_human_by_level_per_obs[level].append(rh)

            if rmse_model_levels:
                model_ermse_per_obs.append(float(np.mean(rmse_model_levels)))
            if rmse_human_levels:
                human_ermse_per_obs.append(float(np.mean(rmse_human_levels)))

    # Scalar eRMSE per level (human-mean vs model-mean, for table)
    ermse_by_level: dict[str, float] = {}
    for level in ordered_lvls:
        h_ov_acc = human_acc_overall.get(level, {})
        m_ov_acc = model_acc_overall.get(level, {})
        common_b = sorted(set(h_ov_acc.keys()) & set(m_ov_acc.keys()))
        if len(common_b) < 2:
            ermse_by_level[level] = float('nan')
        else:
            h_v = np.array([h_ov_acc[b] for b in common_b]) * 100.0
            m_v = np.array([m_ov_acc[b] for b in common_b]) * 100.0
            ermse_by_level[level] = float(np.sqrt(np.mean((h_v - m_v) ** 2)))

    # ── 2. Category-wise Spearman per observer ─────────────────────────────
    # One vector per (level, cat, deadline): flattened over all three axes

    # Build the (level, cat, beep) keys shared by the model and the human mean
    cat_beep_keys: list[tuple[str, int, float]] = []
    model_flat:    list[float] = []
    human_avg_flat: list[float] = []

    for level in ordered_lvls:
        for cat in range(n_choices):
            m_acc_cat = model_acc_by_cat.get(level, {}).get(cat, {})
            h_acc_cat = human_acc_by_cat.get(level, {}).get(cat, {})
            common_b  = sorted(set(m_acc_cat.keys()) & set(h_acc_cat.keys()))
            for b in common_b:
                cat_beep_keys.append((level, cat, b))
                model_flat.append(m_acc_cat[b])
                human_avg_flat.append(h_acc_cat[b])

    model_flat_arr     = np.array(model_flat)
    human_avg_flat_arr = np.array(human_avg_flat)

    model_spearman_per_obs: list[float] = []
    human_spearman_per_obs: list[float] = []
    spearman_by_cat_per_obs: dict[int, list[float]] = {c: [] for c in range(n_choices)}
    spearman_by_cat_human:   dict[int, list[float]] = {c: [] for c in range(n_choices)}
    # Per-level Spearman: per level, list of ρ (one per observer)
    spearman_by_level_per_obs: dict[str, list[float]] = {lvl: [] for lvl in ordered_lvls}
    # Per-level human-within Spearman
    spearman_human_by_level_per_obs: dict[str, list[float]] = {lvl: [] for lvl in ordered_lvls}
    _n_sp_nan = 0   # count observers with NaN Spearman (e.g. constant model output)

    def _safe_spearman(a, b) -> float | None:
        """Spearman with NaN and constant-input handling; None when undefined."""
        a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if len(a_arr) < 3:
            return None
        if np.std(a_arr) < 1e-9 or np.std(b_arr) < 1e-9:
            return None   # constant input → undefined correlation
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                r, _ = spearmanr(a_arr, b_arr)
                return None if np.isnan(r) else float(r)
            except Exception:
                return None

    if per_obs_curves is not None and len(cat_beep_keys) >= 5:
        for sid, sid_data in per_obs_curves.items():
            # Observer vector (only where observer has data)
            obs_vals_all: list[float] = []
            mod_vals_all: list[float] = []
            avg_vals_all: list[float] = []

            for i, (level, cat, beep) in enumerate(cat_beep_keys):
                acc_o = (sid_data
                         .get(level, {})
                         .get('by_cat', {})
                         .get(cat, {})
                         .get(beep))
                if acc_o is not None:
                    obs_vals_all.append(acc_o)
                    mod_vals_all.append(float(model_flat_arr[i]))
                    avg_vals_all.append(float(human_avg_flat_arr[i]))

            if len(obs_vals_all) >= 10:
                r_m = _safe_spearman(obs_vals_all, mod_vals_all)
                if r_m is not None:
                    model_spearman_per_obs.append(r_m)
                else:
                    _n_sp_nan += 1
                r_h = _safe_spearman(obs_vals_all, avg_vals_all)
                if r_h is not None:
                    human_spearman_per_obs.append(r_h)

            # Per-level: Spearman on (cat x beep) for a single level
            for lvl in ordered_lvls:
                obs_lv, mod_lv, avg_lv = [], [], []
                for i, (lv, cat, beep) in enumerate(cat_beep_keys):
                    if lv != lvl:
                        continue
                    acc_o = (sid_data
                             .get(lvl, {})
                             .get('by_cat', {})
                             .get(cat, {})
                             .get(beep))
                    if acc_o is not None:
                        obs_lv.append(acc_o)
                        mod_lv.append(float(model_flat_arr[i]))
                        avg_lv.append(float(human_avg_flat_arr[i]))
                r_lv = _safe_spearman(obs_lv, mod_lv)
                if r_lv is not None:
                    spearman_by_level_per_obs[lvl].append(r_lv)
                r_lv_h = _safe_spearman(obs_lv, avg_lv)
                if r_lv_h is not None:
                    spearman_human_by_level_per_obs[lvl].append(r_lv_h)

            # Per-category: Spearman on (beep x level) for a single category
            for cat in range(n_choices):
                obs_c, mod_c, avg_c = [], [], []
                for i, (level, c, beep) in enumerate(cat_beep_keys):
                    if c != cat:
                        continue
                    acc_o = (sid_data
                             .get(level, {})
                             .get('by_cat', {})
                             .get(cat, {})
                             .get(beep))
                    if acc_o is not None:
                        obs_c.append(acc_o)
                        mod_c.append(float(model_flat_arr[i]))
                        avg_c.append(float(human_avg_flat_arr[i]))
                r_cat_m = _safe_spearman(obs_c, mod_c)
                if r_cat_m is not None:
                    spearman_by_cat_per_obs[cat].append(r_cat_m)
                r_cat_h = _safe_spearman(obs_c, avg_c)
                if r_cat_h is not None:
                    spearman_by_cat_human[cat].append(r_cat_h)

    if _n_sp_nan > 0:
        print(f"  [WARN] Spearman NaN for {_n_sp_nan} observer(s) "
              f"(constant model output or zero variance) - excluded from the boxplot.")

    # Scalar Spearman per level (mean over observers)
    spearman_by_level: dict[str, float] = {}
    for lvl in ordered_lvls:
        vals = spearman_by_level_per_obs[lvl]
        spearman_by_level[lvl] = float(np.nanmean(vals)) if vals else float('nan')

    # Medians per level (for global LaTeX table)
    ermse_model_by_level_median: dict[str, float] = {
        lv: float(np.nanmedian(v)) if v else float('nan')
        for lv, v in ermse_model_by_level_per_obs.items()
    }
    ermse_human_by_level_median: dict[str, float] = {
        lv: float(np.nanmedian(v)) if v else float('nan')
        for lv, v in ermse_human_by_level_per_obs.items()
    }
    spearman_human_by_level: dict[str, float] = {
        lv: float(np.nanmedian(v)) if v else float('nan')
        for lv, v in spearman_human_by_level_per_obs.items()
    }
    spearman_model_by_level_median: dict[str, float] = {
        lv: float(np.nanmedian(v)) if v else float('nan')
        for lv, v in spearman_by_level_per_obs.items()
    }

    # Median Spearman per category
    spearman_cat_median:       dict[int, float] = {}
    spearman_cat_median_human: dict[int, float] = {}
    for cat in range(n_choices):
        vals = spearman_by_cat_per_obs[cat]
        spearman_cat_median[cat] = float(np.nanmedian(vals)) if vals else float('nan')
        vals_h = spearman_by_cat_human[cat]
        spearman_cat_median_human[cat] = float(np.nanmedian(vals_h)) if vals_h else float('nan')

    # ── 3. Steepness kappa (Weibull + curvature) per level ─────────────────────
    steepness: dict[str, dict[str, tuple[float, float]]] = {}
    for level in ordered_lvls:
        h_ov = human_curves['overall'].get(level)
        m_ov = model_curves['overall'].get(level)

        st_h = (float('nan'), float('nan'))
        st_m = (float('nan'), float('nan'))

        if h_ov and len(h_ov['beeps_s']) >= 3:
            beeps = np.array(h_ov['beeps_s'])
            accs  = np.array([_beep(h_ov['per_beep'], b)['acc_mean'] for b in h_ov['beeps_s']])
            # Include chance point at t=0
            t_all   = np.concatenate([[0.0], beeps])
            acc_all = np.concatenate([[_CHANCE],  accs])
            st_h = compute_weibull_steepness(t_all, acc_all)

        if m_ov and len(m_ov['beeps_s']) >= 3:
            beeps = np.array(m_ov['beeps_s'])
            accs  = np.array([_beep(m_ov['per_beep'], b)['acc_mean'] for b in m_ov['beeps_s']])
            t_all   = np.concatenate([[0.0], beeps])
            acc_all = np.concatenate([[_CHANCE],  accs])
            st_m = compute_weibull_steepness(t_all, acc_all)

        steepness[level] = {'human': st_h, 'model': st_m}

    # ── 4. Per-level scalars (for the LaTeX table) ─────────────────────────
    overall_scalars: dict[str, dict] = {}
    for level in ordered_lvls:
        st = steepness.get(level, {})
        sh = st.get('human', (float('nan'), float('nan')))[0]
        sm = st.get('model', (float('nan'), float('nan')))[0]
        overall_scalars[level] = {
            'rmse':             ermse_by_level.get(level, float('nan')),
            'spearman_r':       spearman_by_level.get(level, float('nan')),
            'steepness_human':  sh,
            'steepness_model':  sm,
            'steepness_diff':   sm - sh if not (np.isnan(sm) or np.isnan(sh)) else float('nan'),
        }

    valid_rmse = [v['rmse'] for v in overall_scalars.values()
                  if v and not np.isnan(v.get('rmse', float('nan')))]
    valid_sp = [v for v in spearman_by_level.values() if not np.isnan(v)]
    mean_over_levels = {
        'rmse':            float(np.nanmean(valid_rmse)) if valid_rmse else float('nan'),
        'spearman_r':      float(np.nanmean(valid_sp)) if valid_sp else float('nan'),
        'steepness_human': float(np.nanmean([st['human'][0] for st in steepness.values()
                                             if not np.isnan(st['human'][0])])),
        'steepness_model': float(np.nanmean([st['model'][0] for st in steepness.values()
                                             if not np.isnan(st['model'][0])])),
    }

    return {
        'scenario':              scenario,
        'ordered_levels':        ordered_lvls,
        'display_labels':        display,
        # eRMSE
        'ermse': {
            'model_per_obs':            model_ermse_per_obs,
            'human_per_obs':            human_ermse_per_obs,
            'model_by_level_median':    ermse_model_by_level_median,
            'human_by_level_median':    ermse_human_by_level_median,
        },
        'ermse_by_level':        ermse_by_level,
        # Spearman
        'spearman': {
            'model_per_obs':             model_spearman_per_obs,
            'human_per_obs':             human_spearman_per_obs,
            'by_cat':                    spearman_by_cat_per_obs,
            'by_level_per_obs':          spearman_by_level_per_obs,
            'by_level':                  spearman_by_level,
            'model_by_level_median':     spearman_model_by_level_median,
            'human_by_level':            spearman_human_by_level,
        },
        'spearman_human': {
            'by_cat': spearman_by_cat_human,
        },
        'spearman_cat_median':        spearman_cat_median,
        'spearman_cat_median_human':  spearman_cat_median_human,
        # Steepness
        'steepness':             steepness,
        # Per-level scalars
        'overall':               overall_scalars,
        'mean_over_levels':      mean_over_levels,
        # Per-category summary (median Spearman over levels)
        'category_mean_over_levels': {
            cat: {
                'name':      CATEGORIES[cat] if cat < len(CATEGORIES) else str(cat),
                'rmse':      float('nan'),
                'pearson_r': spearman_cat_median.get(cat, float('nan')),
            }
            for cat in range(n_choices)
        },
    }


def compute_per_obs_bootstrap_ci(per_obs, B: int = 1000, seed: int = 42) -> tuple:
    """
    Non-parametric bootstrap 95% CI by resampling the per-observer array.

    At each of B iterations, resample len(per_obs) values with replacement
    and compute the median. CI = [2.5th, 97.5th] percentile of bootstrap medians.

    Returns (ci_lo, ci_hi) or (nan, nan) if per_obs is degenerate.
    """
    try:
        arr = np.array([float(x) for x in per_obs if x is not None], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 3:
            return (float('nan'), float('nan'))
        rng = np.random.default_rng(seed)
        n = len(arr)
        boot_meds = np.array([
            np.median(arr[rng.choice(n, size=n, replace=True)])
            for _ in range(B)
        ])
        return (float(np.percentile(boot_meds, 2.5)),
                float(np.percentile(boot_meds, 97.5)))
    except Exception:
        return (float('nan'), float('nan'))


def compute_spearman_pvalue(model_cat_medians: dict, human_cat_medians: dict) -> tuple:
    """
    Exact two-tailed Spearman p-value using per-category median vectors.

    Constructs K-length vectors from shared category keys, then calls
    scipy.stats.spearmanr. Returns (r, p) or (nan, nan) on failure.
    """
    try:
        cats = sorted(set(model_cat_medians) & set(human_cat_medians),
                      key=lambda k: int(k))
        x = np.array([float(model_cat_medians[c])
                      for c in cats if model_cat_medians[c] is not None], float)
        y = np.array([float(human_cat_medians[c])
                      for c in cats if human_cat_medians[c] is not None], float)
        # align lengths in case of missing values
        cats_ok = [c for c in cats
                   if model_cat_medians.get(c) is not None
                   and human_cat_medians.get(c) is not None]
        x = np.array([float(model_cat_medians[c]) for c in cats_ok], float)
        y = np.array([float(human_cat_medians[c]) for c in cats_ok], float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            return float('nan'), float('nan')
        r, p = spearmanr(x[ok], y[ok])
        return float(r), float(p)
    except Exception:
        return float('nan'), float('nan')

