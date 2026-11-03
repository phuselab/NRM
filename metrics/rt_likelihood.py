"""KDE negative log-likelihood of human RTs under model RT distributions (§3)."""

from __future__ import annotations

import numpy as np
import warnings
from collections import defaultdict
from scipy.stats import norm, lognorm, gamma, invgauss, gaussian_kde, ks_2samp


def kde_nll(eval_rts: np.ndarray, fit_rts: np.ndarray,
            bw_method: str = "scott", eps: float = 1e-10, *,
            rt_range: tuple[float, float]) -> float:
    """
    Compute NLL of *eval_rts* under a KDE fitted to *fit_rts*.

    Both arrays are clipped to *rt_range* before fitting to avoid
    boundary issues and unrealistic extremes.
    """
    fit_rts  = np.clip(fit_rts,  *rt_range)
    eval_rts = np.clip(eval_rts, *rt_range)

    if len(fit_rts) < 3:
        return float("nan")
    try:
        kde    = gaussian_kde(fit_rts, bw_method=bw_method)
        densities = kde.evaluate(eval_rts)
        return float(-np.mean(np.log(np.maximum(densities, eps))))
    except Exception:
        return float("nan")


def kde_nll_pertrial(eval_rts: np.ndarray, fit_rts: np.ndarray,
                     bw_method: str = "scott", eps: float = 1e-10, *,
                     rt_range: tuple[float, float]) -> np.ndarray:
    """
    Per-trial -log density of *eval_rts* under a KDE fitted to *fit_rts*.

    Same model as :func:`kde_nll`, which is the mean of this; keeping the
    individual terms is what makes the standard error of the NLL reportable.
    """
    fit_rts  = np.clip(fit_rts,  *rt_range)
    eval_rts = np.clip(eval_rts, *rt_range)

    if len(fit_rts) < 3 or len(eval_rts) == 0:
        return np.array([])
    try:
        kde = gaussian_kde(fit_rts, bw_method=bw_method)
        return -np.log(np.maximum(kde.evaluate(eval_rts), eps))
    except Exception:
        return np.array([])


def loto_nll(human_rts: np.ndarray, bw_method: str = "scott",
             eps: float = 1e-10, *, rt_range: tuple[float, float]) -> float:
    """
    Leave-one-trial-out NLL on human RTs (HH noise ceiling).

    For each held-out RT, fits KDE on the rest and evaluates density.
    Skip if < 4 observations.
    """
    rts = np.clip(human_rts, *rt_range)
    n   = len(rts)
    if n < 4:
        return float("nan")

    log_densities = []
    for i in range(n):
        rest = np.concatenate([rts[:i], rts[i + 1:]])
        if len(rest) < 3:
            continue
        try:
            kde = gaussian_kde(rest, bw_method=bw_method)
            d   = float(kde.evaluate(np.array([rts[i]]))[0])
            log_densities.append(np.log(max(d, eps)))
        except Exception:
            pass

    return float(-np.mean(log_densities)) if log_densities else float("nan")


class RTLogLikelihoodCalculator:

    def __init__(self, bandwidth_method='scott', min_observations=2):
        self.bandwidth_method = bandwidth_method
        self.min_observations = min_observations

    def _fit_gaussian_to_rts(self, rts):
        if len(rts) < self.min_observations:
            return None
        return {'mean': float(np.mean(rts)),
                'std': max(float(np.std(rts, ddof=1)) if len(rts) > 1 else 1e-6, 1e-6)}

    def _gaussian_log_likelihood(self, rt, mean, std):
        return float(norm.logpdf(rt, loc=mean, scale=std))

    def _fit_lognormal_to_rts(self, rts):
        if len(rts) < self.min_observations:
            return None
        log_rts = np.log(np.asarray(rts, dtype=float))
        mu = float(np.mean(log_rts))
        sigma = max(float(np.std(log_rts, ddof=1)) if len(log_rts) > 1 else 1e-6, 1e-6)
        return {'mu': mu, 'sigma': sigma}

    def _lognormal_log_likelihood(self, rt, mu, sigma):
        return float(lognorm.logpdf(rt, s=sigma, scale=np.exp(mu)))

    def _fit_gamma_to_rts(self, rts):
        if len(rts) < self.min_observations:
            return None
        try:
            shape, _loc, scale = gamma.fit(rts, floc=0)
            return {'shape': float(shape), 'scale': float(scale)}
        except Exception:
            return None

    def _gamma_log_likelihood(self, rt, shape, scale):
        return float(gamma.logpdf(rt, a=shape, scale=scale))

    def _fit_inverse_gaussian_to_rts(self, rts):
        if len(rts) < self.min_observations:
            return None
        try:
            mean_rt = float(np.mean(rts))
            var_rt = float(np.var(rts))
            if var_rt < 1e-10 or mean_rt < 1e-10:
                return None
            lam = mean_rt ** 3 / var_rt
            return {'mu': mean_rt, 'lambda': lam}
        except Exception:
            return None

    def _inverse_gaussian_log_likelihood(self, rt, mu, lambda_param):
        return float(invgauss.logpdf(rt, mu=mu / lambda_param, scale=lambda_param))

    def _fit_kde_to_rts(self, rts):
        if len(rts) < self.min_observations:
            return None
        try:
            return gaussian_kde(rts, bw_method=self.bandwidth_method)
        except Exception:
            return None

    def _kde_log_likelihood(self, rt, kde):
        d = float(kde.evaluate([rt])[0])
        return float(np.log(max(d, 1e-300)))

    def _compute_ll(self, held_out_rt, train_rts, method):
        tr = np.asarray(train_rts, dtype=float)
        if method == 'lognormal':
            p = self._fit_lognormal_to_rts(tr)
            return self._lognormal_log_likelihood(held_out_rt, p['mu'], p['sigma']) if p else None
        elif method == 'gamma':
            p = self._fit_gamma_to_rts(tr)
            return self._gamma_log_likelihood(held_out_rt, p['shape'], p['scale']) if p else None
        elif method == 'inverse_gaussian':
            p = self._fit_inverse_gaussian_to_rts(tr)
            return (self._inverse_gaussian_log_likelihood(held_out_rt, p['mu'], p['lambda'])
                    if p else None)
        elif method == 'gaussian':
            p = self._fit_gaussian_to_rts(tr)
            return self._gaussian_log_likelihood(held_out_rt, p['mean'], p['std']) if p else None
        elif method == 'kde':
            kde = self._fit_kde_to_rts(tr)
            return self._kde_log_likelihood(held_out_rt, kde) if kde else None
        return None

    def calculate_human_human_loso_ll(self, image_indices, rts, method='lognormal',
                                      use_median=False):
        valid = ['lognormal', 'gamma', 'inverse_gaussian', 'gaussian', 'kde']
        if method not in valid:
            raise ValueError(f"method must be one of {valid}")
        if method == 'gaussian':
            warnings.warn("'gaussian' is incorrect for positive RTs.", UserWarning)

        stim_rts = defaultdict(list)
        for img, rt in zip(image_indices, rts):
            stim_rts[int(img)].append(float(rt))

        per_ll, per_mean, all_lls = {}, {}, []
        n_stim = n_pred = 0

        for img_idx, rt_list in stim_rts.items():
            if len(rt_list) < 2:
                continue
            lls = []
            for i in range(len(rt_list)):
                train = [rt_list[j] for j in range(len(rt_list)) if j != i]
                ll = self._compute_ll(rt_list[i], np.array(train), method)
                if ll is not None:
                    lls.append(ll)
                    all_lls.append(ll)
                    n_pred += 1
            if lls:
                per_ll[img_idx] = lls
                per_mean[img_idx] = (np.median(lls) if use_median else np.mean(lls))
                n_stim += 1

        return {
            'per_stimulus_ll': per_ll,
            'per_stimulus_mean_ll': per_mean,
            'overall_mean_ll': (np.median(all_lls) if use_median else np.mean(all_lls))
                               if all_lls else None,
            'n_stimuli_evaluated': n_stim,
            'n_total_predictions': n_pred,
            'method': method,
            'conditioned': False,
        }

    def calculate_human_human_loso_ll_conditioned(self, image_indices, rts,
                                                   time_limits, method='lognormal',
                                                   use_median=False):
        valid = ['lognormal', 'gamma', 'inverse_gaussian', 'gaussian', 'kde']
        if method not in valid:
            raise ValueError(f"method must be one of {valid}")
        if method == 'gaussian':
            warnings.warn("'gaussian' is incorrect for positive RTs.", UserWarning)

        stim_rts = defaultdict(list)
        for img, rt, tl in zip(image_indices, rts, time_limits):
            if float(tl) <= 0:
                continue
            key = (int(img), round(float(tl), 4))
            stim_rts[key].append(float(rt))

        per_ll, per_mean, all_lls = {}, {}, []
        per_n_train = {}
        n_stim = n_pred = 0

        for key, rt_list in stim_rts.items():
            if len(rt_list) < 2:
                continue
            lls = []
            for i in range(len(rt_list)):
                train = [rt_list[j] for j in range(len(rt_list)) if j != i]
                ll = self._compute_ll(rt_list[i], np.array(train), method)
                if ll is not None:
                    lls.append(ll)
                    all_lls.append(ll)
                    n_pred += 1
            if lls:
                per_ll[key] = lls
                per_mean[key] = (np.median(lls) if use_median else np.mean(lls))
                per_n_train[key] = len(rt_list) - 1
                n_stim += 1

        return {
            'per_stimulus_ll': per_ll,
            'per_stimulus_mean_ll': per_mean,
            'overall_mean_ll': (np.median(all_lls) if use_median else np.mean(all_lls))
                               if all_lls else None,
            'n_stimuli_evaluated': n_stim,
            'n_total_predictions': n_pred,
            'method': method,
            'conditioned': True,
            'per_n_train': per_n_train,
        }

    def calculate_model_rt_ll(self, image_indices, human_rts, model_rts_samples,
                               method='lognormal'):
        valid = ['lognormal', 'gamma', 'inverse_gaussian', 'gaussian', 'kde']
        if method not in valid:
            raise ValueError(f"method must be one of {valid}")
        if method == 'gaussian':
            warnings.warn("'gaussian' is incorrect for positive RTs.", UserWarning)

        if isinstance(model_rts_samples, dict):
            model_by_stim = model_rts_samples
        else:
            model_by_stim = defaultdict(list)
            for obs_idx, img in enumerate(image_indices):
                model_by_stim[int(img)].extend(model_rts_samples[:, obs_idx])

        human_by_stim = defaultdict(list)
        for img, rt in zip(image_indices, human_rts):
            human_by_stim[int(img)].append(float(rt))

        per_ll, per_mean, all_lls = {}, {}, []
        n_stim = n_pred = 0

        for img_idx, human_obs in human_by_stim.items():
            if img_idx not in model_by_stim:
                continue
            model_s = np.array(model_by_stim[img_idx], dtype=float)
            if len(model_s) < self.min_observations:
                continue
            lls = []
            for human_rt in human_obs:
                ll = self._compute_ll(human_rt, model_s, method)
                if ll is not None:
                    lls.append(ll)
                    all_lls.append(ll)
                    n_pred += 1
            if lls:
                per_ll[img_idx] = lls
                per_mean[img_idx] = float(np.mean(lls))
                n_stim += 1

        return {
            'per_stimulus_ll': per_ll,
            'per_stimulus_mean_ll': per_mean,
            'overall_mean_ll': float(np.mean(all_lls)) if all_lls else None,
            'n_stimuli_evaluated': n_stim,
            'n_total_predictions': n_pred,
            'method': method,
            'conditioned': False,
        }

    def calculate_model_rt_ll_conditioned(self, image_indices, human_rts, time_limits,
                                           model_rts_samples_by_image_beep,
                                           method='lognormal',
                                           balance_n_samples=True,
                                           hh_n_train=None,
                                           rng_seed=42):
        valid = ['lognormal', 'gamma', 'inverse_gaussian', 'gaussian', 'kde']
        if method not in valid:
            raise ValueError(f"method must be one of {valid}")
        if method == 'gaussian':
            warnings.warn("'gaussian' is incorrect for positive RTs.", UserWarning)
        if balance_n_samples and hh_n_train is None:
            warnings.warn(
                "balance_n_samples=True but hh_n_train is None: "
                "full sample count will be used. "
                "Pass hh_n_train=hh_results['per_n_train'] for balanced comparison.",
                UserWarning,
            )
            balance_n_samples = False

        rng = np.random.default_rng(rng_seed)

        human_by_key = defaultdict(list)
        for img, rt, tl in zip(image_indices, human_rts, time_limits):
            if float(tl) <= 0:
                continue
            key = (int(img), round(float(tl), 4))
            human_by_key[key].append(float(rt))

        per_ll, per_mean, all_lls = {}, {}, []
        n_stim = n_pred = 0

        for key, human_obs in human_by_key.items():
            if key not in model_rts_samples_by_image_beep:
                continue
            model_all = np.array(model_rts_samples_by_image_beep[key], dtype=float)
            if len(model_all) < self.min_observations:
                continue

            if balance_n_samples and hh_n_train is not None and key in hh_n_train:
                n_train = max(int(hh_n_train[key]), self.min_observations)
                if len(model_all) > n_train:
                    idx = rng.choice(len(model_all), size=n_train, replace=False)
                    model_s = model_all[idx]
                else:
                    model_s = model_all
            else:
                model_s = model_all

            lls = []
            for human_rt in human_obs:
                ll = self._compute_ll(human_rt, model_s, method)
                if ll is not None:
                    lls.append(ll)
                    all_lls.append(ll)
                    n_pred += 1
            if lls:
                per_ll[key] = lls
                per_mean[key] = float(np.mean(lls))
                n_stim += 1

        return {
            'per_stimulus_ll': per_ll,
            'per_stimulus_mean_ll': per_mean,
            'overall_mean_ll': float(np.mean(all_lls)) if all_lls else None,
            'n_stimuli_evaluated': n_stim,
            'n_total_predictions': n_pred,
            'method': method,
            'conditioned': True,
            'balanced': balance_n_samples,
        }

    def compare_predictive_performance(self, human_human_results, model_results):
        hh_keys = set(human_human_results['per_stimulus_mean_ll'].keys())
        mod_keys = set(model_results['per_stimulus_mean_ll'].keys())
        common = hh_keys & mod_keys

        if not common:
            return {
                'n_common_stimuli': 0,
                'human_human_mean_ll': None,
                'model_mean_ll': None,
                'model_advantage': None,
                'per_stimulus_advantage': {},
                'model_wins': 0, 'human_human_wins': 0, 'ties': 0, 'win_rate': 0,
            }

        hh_lls = [human_human_results['per_stimulus_mean_ll'][k] for k in common]
        mod_lls = [model_results['per_stimulus_mean_ll'][k] for k in common]

        hh_mean = float(np.mean(hh_lls))
        mod_mean = float(np.mean(mod_lls))
        adv = mod_mean - hh_mean

        per_adv = {
            k: model_results['per_stimulus_mean_ll'][k] -
               human_human_results['per_stimulus_mean_ll'][k]
            for k in common
        }
        mw = sum(1 for v in per_adv.values() if v > 0)
        hw = sum(1 for v in per_adv.values() if v < 0)
        ti = sum(1 for v in per_adv.values() if v == 0)

        return {
            'n_common_stimuli': len(common),
            'human_human_mean_ll': hh_mean,
            'model_mean_ll': mod_mean,
            'model_advantage': adv,
            'per_stimulus_advantage': per_adv,
            'model_wins': mw,
            'human_human_wins': hw,
            'ties': ti,
            'win_rate': mw / len(common),
        }


def _compute_comparison_for_keys(keys, hh_mean, mod_mean):
    """
    Recompute the comparison dict for a subset of keys.
    Used by create_ll_comparison_plots_per_beep_separate.
    """
    hh_lls  = [hh_mean[k]  for k in keys]
    mod_lls = [mod_mean[k] for k in keys]

    hh_m  = float(np.mean(hh_lls))
    mod_m = float(np.mean(mod_lls))
    adv   = mod_m - hh_m

    per_adv = {k: mod_mean[k] - hh_mean[k] for k in keys}
    mw = sum(1 for v in per_adv.values() if v > 0)
    hw = sum(1 for v in per_adv.values() if v < 0)
    ti = sum(1 for v in per_adv.values() if v == 0)

    return {
        'n_common_stimuli':       len(keys),
        'human_human_mean_ll':    hh_m,
        'model_mean_ll':          mod_m,
        'model_advantage':        adv,
        'per_stimulus_advantage': per_adv,
        'model_wins':             mw,
        'human_human_wins':       hw,
        'ties':                   ti,
        'win_rate':               mw / len(keys) if keys else 0.0,
    }


# =============================================================================
# Pooled beep-level LL  (SAT)
# =============================================================================

def calculate_hh_and_model_ll_beep_pooled(
        image_indices, human_rts, time_limits,
        model_rts_samples_by_image_beep,
        method='lognormal',
        min_obs=10):
    """
    Pooled RT analysis for SAT: grouped by beep, HH-LOSO vs model.

    Structure
    ---------
    For every beep value b with N >= min_obs human observations:

    1. HH-LOSO: for every (human) rt_i, a lognormal fit on the N-1 other human
       RTs at the same beep (ignoring the image) -> LL(rt_i | fit).
       With N ~ 100-300 the fit is stable (sigma does not collapse).

    2. Model: pool of ALL model samples at that beep (N_model ~ N*100).
       ONE lognormal fit on all samples -> LL(rt_i | fit_model)
       for every human rt_i.
       No subsampling: using every available sample gives the best estimate
       of the model's distribution and removes subsampling noise.

    3. KS test (scipy.stats.ks_2samp) between the empirical distribution of
       the human RTs and the distribution of the model samples at beep b.

    4. RT statistics: mean, std, percentiles for humans and model at beep b.

    Why the model samples are NOT balanced here
    -------------------------------------------
    Balancing (using only N model samples instead of N*100) makes sense for
    an analysis per (image, beep), where N_human = 2-5 and the HH sigma is
    unstable. When pooled, N_human >= min_obs (default 10, typically 100-300)
    and the HH fit is already stable. Restricting the model to N samples when
    N*100 are available only adds noise to the estimate of its distribution.

    Interpretation
    --------------
    - Marginal LL per beep: whether the model's RT distribution at beep b is
      calibrated like the human one, regardless of the stimulus.
    - KS distance: complements the LLs with a non-parametric distance between
      the two distributions.
    - Q-Q plot (from the saved percentiles): shows where the two distributions
      differ systematically.

    Parameters
    ----------
    min_obs : int
        Minimum number of human observations to include a beep (default 10).

    Returns
    -------
    dict: { beep -> {
        'hh_lls':         list[float]   per-observation LL (HH-LOSO)
        'model_lls':      list[float]   per-observation LL (single model fit)
        'rts':            list[float]   human RTs
        'model_samples':  list[float]   model RT samples at that beep
        'n_obs':          int
        'n_model_samples':int
        'hh_mean':        float
        'model_mean':     float
        'model_advantage':float
        'ks_stat':        float
        'ks_pval':        float
        'human_mean_rt':  float
        'human_std_rt':   float
        'model_mean_rt':  float
        'model_std_rt':   float
        'human_percentiles': list[float]  0,5,10,...,95,100
        'model_percentiles': list[float]  0,5,10,...,95,100
        'method':         str
    } }
    """
    calc = RTLogLikelihoodCalculator()

    # Group the human RTs by beep
    beep_rts = defaultdict(list)
    for img, rt, tl in zip(image_indices, human_rts, time_limits):
        if float(tl) <= 0:
            continue
        beep_rts[round(float(tl), 4)].append(float(rt))

    # Group the model samples by beep (pooled over all images)
    beep_model_samples = defaultdict(list)
    for (img_idx, beep_key), samples in model_rts_samples_by_image_beep.items():
        beep_model_samples[beep_key].extend(samples)

    pct_grid = np.arange(0, 101, 5)  # 0,5,10,...,95,100
    results  = {}

    for beep, rts_all in beep_rts.items():
        n_obs = len(rts_all)
        if n_obs < min_obs:
            warnings.warn(
                f"Beep {beep:.3f}s: only {n_obs} observations (< {min_obs}), "
                f"skipped in the pooled analysis.",
                UserWarning,
            )
            continue

        rts_arr   = np.array(rts_all, dtype=float)
        model_all = np.array(beep_model_samples.get(beep, []), dtype=float)
        n_total   = len(model_all)

        # ---- HH-LOSO: leave one observation out at a time ----
        hh_lls = []
        for i in range(n_obs):
            train = np.concatenate([rts_arr[:i], rts_arr[i + 1:]])
            ll    = calc._compute_ll(rts_arr[i], train, method)
            hh_lls.append(ll if ll is not None else np.nan)

        # ---- Model: ONE fit on ALL available samples ----
        if n_total < calc.min_observations:
            model_lls          = [np.nan] * n_obs
            model_pct          = np.full(len(pct_grid), np.nan)
            model_mean_rt_val  = np.nan
            model_std_rt_val   = np.nan
        else:
            model_s = model_all   # use ALL samples, no subsampling
            model_lls = []
            for rt in rts_arr:
                ll = calc._compute_ll(rt, model_s, method)
                model_lls.append(ll if ll is not None else np.nan)
            model_pct         = np.percentile(model_s, pct_grid)
            model_mean_rt_val = float(np.mean(model_s))
            model_std_rt_val  = float(np.std(model_s))

        # ---- KS test (empirical distribution) ----
        if n_total >= 2:
            ks_stat_v, ks_pval_v = ks_2samp(rts_arr, model_all)
        else:
            ks_stat_v, ks_pval_v = np.nan, np.nan

        # ---- Summary statistics ----
        valid_hh  = [v for v in hh_lls    if not np.isnan(v)]
        valid_mod = [v for v in model_lls  if not np.isnan(v)]
        adv_list  = [m - h for h, m in zip(hh_lls, model_lls)
                     if not np.isnan(h) and not np.isnan(m)]

        mw_pool = sum(1 for v in adv_list if v > 0)
        hw_pool = sum(1 for v in adv_list if v < 0)

        results[beep] = {
            'hh_lls':            hh_lls,
            'model_lls':         model_lls,
            'rts':               rts_all,
            'model_samples':     model_all.tolist() if n_total > 0 else [],
            'n_obs':             n_obs,
            'n_model_samples':   n_total,
            'hh_mean':           float(np.mean(valid_hh))  if valid_hh  else np.nan,
            'model_mean':        float(np.mean(valid_mod)) if valid_mod else np.nan,
            'model_advantage':   float(np.mean(adv_list))  if adv_list  else np.nan,
            'model_wins':        mw_pool,
            'hh_wins':           hw_pool,
            'model_win_rate':    mw_pool / len(adv_list) if adv_list else np.nan,
            'ks_stat':           float(ks_stat_v),
            'ks_pval':           float(ks_pval_v),
            'human_mean_rt':     float(np.mean(rts_arr)),
            'human_std_rt':      float(np.std(rts_arr)),
            'model_mean_rt':     model_mean_rt_val,
            'model_std_rt':      model_std_rt_val,
            'human_percentiles': np.percentile(rts_arr, pct_grid).tolist(),
            'model_percentiles': model_pct.tolist(),
            'method':            method,
        }

    return results


def save_pooled_beep_ll_csv(pooled_results, save_path):
    """
    Save the pooled per-beep LL results to a summary CSV.
    One row per beep with mean LLs, RT statistics and the KS test.
    """
    from pathlib import Path as _Path
    rows = []
    for beep in sorted(pooled_results.keys()):
        res      = pooled_results[beep]
        hh_valid = [v for v in res['hh_lls']   if not np.isnan(v)]
        mo_valid = [v for v in res['model_lls'] if not np.isnan(v)]
        adv_list = [m - h for h, m in zip(res['hh_lls'], res['model_lls'])
                    if not np.isnan(h) and not np.isnan(m)]
        mw = sum(1 for v in adv_list if v > 0)
        hw = sum(1 for v in adv_list if v < 0)
        rows.append({
            'beep_s':            beep,
            'n_human_obs':       res['n_obs'],
            'n_model_samples':   res.get('n_model_samples', np.nan),
            'hh_mean_ll':        res['hh_mean'],
            'hh_std_ll':         float(np.std(hh_valid)) if hh_valid else np.nan,
            'model_mean_ll':     res['model_mean'],
            'model_std_ll':      float(np.std(mo_valid)) if mo_valid else np.nan,
            'model_advantage':   res['model_advantage'],
            'model_wins':        mw,
            'hh_wins':           hw,
            'model_win_rate':    mw / len(adv_list) if adv_list else np.nan,
            'ks_stat':           res.get('ks_stat', np.nan),
            'ks_pval':           res.get('ks_pval', np.nan),
            'human_mean_rt':     res.get('human_mean_rt', np.nan),
            'human_std_rt':      res.get('human_std_rt', np.nan),
            'model_mean_rt':     res.get('model_mean_rt', np.nan),
            'model_std_rt':      res.get('model_std_rt', np.nan),
            'rt_mean_diff':      (res.get('model_mean_rt', np.nan) -
                                  res.get('human_mean_rt', np.nan)),
            'rt_std_diff':       (res.get('model_std_rt', np.nan) -
                                  res.get('human_std_rt', np.nan)),
            'method':            res.get('method', ''),
        })
    import csv
    if not rows:
        return
    _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Cross-image beep-level LL  (SAT) — non-degenerate scatter
# =============================================================================

def calculate_hh_model_ll_beep_cross_image(
        image_indices, human_rts, time_limits,
        model_rts_samples_by_image_beep,
        method='lognormal',
        min_images=3):
    """
    SAT-only: for every beep and every image j:

    - HH LL (leave-one-image-out): lognormal fit on ALL human RTs at the same
      beep except those of image j → LL(rt_i | fit).
      With N_beep ~ 200 and N_j ~ 5, HH_train has ~195 RTs → a stable fit.

    - Model LL: lognormal fit on the model samples for (image j, beep) →
      LL(rt_i | fit_model_j).

    Why the scatter is 2D
    ---------------------
    Both HH and Model change with the image:
    - HH_train varies because a different image is left out each time.
    - Model changes because the samples depend on the image (the model is
      conditioned on the image embedding).
    The result is a cloud of points (one point = one image, the mean LL of
    all its observations at that beep), not a monotone curve.

    Returns
    -------
    dict: { beep -> {
        'hh_img_means':    list[float]  mean HH LL per image (for the scatter)
        'model_img_means': list[float]  mean Model LL per image (for the scatter)
        'hh_lls':          list[float]  LL per single observation
        'model_lls':       list[float]  LL per single observation
        'n_images':        int          number of images at the beep
        'n_obs':           int          total observations at the beep
        'hh_mean':         float        global mean HH LL
        'model_mean':      float        global mean Model LL
        'model_advantage': float        mean (Model LL - HH LL)
        'method':          str
    } }
    """
    calc = RTLogLikelihoodCalculator()

    # Group the human RTs by (image, beep)
    img_beep_rts = defaultdict(list)
    for img, rt, tl in zip(image_indices, human_rts, time_limits):
        if float(tl) <= 0:
            continue
        img_beep_rts[(int(img), round(float(tl), 4))].append(float(rt))

    # Every beep with its images
    beep_imgs = defaultdict(set)
    for (img, beep) in img_beep_rts:
        beep_imgs[beep].add(img)

    results = {}

    for beep in sorted(beep_imgs.keys()):
        img_list = sorted(beep_imgs[beep])
        if len(img_list) < min_images:
            continue

        hh_img_means    = []
        model_img_means = []
        hh_all          = []
        model_all       = []

        for img_j in img_list:
            rts_j = np.array(img_beep_rts[(img_j, beep)], dtype=float)
            if len(rts_j) == 0:
                continue

            # HH train: all RTs of OTHER images at the same beep
            hh_train_parts = [
                np.array(img_beep_rts[(img_k, beep)], dtype=float)
                for img_k in img_list if img_k != img_j
            ]
            if not hh_train_parts:
                continue
            hh_train = np.concatenate(hh_train_parts)
            if len(hh_train) < 2:
                continue

            # Model samples per (img_j, beep)
            model_s = np.array(
                model_rts_samples_by_image_beep.get((img_j, beep), []),
                dtype=float
            )
            if len(model_s) < 2:
                continue

            hh_lls_j    = []
            model_lls_j = []
            for rt in rts_j:
                hh_ll    = calc._compute_ll(rt, hh_train, method)
                model_ll = calc._compute_ll(rt, model_s,  method)
                if hh_ll is not None and model_ll is not None:
                    hh_lls_j.append(hh_ll)
                    model_lls_j.append(model_ll)

            if not hh_lls_j:
                continue

            hh_img_means.append(float(np.mean(hh_lls_j)))
            model_img_means.append(float(np.mean(model_lls_j)))
            hh_all.extend(hh_lls_j)
            model_all.extend(model_lls_j)

        if not hh_img_means:
            continue

        hh_arr    = np.array(hh_all,    dtype=float)
        model_arr = np.array(model_all, dtype=float)

        results[beep] = {
            'hh_img_means':    hh_img_means,
            'model_img_means': model_img_means,
            'hh_lls':          hh_all,
            'model_lls':       model_all,
            'n_images':        len(hh_img_means),
            'n_obs':           len(hh_all),
            'hh_mean':         float(np.nanmean(hh_arr)),
            'model_mean':      float(np.nanmean(model_arr)),
            'model_advantage': float(np.nanmean(model_arr - hh_arr)),
            'method':          method,
        }

    return results


