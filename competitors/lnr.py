import numpy as np
import pandas as pd
from scipy.stats import lognorm
from scipy.optimize import minimize
from collections import defaultdict


def _lnr_loglik(params, rt, correct, n_choices):
    mu_t, mu_d, sigma, t0 = params

    rt_min = rt.min()
    if sigma <= 0 or t0 <= 0 or t0 >= rt_min * 0.9:
        return 1e10

    t = rt - t0
    if np.any(t <= 0):
        return 1e10

    scale_t = np.exp(mu_t)
    scale_d = np.exp(mu_d)

    logpdf_t = lognorm.logpdf(t, s=sigma, scale=scale_t)
    logpdf_d = lognorm.logpdf(t, s=sigma, scale=scale_d)
    logsf_t = lognorm.logsf(t, s=sigma, scale=scale_t)
    logsf_d = lognorm.logsf(t, s=sigma, scale=scale_d)

    log_lik = np.where(
        correct,
        logpdf_t + (n_choices - 1) * logsf_d,
        logpdf_d + logsf_t + (n_choices - 2) * logsf_d,
    )

    nll = -np.sum(log_lik)
    return float(nll) if np.isfinite(nll) else 1e10


def fit_lnr(data, n_choices):
    rt = data["resp_rt"].values
    correct = data["correct"].values

    bounds = [
        (-5.0, 2.0),
        (-5.0, 2.0),
        (0.01, 3.0),
        (0.01, rt.min() * 0.9),
    ]

    best = None
    for mu_t0, mu_d0 in [(-0.3, 0.2), (-0.8, 0.0), (0.0, 0.5)]:
        x0 = [mu_t0, mu_d0, 0.5, rt.min() * 0.4]
        res = minimize(
            _lnr_loglik, x0,
            args=(rt, correct, n_choices),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-13},
        )
        if best is None or res.fun < best.fun:
            best = res

    return best


def simulate_lnr(params, n_trials, n_choices, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    mu_t, mu_d, sigma, t0 = params
    scale_t = np.exp(mu_t)
    scale_d = np.exp(mu_d)

    stim = rng.integers(0, n_choices, size=n_trials)

    T_target = lognorm.rvs(s=sigma, scale=scale_t, size=n_trials, random_state=rng)
    T_dist = lognorm.rvs(s=sigma, scale=scale_d,
                         size=(n_trials, n_choices - 1),
                         random_state=rng)

    T_all = np.concatenate([T_target[:, None], T_dist], axis=1)
    winner = np.argmin(T_all, axis=1)
    rt = t0 + np.min(T_all, axis=1)

    response = np.zeros(n_trials, dtype=int)
    correct = np.zeros(n_trials, dtype=bool)

    for i in range(n_trials):
        if winner[i] == 0:
            response[i] = stim[i]
            correct[i] = True
        else:
            distractors = [j for j in range(n_choices) if j != stim[i]]
            response[i] = distractors[winner[i] - 1]
            correct[i] = False

    return pd.DataFrame({
        "stim": stim,
        "response": response,
        "resp_rt": rt,
        "correct": correct,
    })


def simulate_lnr_with_samples(lnr_params, test_data, n_choices,
                              n_rt_samples=100, rng=None):
    """Simulate LNR predictions aligned to ``test_data``'s rows.

    The name distinguishes it from the trial-count variant above, which
    generates synthetic trials unaligned with any test frame.

    Index 0 of the race is *the correct accumulator*, not class 0. Each trial
    therefore draws its own target label and the winning accumulator is mapped
    back onto a real label — reading index 0 as class 0 instead makes the modal
    choice 0 on every trial (the fitted target always beats the distractor), so
    the model answers one class everywhere and its AUC/kappa collapse to exactly
    0.500 / 0.000.

    ``rng`` is consumed in a fixed order — the target label *before* the
    lognormal draws — so a given seed always gives the same predictions.

    Returns
    -------
    response            : (N,) int   — modal choice across samples
    rt                  : (N,) float — median RT across samples
    rt_samples_by_image : dict[int, list[float]]
    """
    if rng is None:
        rng = np.random.default_rng()

    mu_t, mu_d, sigma, t0 = lnr_params
    scale_t = np.exp(mu_t)
    scale_d = np.exp(mu_d)

    n_trials = len(test_data)
    image_indices = test_data["image_index"].values

    rt = np.zeros(n_trials)
    response = np.zeros(n_trials, dtype=int)
    rt_samples_by_image = defaultdict(list)

    for i in range(n_trials):
        target = rng.integers(0, n_choices)
        distractors = [j for j in range(n_choices) if j != target]

        T_target = lognorm.rvs(s=sigma, scale=scale_t,
                               size=n_rt_samples, random_state=rng)
        T_dist = lognorm.rvs(s=sigma, scale=scale_d,
                             size=(n_rt_samples, n_choices - 1),
                             random_state=rng)
        T_all = np.concatenate([T_target[:, None], T_dist], axis=1)
        winner_idx = np.argmin(T_all, axis=1)
        rts_trial = t0 + T_all[np.arange(n_rt_samples), winner_idx]

        choices_trial = np.where(
            winner_idx == 0,
            target,
            np.array(distractors)[np.minimum(winner_idx - 1, len(distractors) - 1)],
        )

        rt[i] = np.median(rts_trial)
        response[i] = np.bincount(choices_trial, minlength=n_choices).argmax()
        rt_samples_by_image[int(image_indices[i])].extend(rts_trial.tolist())

    return response, rt, dict(rt_samples_by_image)
