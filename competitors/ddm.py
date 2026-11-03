import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from collections import defaultdict


def _wald_logpdf(t, v, a):
    return (np.log(a)
            - 0.5 * np.log(2.0 * np.pi)
            - 1.5 * np.log(t)
            - (v * t - a) ** 2 / (2.0 * t))


def _wald_logsf(t, v, a):
    sq = np.sqrt(t)
    z1 = (v * t - a) / sq
    z2 = (v * t + a) / sq

    log_p1 = norm.logcdf(z1)
    log_p2 = 2.0 * a * v + norm.logcdf(-z2)

    log_cdf = np.logaddexp(log_p1, log_p2)
    log_cdf = np.minimum(log_cdf, 0.0)

    log_sf = np.where(
        log_cdf > -1e-7,
        np.log(np.maximum(-np.expm1(log_cdf), 1e-300)),
        np.log1p(-np.exp(log_cdf)),
    )
    return log_sf


def _rdm_loglik(params, rt, correct, n_choices):
    v_t, v_d, a, t0 = params

    rt_min = rt.min()
    if v_t <= 0 or v_d <= 0 or a <= 0 or t0 <= 0 or t0 >= rt_min * 0.9:
        return 1e10

    t = rt - t0
    if np.any(t <= 0):
        return 1e10

    logpdf_t = _wald_logpdf(t, v_t, a)
    logpdf_d = _wald_logpdf(t, v_d, a)
    logsf_t = _wald_logsf(t, v_t, a)
    logsf_d = _wald_logsf(t, v_d, a)

    log_lik = np.where(
        correct,
        logpdf_t + (n_choices - 1) * logsf_d,
        logpdf_d + logsf_t + (n_choices - 2) * logsf_d,
    )

    nll = -np.sum(log_lik)
    return float(nll) if np.isfinite(nll) else 1e10


def fit_rdm(data, n_choices):
    rt = data["resp_rt"].values
    correct = data["correct"].values

    bounds = [
        (0.01, 20.0),
        (0.01, 20.0),
        (0.01, 10.0),
        (0.01, rt.min() * 0.9),
    ]

    best = None
    for v_t0, v_d0, a0 in [
        (3.0, 1.0, 1.0),
        (5.0, 2.0, 2.0),
        (1.5, 0.5, 0.5),
        (8.0, 3.0, 1.5),
    ]:
        x0 = [v_t0, v_d0, a0, rt.min() * 0.3]
        res = minimize(
            _rdm_loglik, x0,
            args=(rt, correct, n_choices),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-13},
        )
        if best is None or res.fun < best.fun:
            best = res

    return best


def _sample_wald(v, a, size, rng):
    mu = a / v
    lam = a ** 2
    nu = rng.standard_normal(size)
    y = nu ** 2
    x = (mu
         + mu ** 2 * y / (2.0 * lam)
         - mu / (2.0 * lam) * np.sqrt(4.0 * mu * lam * y + mu ** 2 * y ** 2))
    u = rng.uniform(size=size)
    return np.where(u <= mu / (mu + x), x, mu ** 2 / x)


def simulate_rdm(rdm_params, test_data, n_choices, n_rt_samples=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    v_t, v_d, a, t0 = rdm_params
    n_trials = len(test_data)
    image_indices = test_data["image_index"].values

    rt = np.zeros(n_trials)
    response = np.zeros(n_trials, dtype=int)
    rt_samples_by_image = defaultdict(list)

    for i in range(n_trials):
        target = rng.integers(0, n_choices)
        distractors = [j for j in range(n_choices) if j != target]

        T_target = _sample_wald(v_t, a, n_rt_samples, rng)
        T_dist = np.stack(
            [_sample_wald(v_d, a, n_rt_samples, rng) for _ in range(n_choices - 1)],
            axis=1,
        )
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
