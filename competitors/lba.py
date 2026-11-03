import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from collections import defaultdict


def _lba_pdf(t, v, sv, A, b):
    e1 = (b - A - v * t) / (sv * t)
    e2 = (b - v * t) / (sv * t)
    f = (1.0 / A) * (-v * (norm.cdf(e1) - norm.cdf(e2))
                      + sv * (norm.pdf(e1) - norm.pdf(e2)))
    return np.maximum(f, 0.0)


def _lba_sf(t, v, sv, A, b):
    e1 = (b - A - v * t) / (sv * t)
    e2 = (b - v * t) / (sv * t)
    sf = (norm.cdf(e1)
          - (b - v * t) / A * (norm.cdf(e1) - norm.cdf(e2))
          - t * sv / A * (norm.pdf(e1) - norm.pdf(e2)))
    return np.maximum(sf, 0.0)


def _lba_loglik(params, rt, correct, n_choices):
    v_t, v_d, sv, A, b, t0 = params

    rt_min = rt.min()
    if sv <= 0 or A <= 0 or b <= A or t0 <= 0 or t0 >= rt_min * 0.9:
        return 1e10

    t = rt - t0
    if np.any(t <= 0):
        return 1e10

    pdf_t = _lba_pdf(t, v_t, sv, A, b)
    pdf_d = _lba_pdf(t, v_d, sv, A, b)
    sf_t = _lba_sf(t, v_t, sv, A, b)
    sf_d = _lba_sf(t, v_d, sv, A, b)

    eps = 1e-300

    log_lik = np.where(
        correct,
        np.log(np.maximum(pdf_t, eps)) + (n_choices - 1) * np.log(np.maximum(sf_d, eps)),
        np.log(np.maximum(pdf_d, eps)) + np.log(np.maximum(sf_t, eps))
        + (n_choices - 2) * np.log(np.maximum(sf_d, eps)),
    )

    nll = -np.sum(log_lik)
    return float(nll) if np.isfinite(nll) else 1e10


def fit_lba(data, n_choices):
    rt = data["resp_rt"].values
    correct = data["correct"].values

    bounds = [
        (0.01, 15.0),
        (0.01, 15.0),
        (0.01, 5.0),
        (0.01, 5.0),
        (0.05, 15.0),
        (0.01, rt.min() * 0.9),
    ]

    best = None
    for v_t0 in [2.0, 4.0, 6.0, 1.5]:
        x0 = [v_t0, max(v_t0 * 0.4, 0.5), 1.0, 0.5, 1.2, rt.min() * 0.3]
        res = minimize(
            _lba_loglik, x0,
            args=(rt, correct, n_choices),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-13},
        )
        if best is None or res.fun < best.fun:
            best = res

    return best


def simulate_lba(lba_params, test_data, n_choices, n_rt_samples=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    v_t, v_d, sv, A, b, t0 = lba_params

    n_trials = len(test_data)
    image_indices = test_data["image_index"].values

    rt = np.zeros(n_trials)
    response = np.zeros(n_trials, dtype=int)
    rt_samples_by_image = defaultdict(list)

    for i in range(n_trials):
        # One target label per *trial*, drawn before the sample loop.
        # Re-drawing it per sample instead scatters the modal choice across
        # all classes and consumes the RNG in a different order.
        target = rng.integers(0, n_choices)

        rts_trial = np.zeros(n_rt_samples)
        choices_trial = np.zeros(n_rt_samples, dtype=int)

        for j in range(n_rt_samples):
            drift = np.full(n_choices, v_d)
            drift[target] = v_t
            drift = rng.normal(drift, sv)
            drift = np.maximum(drift, 1e-6)

            k = rng.uniform(0, A, size=n_choices)
            T = (b - k) / drift

            winner = np.argmin(T)
            rts_trial[j] = t0 + T[winner]
            choices_trial[j] = winner

        rt[i] = np.median(rts_trial)
        response[i] = np.bincount(choices_trial, minlength=n_choices).argmax()

        img_id = int(image_indices[i])
        rt_samples_by_image[img_id].extend(rts_trial.tolist())

    return response, rt, dict(rt_samples_by_image)
