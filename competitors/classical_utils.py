import numpy as np
import pandas as pd


def prepare_data(df, rt_min=0.01, rt_max=10.0):
    cols = ["stim", "response", "resp_rt", "correct"]
    data = df[cols].dropna().copy()
    data = data[(data["resp_rt"] > rt_min) & (data["resp_rt"] < rt_max)]
    data["correct"] = data["correct"].astype(bool)
    return data.reset_index(drop=True)


def aic(nll, k):
    return 2.0 * k + 2.0 * nll


def bic(nll, k, n):
    return k * np.log(n) + 2.0 * nll


def print_fit(name, result, param_names, n_obs):
    k = len(param_names)
    nll = result.fun
    _aic = aic(nll, k)
    _bic = bic(nll, k, n_obs)

    print(f"\n{'='*54}")
    print(f"  {name}")
    print(f"{'='*54}")
    print(f"  Converged : {result.success}")
    print(f"  {'Parameter':<22} {'Estimate':>10}")
    print(f"  {'-'*34}")
    for p, v in zip(param_names, result.x):
        print(f"  {p:<22} {v:>10.4f}")
    print(f"  {'-'*34}")
    print(f"  {'NLL':<22} {nll:>10.2f}")
    print(f"  {'AIC':<22} {_aic:>10.2f}")
    print(f"  {'BIC':<22} {_bic:>10.2f}")

    return dict(name=name, nll=nll, aic=_aic, bic=_bic,
                params=dict(zip(param_names, result.x)))
