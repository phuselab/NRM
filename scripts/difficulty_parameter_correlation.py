"""Stimulus difficulty vs learned parameters: drift, diffusion and threshold (§3)."""

import os
import numpy as np
import pandas as pd
import torch
import sys as _sys_for_backend

import matplotlib
# Force the headless backend only when nothing has set one up yet, i.e. when
# this module is imported by a script. A notebook keeps its inline backend, so
# plt.show() still renders there.
if "matplotlib.pyplot" not in _sys_for_backend.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

import sys
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nrm.config import build_model
from scripts.results_io import display_path
from datasets.rdm.loader import (
    load_rdm_nrm_dataset, PKL_OUT, ME_EMBS_OUT, COHERENCE_LIST,
)

OUT = os.path.join(_REPO_ROOT, "outputs", "rdm")
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(_REPO_ROOT, "nrm", "weights", "NRM_rdm.pth")
EMBED_DIM = 24          # 8 directions x 3 speeds
COND_DIM = 32
SEED = 42
lines = []
def emit(s=""):
    print(s); lines.append(s)


# ---------------------------------------------------------------------------
# 1. model + data
# ---------------------------------------------------------------------------
def load_model():
    # build_model("rdm") carries the RDM configuration, so this
    # analysis always probes the same architecture the weights were trained on.
    model = build_model("rdm", device=DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def load_test_set():
    ds = load_rdm_nrm_dataset(pkl_out=PKL_OUT, embeddings_path=str(ME_EMBS_OUT))
    ds.remove_rt_outliers_mad(mad_thresh=3.0, min_rt=0.05, max_rt=3.0)
    _, _, test_ds = ds.train_val_test_split(val_size=0.10, test_size=0.10, seed=SEED)
    return test_ds


# ---------------------------------------------------------------------------
# 2. extract learned parameter fields (no SDE integration needed:
#    drift/diffusion depend only on (t, cond); threshold only on cond)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_parameters(model, test_ds, batch=1024):
    embs = test_ds.embeddings                                  # [N,150,24]
    df = test_ds.df
    coh_idx = df["coherence_idx"].values.astype(int)           # 0..6
    correct = df["direction_label"].values.astype(int)         # winning accumulator
    N = len(df)
    T = int(model.t_max / model.dt)                            # 300
    ts = torch.linspace(0, model.t_max, steps=T, device=DEVICE)
    dur = embs.shape[1] / model.fs                             # 150/75 = 2.0 s

    drift_mag = np.zeros(N); drift_contrast = np.zeros(N)
    diff_mag = np.zeros(N);  thr_mean = np.zeros(N)
    snr_contrast = np.zeros(N); snr_mag = np.zeros(N)
    cond_bar = np.zeros((N, COND_DIM), dtype=np.float32)

    for s in range(0, N, batch):
        e = min(s + batch, N)
        emb = embs[s:e].to(DEVICE).float()                    # [b,150,24]
        cov = {"img_embedding": emb}
        cond = model.cond_encoder(cov, n_reps=1)              # [b,150,32]
        cond = model.interpolate_covariates(cond, ts, dur, "nearest")  # [b,300,32]
        b = emb.shape[0]
        xdummy = torch.zeros(b, model.n_choices, device=DEVICE)

        drift_ts = torch.empty(b, T, model.n_choices, device=DEVICE)
        diff_ts = torch.empty(b, T, model.n_choices, device=DEVICE)
        for k in range(T):
            drift_ts[:, k, :] = model.f_net(ts[k], xdummy, cond[:, k, :])
            diff_ts[:, k, :] = model.g_net(ts[k], xdummy, cond[:, k, :])
        theta_ts = model.threshold_net(cond)                  # [b,300]

        ci = torch.tensor(correct[s:e], device=DEVICE).long()
        d_corr = drift_ts.gather(2, ci.view(b, 1, 1).expand(b, T, 1)).squeeze(2)
        d_inc = drift_ts.gather(2, (1 - ci).view(b, 1, 1).expand(b, T, 1)).squeeze(2)

        g_norm = diff_ts.norm(dim=2)                          # [b,T] noise scale
        drift_mag[s:e] = drift_ts.norm(dim=2).mean(1).cpu().numpy()
        drift_contrast[s:e] = (d_corr - d_inc).mean(1).cpu().numpy()
        diff_mag[s:e] = g_norm.mean(1).cpu().numpy()
        thr_mean[s:e] = theta_ts.mean(1).cpu().numpy()
        # explicit signal-to-noise ratio, per time step then averaged:
        #   snr_contrast = (drift towards correct - towards incorrect) / ||diffusion||
        #   snr_mag      = ||drift|| / ||diffusion||
        snr_contrast[s:e] = ((d_corr - d_inc) / (g_norm + 1e-8)).mean(1).cpu().numpy()
        snr_mag[s:e] = (drift_ts.norm(dim=2) / (g_norm + 1e-8)).mean(1).cpu().numpy()
        cond_bar[s:e] = cond.mean(1).cpu().numpy()

    out = pd.DataFrame({
        "coh_idx": coh_idx,
        "coherence_pct": np.array(COHERENCE_LIST)[coh_idx],
        "drift_mag": drift_mag,
        "drift_contrast": drift_contrast,
        "diffusion_mag": diff_mag,
        "threshold": thr_mean,
        "snr_contrast": snr_contrast,
        "snr_mag": snr_mag,
    })
    return out, cond_bar


# ---------------------------------------------------------------------------
# 3. Approach 1 — correlations
# ---------------------------------------------------------------------------
#: The six per-trial scalars the mechanism table reports, in table order.
CORR_PARAMS = ["drift_mag", "drift_contrast", "diffusion_mag", "threshold",
               "snr_contrast", "snr_mag"]

def approach1(df):
    emit("\n" + "=" * 74)
    emit("Difficulty–parameter correlation / APPROACH 1 — coherence vs learned parameters")
    emit("=" * 74)
    logc = np.log10(df["coherence_pct"].values)
    stats = []
    for p in CORR_PARAMS:
        y = df[p].values
        r_t, pr_t = pearsonr(logc, y)               # trial-level, log-coherence
        rho_t, prho_t = spearmanr(df["coh_idx"].values, y)
        # level-level (7 per-coherence means)
        lvl = df.groupby("coh_idx")[p].mean()
        logc7 = np.log10(np.array(COHERENCE_LIST))
        r_l, pr_l = pearsonr(logc7, lvl.values)
        rho_l, _ = spearmanr(np.arange(7), lvl.values)
        stats.append(dict(param=p, pearson_r_trial=r_t, p_trial=pr_t,
                          spearman_rho_trial=rho_t, p_spearman_trial=prho_t,
                          pearson_r_level=r_l, p_level=pr_l, spearman_rho_level=rho_l))
        emit(f"\n{p}:")
        emit(f"  trial-level  Pearson r(log-coh) = {r_t:+.3f}  (p={pr_t:.2e}, n={len(y)})")
        emit(f"  trial-level  Spearman rho        = {rho_t:+.3f}  (p={prho_t:.2e})")
        emit(f"  level-level  Pearson r(7 means)  = {r_l:+.3f}  (p={pr_l:.2e})")
        emit(f"  level-level  Spearman rho        = {rho_l:+.3f}")
    stats = pd.DataFrame(stats)
    stats.to_csv(f"{OUT}/mechanism_corr_stats.csv", index=False)

    # per-level table
    tbl = df.groupby("coherence_pct")[CORR_PARAMS].agg(["mean", "std"])
    tbl.to_csv(f"{OUT}/mechanism_corr_perlevel.csv")
    emit("\nPer-coherence means:")
    emit(df.groupby("coherence_pct")[CORR_PARAMS].mean().round(4).to_string())

    # per-parameter figures
    fig, axes = plt.subplots(1, len(CORR_PARAMS), figsize=(4.5 * len(CORR_PARAMS), 4))
    for ax, p in zip(axes, CORR_PARAMS):
        m = df.groupby("coherence_pct")[p].mean()
        sd = df.groupby("coherence_pct")[p].std()
        ax.errorbar(m.index, m.values, yerr=sd.values, marker="o", capsize=3)
        ax.set_xscale("log"); ax.set_xlabel("coherence (%)"); ax.set_title(p)
    fig.tight_layout(); fig.savefig(f"{OUT}/mechanism_corr_perlevel.png", dpi=130)
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# 4. Approach 2 — supervised DR of the conditioning vector
# ---------------------------------------------------------------------------
def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return (a.mean() - b.mean()) / (sp + 1e-12)

def approach2(df, cond_bar):
    emit("\n" + "=" * 74)
    emit("Difficulty–parameter correlation / APPROACH 2 — supervised DR of the conditioning vector c-bar (R^32)")
    emit("=" * 74)
    targets = ["drift_contrast", "diffusion_mag", "threshold"]
    rows = []
    fig, axes = plt.subplots(1, len(targets), figsize=(15, 4.5))
    for ax, p in zip(axes, targets):
        y = df[p].values
        ybin = (y > np.median(y)).astype(int)                 # high / low split
        # LDA high/low classifier from the embedding (5-fold CV accuracy)
        lda_pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        acc = cross_val_score(lda_pipe, cond_bar, ybin, cv=cv, scoring="accuracy")
        # 1-D LDA coordinate (fit on all) + Cohen's d + Spearman vs continuous param
        Xs = StandardScaler().fit_transform(cond_bar)
        ld = LinearDiscriminantAnalysis().fit(Xs, ybin).transform(Xs)[:, 0]
        if spearmanr(ld, y).correlation < 0:                  # orient axis with param
            ld = -ld
        d = cohens_d(ld[ybin == 1], ld[ybin == 0])
        rho = spearmanr(ld, y).correlation
        # linear-probe R^2: how much of the continuous parameter c-bar encodes
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        r2 = cross_val_score(ridge, cond_bar, y, cv=KFold(5, shuffle=True, random_state=SEED),
                             scoring="r2")
        rows.append(dict(param=p, lda_cv_acc=acc.mean(), lda_cv_acc_std=acc.std(),
                         cohens_d=abs(d), spearman_lda_vs_param=rho,
                         linearprobe_R2=r2.mean(), linearprobe_R2_std=r2.std()))
        emit(f"\n{p}:")
        emit(f"  LDA high/low from embedding: 5-fold CV acc = {acc.mean():.3f} +/- {acc.std():.3f}")
        emit(f"  |Cohen's d| along LDA axis  = {abs(d):.2f}")
        emit(f"  Spearman(LDA coord, param)  = {rho:+.3f}")
        emit(f"  linear-probe R^2 (Ridge,5f) = {r2.mean():.3f} +/- {r2.std():.3f}")
        sc = ax.scatter(ld, y, c=df["coherence_pct"].values, s=6, cmap="viridis",
                        norm=matplotlib.colors.LogNorm())
        ax.set_xlabel("supervised LDA coordinate"); ax.set_ylabel(p); ax.set_title(p)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist()); cbar.set_label("coherence (%)")
    fig.savefig(f"{OUT}/mechanism_dr_lda.png", dpi=130); plt.close(fig)
    stats = pd.DataFrame(rows)
    stats.to_csv(f"{OUT}/mechanism_dr_stats.csv", index=False)
    return stats


# ---------------------------------------------------------------------------
# 5. Approach 3 — tercile LDA of the conditioning vector (LDA.pdf)
# ---------------------------------------------------------------------------
# Complements approach2, which uses a *binary* high/low split and so yields a
# single LDA axis. Splitting each parameter into TERCILES (low / mid / high)
# makes it a 3-class problem, which yields TWO discriminant axes (LD1, LD2). The
# LDA is fitted on the standardised time-averaged conditioning vector c-bar
# (R^32) and LD1 is plotted against LD2, coloured by tercile. Drift and
# diffusion enter by MAGNITUDE (||f||, ||g||), not by contrast.

LDA_PARAMS = [
    ("drift_mag",     r"drift magnitude $\|f\|$"),
    ("diffusion_mag", r"diffusion magnitude $\|g\|$"),
    ("threshold",     r"threshold $\theta$"),
]
LEVEL_NAMES = ["low", "mid", "high"]
LEVEL_COLORS = ["#2c7fb8", "#bdbdbd", "#d95f0e"]   # low=blue, mid=grey, high=orange


def terciles(y):
    q1, q2 = np.quantile(y, [1 / 3, 2 / 3])
    return np.digitize(y, [q1, q2])                # 0=low, 1=mid, 2=high


def plot_lda2d(df, cond_bar, save_path=None, stats_path=None):
    """``LDA.pdf``: one panel per learned parameter.

    Returns ``(fig, stats)`` where ``stats`` holds the 3-class cross-validated
    accuracy and the variance carried by each discriminant axis. ``save_path``
    defaults to ``outputs/rdm/LDA.pdf``; a ``.png`` twin is written beside it.
    ``stats_path`` defaults to ``outputs/rdm/mechanism_lda2d_stats.csv``.
    """
    Xs = StandardScaler().fit_transform(cond_bar)  # standardised c-bar, R^32

    fig, axes = plt.subplots(1, len(LDA_PARAMS), figsize=(15, 4.6))
    rows = []
    for ax, (p, nice) in zip(axes, LDA_PARAMS):
        y = df[p].values
        lab = terciles(y)

        # 3-class LDA -> two discriminant axes (LD1, LD2)
        lda = LinearDiscriminantAnalysis(n_components=2).fit(Xs, lab)
        Z = lda.transform(Xs)                       # [N, 2]
        evr = lda.explained_variance_ratio_

        # 5-fold CV accuracy of the 3-class classifier from the embedding
        pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        acc = cross_val_score(pipe, cond_bar, lab, cv=cv, scoring="accuracy")

        rows.append(dict(param=p, cv_acc=acc.mean(), cv_acc_std=acc.std(),
                         chance=1 / 3, evr_LD1=evr[0], evr_LD2=evr[1]))

        for k in range(3):
            m = lab == k
            ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.45,
                       color=LEVEL_COLORS[k], label=LEVEL_NAMES[k], linewidths=0)
        ax.set_xlabel("LD1"); ax.set_ylabel("LD2")
        ax.set_title(f"{nice}\n3-class CV acc = {acc.mean():.2f} (chance 0.33)")
        ax.legend(title=p, markerscale=2.5, framealpha=0.9, loc="best")

    fig.tight_layout()
    save_path = os.path.join(OUT, "LDA.pdf") if save_path is None else str(save_path)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    # Dropping /CreationDate keeps the PDF deterministic; the PNG backend
    # stamps no date, so it needs nothing.
    fig.savefig(save_path, metadata={"CreationDate": None})
    fig.savefig(os.path.splitext(save_path)[0] + ".png", dpi=140)
    print(f"[saved] {display_path(save_path)}")

    stats = pd.DataFrame(rows)
    stats_path = (os.path.join(OUT, "mechanism_lda2d_stats.csv")
                  if stats_path is None else str(stats_path))
    stats.to_csv(stats_path, index=False)
    print(stats.round(4).to_string(index=False))
    return fig, stats


# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    emit(f"device={DEVICE}  model={display_path(MODEL_PATH)}")
    model = load_model()
    test_ds = load_test_set()
    emit(f"test trials: {len(test_ds.df)}")
    df, cond_bar = extract_parameters(model, test_ds)
    df.to_csv(f"{OUT}/mechanism_per_trial.csv", index=False)
    approach1(df)
    approach2(df, cond_bar)
    fig, _ = plot_lda2d(df, cond_bar)
    plt.close(fig)
    with open(f"{OUT}/mechanism_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\n[saved] {display_path(OUT)}/mechanism_summary.txt + mechanism_*.csv + LDA.pdf")


if __name__ == "__main__":
    main()
