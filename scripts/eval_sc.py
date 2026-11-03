"""
scripts/eval_sc.py
==================
SC-specific evaluation for the Neural Race Model (NRM).

The natural grouping variable for SC is the *naturalness probability*
(prob ∈ {0.2, 0.4, 0.6, 0.8, 1.0}) derived from the stimulus folder name.
At each prob level there are two folder types: natural_XX (gt=0) and
manmade_XX (gt=1), both with the same visual ambiguity.

Implements
----------
1. SCEvaluator.collect_predictions()  — forward pass on test loader
2. SCEvaluator.compute_metrics()      — per-prob + global metrics
3. Plot suite:
     a. psychometric_chronometric.pdf
     b. rt_distribution_comparison_prob_*.pdf
     c. rt_likelihood_comparison_prob_*.pdf
     d. signed_rt_distributions.pdf
     e. summary_overview.pdf
4. SCEvaluator.print_paper_table()
5. run_sc_analysis()  — the same analysis on ModelEvaluator predictions

Metrics computed
----------------
Global (one value per model):
  Accuracy, r_acc (Pearson), r_rt (Pearson), CCC_rt, Cohen's kappa,
  Macro AUC, NLL (KDE of model RTs evaluated at human RTs), HH NLL (LOTO)

Per-prob (one value per level):
  Same metric set as global.
  Human Noise Ceiling (NC): computed globally AND per-prob.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

import sys as _sys_for_backend

import matplotlib
# Force the headless backend only when nothing has set one up yet, i.e. when
# this module is imported by a script. A notebook keeps its inline backend, so
# plt.show() still renders there.
if "matplotlib.pyplot" not in _sys_for_backend.modules:
    matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, wasserstein_distance

from metrics.point_metrics import safe_auc, safe_kappa, safe_mean
from metrics.relational_metrics import safe_corr, split_half_rt_nc
from metrics.rt_likelihood import kde_nll, loto_nll
from metrics.plots_sc import SCPlotsMixin
from scripts.results_io import display_path

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Constants ─────────────────────────────────────────────────────────────────

PROB_LIST = [0.2, 0.4, 0.6, 0.8, 1.0]

# RTs are clipped to this window (s) before every KDE fit and evaluation.
NLL_RT_RANGE = (0.001, 2.0)


def compute_human_nc_global(
    human_choices: np.ndarray,
    true_labels:   np.ndarray,
    human_rts:     np.ndarray,
    prob_values:   np.ndarray,
    prob_list:     list[float],
) -> dict:
    """
    Global human noise ceiling:
      * NLL NC   : leave-one-trial-out NLL on all human RTs
      * r_RT NC  : split-half Pearson r across prob levels
      * Acc NC   : human accuracy (how often humans match ground truth)
      * Kappa NC : Cohen's kappa of human choices vs ground truth
      * AUC NC   : macro AUC of human choices as soft classifier
    """
    nll_nc    = loto_nll(human_rts, rt_range=NLL_RT_RANGE)
    r_rt_nc, _ = split_half_rt_nc(prob_values, human_rts, prob_list)

    acc_nc   = float((human_choices == true_labels).mean())
    kappa_nc = safe_kappa(true_labels, human_choices)

    # For AUC, treat human choice as a "probability" (0 or 1)
    auc_nc = safe_auc(true_labels, human_choices.astype(float))

    # r_acc NC: split-half per-prob human accuracy correlation
    rng   = np.random.default_rng(42)
    n     = len(human_choices)
    r_acc_splits: list[float] = []
    for _ in range(100):
        idx    = rng.permutation(n)
        half   = n // 2
        ma     = np.zeros(n, dtype=bool); ma[idx[:half]] = True
        mb     = ~ma
        acc_a, acc_b = [], []
        for prob in prob_list:
            m      = prob_values == prob
            ca     = human_choices[m & ma]; la = true_labels[m & ma]
            cb     = human_choices[m & mb]; lb = true_labels[m & mb]
            if len(la) > 0 and len(lb) > 0:
                acc_a.append(float((ca == la).mean()))
                acc_b.append(float((cb == lb).mean()))
        if len(acc_a) >= 3:
            pr, _, _ = safe_corr(acc_a, acc_b)
            if np.isfinite(pr):
                r_acc_splits.append(pr)
    r_acc_nc = float(np.mean(r_acc_splits)) if r_acc_splits else float("nan")

    return {
        "acc":      acc_nc,
        "nll":      nll_nc,
        "r_rt":     r_rt_nc,
        "r_acc":    r_acc_nc,
        "kappa":    kappa_nc,
        "auc":      auc_nc,
    }


def compute_human_nc_per_prob(
    human_choices: np.ndarray,
    true_labels:   np.ndarray,
    human_rts:     np.ndarray,
    prob_values:   np.ndarray,
    prob_list:     list[float],
) -> dict[float, dict]:
    """
    Per-prob human noise ceiling.
    For each prob level: LOO-NLL NC, human accuracy NC, kappa NC.
    (r_rt NC is not meaningful within a single prob level — only across levels.)
    """
    nc_per_prob: dict[float, dict] = {}
    for prob in prob_list:
        mask   = prob_values == prob
        h_rts  = human_rts[mask]
        h_ch   = human_choices[mask]
        t_lab  = true_labels[mask]

        nll_nc   = loto_nll(h_rts, rt_range=NLL_RT_RANGE)
        acc_nc   = float((h_ch == t_lab).mean()) if len(t_lab) > 0 else float("nan")
        kappa_nc = safe_kappa(t_lab, h_ch)
        auc_nc   = safe_auc(t_lab, h_ch.astype(float))

        nc_per_prob[prob] = {
            "nll":   nll_nc,
            "acc":   acc_nc,
            "kappa": kappa_nc,
            "auc":   auc_nc,
        }
    return nc_per_prob


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluator class
# ══════════════════════════════════════════════════════════════════════════════

class SCEvaluator(SCPlotsMixin):
    """
    End-to-end SC evaluator for NRM (NRM).

    Parameters
    ----------
    model      : trained NRM
    device     : torch.device
    n_choices  : 1 for single-accumulator binary task
    prob_list  : ordered naturalness probability levels (default ascending)
    """

    def __init__(
        self,
        model,
        device: torch.device,
        n_choices: int = 1,
        prob_list: list[float] | None = None,
    ):
        self.model     = model
        self.device    = device
        self.n_choices = n_choices
        self.prob_list = prob_list or PROB_LIST

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Collect predictions
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect_predictions(
        self,
        test_loader,
        n_rt_samples: int = 50,
    ) -> dict:
        """
        Run the model on the test set; return per-trial data and
        per-prob model RT sample pools.

        Returns
        -------
        data : dict with keys
            image_indices  : (N,) int
            true_labels    : (N,) int   — ground-truth (0=natural, 1=manmade)
            human_choices  : (N,) int
            pred_choices   : (N,) int   — majority vote over n_rt_samples
            true_rts       : (N,) float — human RT in seconds
            pred_rts       : (N,) float — model median RT
            prob_values    : (N,) float — naturalness prob per trial
            levels         : list[str]  — prob as string, one per trial

            rt_samples_by_prob  : dict{float → np.ndarray}
            human_rts_by_prob   : dict{float → np.ndarray}
            model_info_by_prob  : dict{float → dict}
        """
        self.model.eval()

        img_idxs, true_labs, hum_ch, pred_ch = [], [], [], []
        true_rt, pred_rt = [], []
        prob_vals: list[float] = []
        levels:   list[str]  = []

        rt_samples_by_prob:     dict[float, list[float]] = defaultdict(list)
        human_rts_by_prob:      dict[float, list[float]] = defaultdict(list)
        trial_pred_rts_by_prob: dict[float, list[float]] = defaultdict(list)
        pred_probs_by_prob:     dict[float, list[float]] = defaultdict(list)
        model_info_by_prob: dict[float, dict] = defaultdict(
            lambda: {"pred_choices": [], "true_labels": [], "human_choices": []}
        )

        print("Collecting SC predictions from test set...")
        n_batches = len(test_loader)

        for b_idx, batch in enumerate(test_loader):
            print(f"  Batch {b_idx+1}/{n_batches}...", end="\r")

            embs       = batch["embeddings"].to(self.device)
            h_choices  = batch["human_choices"].to(self.device)
            t_labels   = batch["choices"].to(self.device)
            h_rts      = batch["rts"].to(self.device)
            b_img_idxs = batch["image_index"].to(self.device)
            b_probs: list[float] = batch["probs"]
            b_levels: list[str]  = batch.get("levels", [f"{p:.1f}" for p in b_probs])

            B = len(h_rts)
            if B < 2:
                continue

            cov = {"img_embedding": embs}

            # Expand batch × n_rt_samples to get individual RT samples
            exp_embs = embs.repeat_interleave(n_rt_samples, dim=0)
            exp_cov  = {"img_embedding": exp_embs}

            _, _, pred_ch_all, pred_rt_all = self.model.simulate_nDDM(
                batch_size=B * n_rt_samples,
                covariates=exp_cov,
                soft=True,
                n_reps=1,
                t=0.993,
            )

            pred_rt_mat = pred_rt_all.view(B, n_rt_samples)   # (B, M)
            pred_ch_mat = pred_ch_all.view(B, n_rt_samples)   # (B, M)

            maj_choices = pred_ch_mat.mode(dim=1).values       # (B,)
            med_rts     = pred_rt_mat.median(dim=1).values     # (B,)

            img_idxs.append(b_img_idxs.cpu())
            true_labs.append(t_labels.cpu())
            hum_ch.append(h_choices.cpu())
            pred_ch.append(maj_choices.cpu())
            true_rt.append(h_rts.cpu())
            pred_rt.append(med_rts.cpu())
            levels.extend(b_levels)
            prob_vals.extend(b_probs)

            pred_rt_np   = pred_rt_mat.cpu().numpy()
            med_rts_np   = med_rts.cpu().numpy()
            pred_prob_np = pred_ch_mat.float().mean(dim=1).cpu().numpy()

            for i, prob in enumerate(b_probs):
                rt_samples_by_prob[prob].extend(pred_rt_np[i].tolist())
                human_rts_by_prob[prob].append(float(h_rts[i].cpu()))
                trial_pred_rts_by_prob[prob].append(float(med_rts_np[i]))
                pred_probs_by_prob[prob].append(float(pred_prob_np[i]))
                model_info_by_prob[prob]["pred_choices"].append(int(maj_choices[i].cpu()))
                model_info_by_prob[prob]["true_labels"].append(int(t_labels[i].cpu()))
                model_info_by_prob[prob]["human_choices"].append(int(h_choices[i].cpu()))

        print()

        data = {
            "image_indices": torch.cat(img_idxs).numpy(),
            "true_labels":   torch.cat(true_labs).numpy(),
            "human_choices": torch.cat(hum_ch).numpy(),
            "pred_choices":  torch.cat(pred_ch).numpy(),
            "true_rts":      torch.cat(true_rt).numpy(),
            "pred_rts":      torch.cat(pred_rt).numpy(),
            "prob_values":   np.array(prob_vals, dtype=float),
            "levels":        levels,
            "rt_samples_by_prob":     {k: np.array(v) for k, v in rt_samples_by_prob.items()},
            "human_rts_by_prob":      {k: np.array(v) for k, v in human_rts_by_prob.items()},
            "trial_pred_rts_by_prob": {k: np.array(v) for k, v in trial_pred_rts_by_prob.items()},
            "pred_probs_by_prob":     {k: np.array(v) for k, v in pred_probs_by_prob.items()},
            "model_info_by_prob":     dict(model_info_by_prob),
        }

        for prob in self.prob_list:
            n_h = len(data["human_rts_by_prob"].get(prob, []))
            n_m = len(data["rt_samples_by_prob"].get(prob, []))
            print(f"  Prob {prob:.1f}  human_trials={n_h:5d}  model_RT_samples={n_m:7d}")

        return data

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Compute metrics
    # ──────────────────────────────────────────────────────────────────────────

    def compute_metrics(self, data: dict) -> dict:
        """
        Compute all metrics including human noise ceiling (global + per-prob).

        Returns
        -------
        metrics : dict
            per_prob : dict{prob → dict}
                n_trials, human_acc, model_acc,
                human_median_rt, model_median_rt,
                w1_ms, nll_model, nll_hh,
                pearson_r, spearman_r, ccc_rt,
                auc, kappa,
                nc : dict (per-prob noise ceiling)
            global : dict
                acc, human_acc, r_acc, r_rt, ccc_rt, hh_r_rt,
                mean_nll, hh_mean_nll, mean_w1_ms, mean_auc, mean_kappa,
                nc : dict (global noise ceiling)
        """
        print("\nComputing SC metrics...")

        per_prob: dict[float, dict] = {}
        accs_model, accs_human     = [], []
        med_rt_model, med_rt_human = [], []
        nlls_model, nlls_hh, w1s   = [], [], []
        aucs, kappas               = [], []

        for prob in self.prob_list:
            info   = data["model_info_by_prob"].get(prob, {})
            h_rts  = data["human_rts_by_prob"].get(prob, np.array([]))
            m_rts  = data["rt_samples_by_prob"].get(prob, np.array([]))

            pred_choices  = np.array(info.get("pred_choices",  []), dtype=int)
            true_labels   = np.array(info.get("true_labels",   []), dtype=int)
            human_choices = np.array(info.get("human_choices", []), dtype=int)
            pred_probs_s  = data["pred_probs_by_prob"].get(prob, np.array([]))

            n = len(true_labels)
            if n == 0:
                continue

            model_acc   = float((pred_choices  == true_labels).mean())
            human_acc_c = float((human_choices == true_labels).mean())

            h_med_rt = float(np.median(h_rts)) if len(h_rts) > 0 else float("nan")
            m_med_rt = float(np.median(m_rts)) if len(m_rts) > 0 else float("nan")

            w1 = float("nan")
            if len(h_rts) > 1 and len(m_rts) > 1:
                w1 = wasserstein_distance(h_rts, m_rts) * 1000.0

            nll_m = kde_nll(eval_rts=h_rts, fit_rts=m_rts, rt_range=NLL_RT_RANGE)
            nll_h = loto_nll(h_rts, rt_range=NLL_RT_RANGE)

            trial_pred = data["trial_pred_rts_by_prob"].get(prob, np.array([]))
            pr, sr, ccc_val = safe_corr(trial_pred, h_rts)

            auc   = safe_auc(true_labels, pred_probs_s)
            kappa = safe_kappa(true_labels, pred_choices)

            per_prob[prob] = {
                "n_trials":        n,
                "human_acc":       human_acc_c,
                "model_acc":       model_acc,
                "human_median_rt": h_med_rt,
                "model_median_rt": m_med_rt,
                "w1_ms":           w1,
                "nll_model":       nll_m,
                "nll_hh":          nll_h,
                "pearson_r":       pr,
                "spearman_r":      sr,
                "ccc_rt":          ccc_val,
                "auc":             auc,
                "kappa":           kappa,
            }

            accs_model.append(model_acc)
            accs_human.append(human_acc_c)
            med_rt_model.append(m_med_rt)
            med_rt_human.append(h_med_rt)
            nlls_model.append(nll_m)
            nlls_hh.append(nll_h)
            w1s.append(w1)
            if np.isfinite(auc):   aucs.append(auc)
            if np.isfinite(kappa): kappas.append(kappa)

            print(
                f"  Prob {prob:.1f}  n={n:5d}  "
                f"h_acc={human_acc_c:.3f}  m_acc={model_acc:.3f}  "
                f"RT_h={h_med_rt*1000:.0f}ms  RT_m={m_med_rt*1000:.0f}ms  "
                f"W1={w1:.1f}ms  NLL={nll_m:.3f}  HH_NLL={nll_h:.3f}  "
                f"r={pr:.3f}  AUC={auc:.3f}  κ={kappa:.3f}"
            )

        # ── Global metrics ────────────────────────────────────────────────────
        def _r(x, y):
            x, y = np.array(x, dtype=float), np.array(y, dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3:
                return float("nan")
            return float(pearsonr(x[ok], y[ok])[0])

        r_acc = _r(accs_model, accs_human)
        r_rt  = _r(med_rt_model, med_rt_human)
        _, _, ccc_rt_global = safe_corr(med_rt_model, med_rt_human)

        all_prob_vals = data["prob_values"]
        all_h_rts     = data["true_rts"]
        hh_r_rt, _    = split_half_rt_nc(all_prob_vals, all_h_rts, self.prob_list)

        model_acc_global = float((data["pred_choices"] == data["true_labels"]).mean())
        human_acc_global = float((data["human_choices"] == data["true_labels"]).mean())

        # ── Human noise ceiling (global) ──────────────────────────────────────
        nc_global = compute_human_nc_global(
            human_choices=data["human_choices"],
            true_labels=data["true_labels"],
            human_rts=data["true_rts"],
            prob_values=all_prob_vals,
            prob_list=self.prob_list,
        )
        # override hh_r_rt with the already-computed split-half value
        nc_global["r_rt"] = hh_r_rt

        # ── Human noise ceiling (per-prob) ────────────────────────────────────
        nc_per_prob = compute_human_nc_per_prob(
            human_choices=data["human_choices"],
            true_labels=data["true_labels"],
            human_rts=data["true_rts"],
            prob_values=all_prob_vals,
            prob_list=self.prob_list,
        )
        for prob in per_prob:
            per_prob[prob]["nc"] = nc_per_prob.get(prob, {})

        glb = {
            "acc":         model_acc_global,
            "human_acc":   human_acc_global,
            "r_acc":       r_acc,
            "r_rt":        r_rt,
            "ccc_rt":      ccc_rt_global,
            "hh_r_rt":     hh_r_rt,
            "mean_nll":    safe_mean(nlls_model),
            "hh_mean_nll": safe_mean(nlls_hh),
            "mean_w1_ms":  safe_mean(w1s),
            "mean_auc":    safe_mean(aucs),
            "mean_kappa":  safe_mean(kappas),
            "per_prob_accs_model": accs_model,
            "per_prob_accs_human": accs_human,
            "per_prob_rt_model":   med_rt_model,
            "per_prob_rt_human":   med_rt_human,
            "probs":               list(per_prob.keys()),
            "nc":                  nc_global,
        }

        return {"per_prob": per_prob, "global": glb}

    # ──────────────────────────────────────────────────────────────────────────
    # 3a. Psychometric + Chronometric curves
    # ──────────────────────────────────────────────────────────────────────────


    # ──────────────────────────────────────────────────────────────────────────
    # 4. Print paper table
    # ──────────────────────────────────────────────────────────────────────────

    def print_paper_table(self, metrics: dict) -> None:
        glb = metrics["global"]
        nc  = glb.get("nc", {})

        def _f(v, fmt=".3f"):
            return "—" if not np.isfinite(v) else format(v, fmt)

        W = 100
        print("\n" + "=" * W)
        print("PAPER TABLE — NRM (NRM) on SC task")
        print("=" * W)
        header = (
            f"{'Model':22s}  "
            f"{'Acc(%)':>7}  "
            f"{'AUC':>6}  "
            f"{'Kappa':>7}  "
            f"{'r_acc':>6}  "
            f"{'r_RT':>6}  "
            f"{'CCC_RT':>7}  "
            f"{'NLL↓':>7}  "
        )
        print(header)
        print("-" * W)
        # Noise ceiling row
        print(
            f"{'Human (NC)':22s}  "
            f"{_f(nc.get('acc', float('nan'))*100, '.1f'):>7}  "
            f"{_f(nc.get('auc', float('nan'))):>6}  "
            f"{_f(nc.get('kappa', float('nan'))):>7}  "
            f"{_f(nc.get('r_acc', float('nan'))):>6}  "
            f"{_f(nc.get('r_rt', float('nan'))):>6}  "
            f"{'—':>7}  "
            f"{_f(glb['hh_mean_nll']):>7}  "
        )
        print("-" * W)
        # NRM row
        print(
            f"{'NRM (ours)':22s}  "
            f"{glb['acc']*100:>7.1f}  "
            f"{_f(glb['mean_auc']):>6}  "
            f"{_f(glb['mean_kappa']):>7}  "
            f"{_f(glb['r_acc']):>6}  "
            f"{_f(glb['r_rt']):>6}  "
            f"{_f(glb['ccc_rt']):>7}  "
            f"{_f(glb['mean_nll']):>7}  "
        )
        print("=" * W)

        print("\nPer-prob breakdown:")
        hdr2 = (
            f"  {'Prob':>5}  {'N':>5}  "
            f"{'Acc_H':>6}  {'Acc_M':>6}  "
            f"{'NC_Acc':>7}  "
            f"{'RT_H ms':>8}  {'RT_M ms':>8}  "
            f"{'W1 ms':>7}  {'NLL_M':>6}  {'NLL_HH':>7}  "
            f"{'NC_NLL':>7}  "
            f"{'r_RT':>6}  {'AUC':>6}  {'Kappa':>7}"
        )
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for prob in self.prob_list:
            if prob not in metrics["per_prob"]:
                continue
            p  = metrics["per_prob"][prob]
            nc_p = p.get("nc", {})
            print(
                f"  {prob:5.1f}  {p['n_trials']:>5d}  "
                f"{p['human_acc']:>6.3f}  {p['model_acc']:>6.3f}  "
                f"{_f(nc_p.get('acc', float('nan'))):>7}  "
                f"{p['human_median_rt']*1000:>8.1f}  "
                f"{p['model_median_rt']*1000:>8.1f}  "
                f"{p['w1_ms']:>7.1f}  "
                f"{_f(p['nll_model']):>6}  "
                f"{_f(p['nll_hh']):>7}  "
                f"{_f(nc_p.get('nll', float('nan'))):>7}  "
                f"{_f(p.get('pearson_r', float('nan'))):>6}  "
                f"{_f(p.get('auc',   float('nan'))):>6}  "
                f"{_f(p.get('kappa', float('nan'))):>7}"
            )
        print("=" * W)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Full evaluate() pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_loader,
        save_dir:     str | Path = "outputs/sc",
        n_rt_samples: int = 50,
    ) -> dict:
        """
        Full evaluation pipeline.

        Parameters
        ----------
        test_loader  : DataLoader for the SC test set
        save_dir     : directory for all outputs
        n_rt_samples : model RT samples per test trial

        Returns
        -------
        dict: {data, metrics, plot_paths, save_dir}
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        data    = self.collect_predictions(test_loader, n_rt_samples=n_rt_samples)
        metrics = self.compute_metrics(data)

        print("\nGenerating SC plots...")
        plot_paths: dict = {}

        plot_paths["psychometric_chronometric"] = self.plot_psychometric_chronometric(
            data, metrics, save_dir
        )
        plot_paths["rt_distributions"] = self.plot_rt_distributions(
            data, metrics, save_dir
        )
        plot_paths["rt_likelihood"] = self.plot_rt_likelihood(
            data, metrics, save_dir
        )
        plot_paths["signed_rt"] = self.plot_signed_rt_histograms(
            data, metrics, save_dir
        )
        plot_paths["summary"] = self.plot_summary_overview(
            data, metrics, save_dir
        )

        # ── Save CSV summaries ────────────────────────────────────────────────
        rows = []
        for prob, p in metrics["per_prob"].items():
            nc_p = p.get("nc", {})
            rows.append({
                "prob":             prob,
                "n_trials":         p["n_trials"],
                "human_acc":        p["human_acc"],
                "model_acc":        p["model_acc"],
                "nc_acc":           nc_p.get("acc",   float("nan")),
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
            save_dir / "sc_metrics_per_prob.csv", index=False
        )

        glb = metrics["global"]
        nc  = glb.get("nc", {})
        pd.DataFrame([{
            "acc":           glb["acc"],
            "human_acc":     glb["human_acc"],
            "r_acc":         glb["r_acc"],
            "r_rt":          glb["r_rt"],
            "ccc_rt":        glb["ccc_rt"],
            "hh_r_rt":       glb["hh_r_rt"],
            "mean_nll":      glb["mean_nll"],
            "hh_mean_nll":   glb["hh_mean_nll"],
            "mean_w1_ms":    glb["mean_w1_ms"],
            "mean_auc":      glb["mean_auc"],
            "mean_kappa":    glb["mean_kappa"],
            "nc_acc":        nc.get("acc",    float("nan")),
            "nc_nll":        nc.get("nll",    float("nan")),
            "nc_r_rt":       nc.get("r_rt",   float("nan")),
            "nc_r_acc":      nc.get("r_acc",  float("nan")),
            "nc_kappa":      nc.get("kappa",  float("nan")),
        }]).to_csv(save_dir / "sc_metrics_global.csv", index=False)

        print(f"\nAll outputs saved to: {display_path(save_dir)}")

        self.print_paper_table(metrics)

        return {
            "data":       data,
            "metrics":    metrics,
            "plot_paths": plot_paths,
            "save_dir":   str(save_dir),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Entry point for ModelEvaluator predictions
# ══════════════════════════════════════════════════════════════════════════════

def run_sc_analysis(
    model,
    device: torch.device,
    data:   dict,
    save_dir: str | Path,
    n_rt_samples: int = 50,
    prob_list: list[float] | None = None,
) -> dict:
    """
    Thin wrapper that runs the SC analysis on ModelEvaluator predictions.

    Parameters
    ----------
    model        : trained NRM
    device       : torch.device
    data         : dict from ModelEvaluator.collect_predictions().
                   Must contain 'probs' (list) or 'levels' (list of prob strings)
                   and 'rt_samples_by_image' (dict).
    save_dir     : output directory
    n_rt_samples : RT samples per trial for NC computation
    prob_list    : override default [0.2, 0.4, 0.6, 0.8, 1.0]

    Returns
    -------
    dict with per_prob and global metrics, and plot paths.
    """
    save_dir  = Path(save_dir)
    prob_list = prob_list or PROB_LIST

    # Resolve per-trial naturalness probabilities from data
    if "probs" in data and len(data["probs"]) > 0:
        prob_values = np.array(data["probs"], dtype=float)
    elif "levels" in data and len(data["levels"]) > 0:
        try:
            prob_values = np.array([float(l) for l in data["levels"]], dtype=float)
        except ValueError:
            print("  [SC] 'levels' does not contain parseable prob values — skipping.")
            return {}
    else:
        print("  [SC] No prob/level information in data — skipping SC analysis.")
        return {}

    # Build per-trial structures from already-collected data
    true_labels   = data["true_choices"]    if "true_choices"   in data else data.get("true_labels")
    human_choices = data["human_choices"]
    pred_choices  = data["pred_choices"]
    true_rts      = data["true_rts"]
    pred_rts      = data["pred_rts"]
    image_indices = data["image_indices"]

    rt_samples_by_image = data.get("rt_samples_by_image", {})

    # Group per prob
    rt_samples_by_prob:     dict[float, list] = defaultdict(list)
    human_rts_by_prob:      dict[float, list] = defaultdict(list)
    trial_pred_rts_by_prob: dict[float, list] = defaultdict(list)
    pred_probs_by_prob:     dict[float, list] = defaultdict(list)
    model_info_by_prob: dict[float, dict] = defaultdict(
        lambda: {"pred_choices": [], "true_labels": [], "human_choices": []}
    )

    for i in range(len(true_labels)):
        prob = float(prob_values[i])
        human_rts_by_prob[prob].append(float(true_rts[i]))
        trial_pred_rts_by_prob[prob].append(float(pred_rts[i]))
        model_info_by_prob[prob]["pred_choices"].append(int(pred_choices[i]))
        model_info_by_prob[prob]["true_labels"].append(int(true_labels[i]))
        model_info_by_prob[prob]["human_choices"].append(int(human_choices[i]))
        # Soft probability: use pred_choices as 0/1 (no soft probs from ModelEvaluator)
        pred_probs_by_prob[prob].append(float(pred_choices[i]))

    # Add model RT samples per prob (from rt_samples_by_image)
    img_to_prob = {int(image_indices[i]): float(prob_values[i])
                   for i in range(len(image_indices))}
    for img_idx, samples in rt_samples_by_image.items():
        prob = img_to_prob.get(int(img_idx))
        if prob is not None:
            rt_samples_by_prob[prob].extend(samples)

    collected = {
        "image_indices": image_indices,
        "true_labels":   true_labels,
        "human_choices": human_choices,
        "pred_choices":  pred_choices,
        "true_rts":      true_rts,
        "pred_rts":      pred_rts,
        "prob_values":   prob_values,
        "levels":        data.get("levels", [f"{p:.1f}" for p in prob_values]),
        "rt_samples_by_prob":     {k: np.array(v) for k, v in rt_samples_by_prob.items()},
        "human_rts_by_prob":      {k: np.array(v) for k, v in human_rts_by_prob.items()},
        "trial_pred_rts_by_prob": {k: np.array(v) for k, v in trial_pred_rts_by_prob.items()},
        "pred_probs_by_prob":     {k: np.array(v) for k, v in pred_probs_by_prob.items()},
        "model_info_by_prob":     dict(model_info_by_prob),
    }

    evaluator = SCEvaluator(model=model, device=device, n_choices=1,
                               prob_list=prob_list)
    metrics = evaluator.compute_metrics(collected)

    print("\nGenerating SC plots...")
    plot_paths: dict = {}
    plot_paths["sc_psychometric"] = evaluator.plot_psychometric_chronometric(
        collected, metrics, save_dir
    )
    plot_paths["sc_summary"] = evaluator.plot_summary_overview(
        collected, metrics, save_dir
    )

    evaluator.print_paper_table(metrics)

    return {"data": collected, "metrics": metrics, "paths": plot_paths}
