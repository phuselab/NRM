"""
scripts/eval_rdm.py
===================
RDM-specific evaluation for the Neural Race Model (NRM).

The key design decision: for RDM, the natural unit of analysis is the
*coherence level* (7 levels, 0.8%–51.2%), not individual video stimuli.
Each video is a unique synthetic trial → per-image grouping gives n=1
per stimulus, making HH-LOSO and distribution comparisons degenerate.

Implements
----------
1. collect_predictions()  — forward pass on test loader, groups by coherence
2. compute_metrics()      — per-coherence + global metrics for the paper table
3. Five plot types:
      a. psychometric_chronometric.pdf       — acc + RT vs coherence
      b. rt_distribution_comparison_coh_*.pdf — 7 separate distribution plots
      c. rt_likelihood_comparison_coh_*.pdf   — 7 separate likelihood plots
      d. signed_rt_distributions.pdf          — 7-panel signed-RT (RTify style)
      e. summary_overview.pdf                 — combined overview
4. print_paper_table()    — prints metrics for the LaTeX table

Table columns produced
----------------------
    Choice:   Acc (%), r_acc
    RT dist:  NLL ↓, r_RT ↑

NLL  = negative log-likelihood of human test RTs under model RT density
       (KDE fitted to model RT samples; lower = model closer to humans)
HH NLL = same using leave-one-trial-out human density (noise ceiling)
W1   = Wasserstein-1 distance (ms) between model and human RT distributions
r_RT = Pearson r between per-coherence human median RT and model median RT
r_acc= Pearson r between per-coherence human accuracy and model accuracy
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import List

import sys as _sys_for_backend

import matplotlib
# Force the headless backend only when nothing has set one up yet, i.e. when
# this module is imported by a script. A notebook keeps its inline backend, so
# plt.show() still renders there.
if "matplotlib.pyplot" not in _sys_for_backend.modules:
    matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False   # avoid LaTeX renderer for special chars
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, wasserstein_distance

from metrics.point_metrics import safe_auc, safe_kappa
from metrics.relational_metrics import safe_corr, split_half_rt_nc
from metrics.rt_likelihood import kde_nll, loto_nll
from metrics.plots_rdm import RDMPlotsMixin
from scripts.results_io import display_path

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Constants ─────────────────────────────────────────────────────────────────

COHERENCE_LIST = [0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2]   # ascending

# RTs are clipped to this window (s) before every KDE fit and evaluation.
NLL_RT_RANGE = (0.01, 5.0)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: parse coherence string
# ══════════════════════════════════════════════════════════════════════════════

def _parse_coh(level_str: str) -> float:
    """'12.8%' → 12.8"""
    return float(level_str.replace("%", "").strip())


class RDMEvaluator(RDMPlotsMixin):
    """
    End-to-end RDM evaluator for NRM (NRM).

    Parameters
    ----------
    model      : trained NRM / TimeEmbeddedNRM
    device     : torch.device
    n_choices  : number of decision alternatives (default 2)
    coherence_list : ordered list of coherence % values (default ascending)
    """

    def __init__(
        self,
        model,
        device: torch.device,
        n_choices: int = 2,
        coherence_list: List[float] | None = None,
    ):
        self.model          = model
        self.device         = device
        self.n_choices      = n_choices
        self.coherence_list = coherence_list or COHERENCE_LIST

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
        per-coherence model RT sample pools.

        Returns
        -------
        data : dict with keys
            image_indices  : (N,) int   — video_index per trial
            true_labels    : (N,) int   — ground-truth direction (0/1)
            human_choices  : (N,) int   — what the human said
            pred_choices   : (N,) int   — model's majority choice
            true_rts       : (N,) float — human RT in seconds
            pred_rts       : (N,) float — model median RT (over n_rt_samples)
            levels         : list[str]  — coherence strings, one per trial
            coh_values     : (N,) float — coherence in % (parsed from levels)

            rt_samples_by_coh : dict{float coh → list[float]}
                                  all model RT samples (n_rt_samples per trial)
                                  pooled over all test trials at that coherence
            human_rts_by_coh  : dict{float coh → np.ndarray}
                                  all human RTs at that coherence
            model_info_by_coh : dict{float coh → dict}
                                  lists for per-coherence pred/true/human choices
        """
        self.model.eval()

        # Accumulators
        img_idxs, true_labs, hum_ch, pred_ch, true_rt, pred_rt = [], [], [], [], [], []
        coh_vals: list[float] = []
        levels:   list[str]  = []

        rt_samples_by_coh:    dict[float, list[float]] = defaultdict(list)
        human_rts_by_coh:     dict[float, list[float]] = defaultdict(list)
        trial_pred_rts_by_coh: dict[float, list[float]] = defaultdict(list)
        pred_probs_by_coh:    dict[float, list[float]] = defaultdict(list)
        sub_ids_by_coh:       dict[float, list[str]]   = defaultdict(list)
        model_info_by_coh:    dict[float, dict]        = defaultdict(
            lambda: {"pred_choices": [], "true_labels": [], "human_choices": []}
        )

        print("Collecting RDM predictions from test set...")
        n_batches = len(test_loader)

        for b_idx, batch in enumerate(test_loader):
            print(f"  Batch {b_idx+1}/{n_batches}...", end="\r")

            embs         = batch["embeddings"].to(self.device)
            h_choices    = batch["human_choices"].to(self.device)
            t_labels     = batch["choices"].to(self.device)
            h_rts        = batch["rts"].to(self.device)
            b_img_idxs   = batch["image_index"].to(self.device)
            b_levels: list[str] = batch["levels"]
            b_sub_ids: list[str] = batch.get("sub_ids", ["unknown"] * len(b_levels))

            B = len(h_rts)
            if B < 2:
                continue

            cov = {"img_embedding": embs}

            # ── Expand batch × n_rt_samples → get individual RT samples ──────
            exp_embs = embs.repeat_interleave(n_rt_samples, dim=0)  # (B*M, T, D)
            exp_cov  = {"img_embedding": exp_embs}

            _, _, pred_ch_all, pred_rt_all = self.model.simulate_nDDM(
                batch_size=B * n_rt_samples,
                covariates=exp_cov,
                soft=True,
                n_reps=1,
                t=0.993,
            )
            # pred_ch_all, pred_rt_all: (B*M,)
            pred_rt_mat  = pred_rt_all.view(B, n_rt_samples)   # (B, M)
            pred_ch_mat  = pred_ch_all.view(B, n_rt_samples)   # (B, M)

            # Aggregate: majority vote for choice, median for RT
            maj_choices  = pred_ch_mat.mode(dim=1).values      # (B,)
            med_rts      = pred_rt_mat.median(dim=1).values    # (B,)

            # ── Store per-trial data ──────────────────────────────────────────
            img_idxs.append(b_img_idxs.cpu())
            true_labs.append(t_labels.cpu())
            hum_ch.append(h_choices.cpu())
            pred_ch.append(maj_choices.cpu())
            true_rt.append(h_rts.cpu())
            pred_rt.append(med_rts.cpu())
            levels.extend(b_levels)
            b_cohs = [_parse_coh(s) for s in b_levels]
            coh_vals.extend(b_cohs)

            # ── Group into per-coherence pools ────────────────────────────────
            pred_rt_np = pred_rt_mat.cpu().numpy()   # (B, M)
            med_rts_np = med_rts.cpu().numpy()   # (B,)
            # Soft probability: fraction of RT samples voting class 1
            pred_prob_np = pred_ch_mat.float().mean(dim=1).cpu().numpy()  # (B,)

            for i, coh in enumerate(b_cohs):
                rt_samples_by_coh[coh].extend(pred_rt_np[i].tolist())
                human_rts_by_coh[coh].append(float(h_rts[i].cpu()))
                trial_pred_rts_by_coh[coh].append(float(med_rts_np[i]))
                pred_probs_by_coh[coh].append(float(pred_prob_np[i]))
                sub_ids_by_coh[coh].append(b_sub_ids[i])
                model_info_by_coh[coh]["pred_choices"].append(int(maj_choices[i].cpu()))
                model_info_by_coh[coh]["true_labels"].append(int(t_labels[i].cpu()))
                model_info_by_coh[coh]["human_choices"].append(int(h_choices[i].cpu()))

        print()

        data = {
            "image_indices": torch.cat(img_idxs).numpy(),
            "true_labels":   torch.cat(true_labs).numpy(),
            "human_choices": torch.cat(hum_ch).numpy(),
            "pred_choices":  torch.cat(pred_ch).numpy(),
            "true_rts":      torch.cat(true_rt).numpy(),
            "pred_rts":      torch.cat(pred_rt).numpy(),
            "levels":        levels,
            "coh_values":    np.array(coh_vals, dtype=float),
            "rt_samples_by_coh":     {k: np.array(v) for k, v in rt_samples_by_coh.items()},
            "human_rts_by_coh":      {k: np.array(v) for k, v in human_rts_by_coh.items()},
            "trial_pred_rts_by_coh": {k: np.array(v) for k, v in trial_pred_rts_by_coh.items()},
            "pred_probs_by_coh":     {k: np.array(v) for k, v in pred_probs_by_coh.items()},
            "sub_ids_by_coh":        dict(sub_ids_by_coh),
            "model_info_by_coh":     dict(model_info_by_coh),
        }

        for coh in self.coherence_list:
            n_h = len(data["human_rts_by_coh"].get(coh, []))
            n_m = len(data["rt_samples_by_coh"].get(coh, []))
            print(f"  Coh {coh:5.1f}%  human_trials={n_h:5d}  model_RT_samples={n_m:7d}")

        return data

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Compute metrics
    # ──────────────────────────────────────────────────────────────────────────

    def compute_metrics(self, data: dict) -> dict:
        """
        Compute all metrics for the paper table.

        Ambiguity resolutions:
          1B  – Choice accuracy aggregated per coherence, no LOSO.
          2B  – NC for r_RT via split-half across trials (not LOSO-subject).
          3   – HH NLL via leave-one-trial-out (loto_nll), not LOSO-subject.

        Returns
        -------
        metrics : dict
            per_coherence : dict{coh_pct → dict}
                n_trials, human_acc, model_acc,
                human_median_rt, model_median_rt,
                w1_ms, nll_model, nll_hh,
                pearson_r, spearman_r, ccc,
                auc, kappa
            global : dict
                acc, human_acc, r_acc, r_rt, hh_r_rt,
                mean_nll, hh_mean_nll, mean_w1_ms,
                mean_auc, mean_kappa,
                human_human_rt_corr
        """
        print("\nComputing RDM metrics (1B / 2B / LOTO-NLL)...")

        per_coh: dict[float, dict] = {}

        accs_model, accs_human  = [], []
        med_rt_model, med_rt_human = [], []
        nlls_model, nlls_hh, w1s = [], [], []
        aucs, kappas = [], []

        for coh in self.coherence_list:
            info   = data["model_info_by_coh"].get(coh, {})
            h_rts  = data["human_rts_by_coh"].get(coh, np.array([]))
            m_rts  = data["rt_samples_by_coh"].get(coh, np.array([]))

            pred_choices  = np.array(info.get("pred_choices",  []), dtype=int)
            true_labels   = np.array(info.get("true_labels",   []), dtype=int)
            human_choices = np.array(info.get("human_choices", []), dtype=int)
            pred_probs    = data["pred_probs_by_coh"].get(coh, np.array([]))

            n = len(true_labels)
            if n == 0:
                continue

            # ── 1B: accuracy per coherence, no LOSO ──────────────────────────
            model_acc   = float((pred_choices  == true_labels).mean())
            human_acc_c = float((human_choices == true_labels).mean())

            h_med_rt = float(np.median(h_rts)) if len(h_rts) > 0 else float("nan")
            m_med_rt = float(np.median(m_rts)) if len(m_rts) > 0 else float("nan")

            # Wasserstein-1 (seconds → ms)
            w1 = float("nan")
            if len(h_rts) > 1 and len(m_rts) > 1:
                w1 = wasserstein_distance(h_rts, m_rts) * 1000.0

            # ── 3: NLL via leave-one-trial-out ────────────────────────────────
            nll_m = kde_nll(eval_rts=h_rts, fit_rts=m_rts, rt_range=NLL_RT_RANGE)
            nll_h = loto_nll(h_rts, rt_range=NLL_RT_RANGE)

            # Per-trial RT correlation
            trial_pred = data["trial_pred_rts_by_coh"].get(coh, np.array([]))
            pr, sr, ccc_val = safe_corr(trial_pred, h_rts)

            # ── Macro AUC and Cohen's Kappa per coherence ─────────────────────
            auc   = safe_auc(true_labels, pred_probs)
            kappa = safe_kappa(true_labels, pred_choices)

            per_coh[coh] = {
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
                "ccc":             ccc_val,
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
                f"  Coh {coh:5.1f}%  n={n:5d}  "
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

        def _safe_mean(lst):
            v = [x for x in lst if np.isfinite(x)]
            return float(np.mean(v)) if v else float("nan")

        r_acc = _r(accs_model, accs_human)
        r_rt  = _r(med_rt_model, med_rt_human)

        # ── 2B: NC for r_RT via split-half across trials ──────────────────────
        all_coh_vals = data["coh_values"]
        all_h_rts    = data["true_rts"]
        hh_r_rt, hh_rho_rt = split_half_rt_nc(
            all_coh_vals, all_h_rts, self.coherence_list
        )

        model_acc_global = float((data["pred_choices"] == data["true_labels"]).mean())
        human_acc_global = float((data["human_choices"] == data["true_labels"]).mean())

        rt_corr = float("nan")
        if len(data["pred_rts"]) > 2:
            try:
                rt_corr = float(pearsonr(data["true_rts"], data["pred_rts"])[0])
            except Exception:
                pass

        glb = {
            "acc":                  model_acc_global,
            "human_acc":            human_acc_global,
            "r_acc":                r_acc,
            "r_rt":                 r_rt,
            "hh_r_rt":              hh_r_rt,       # 2B: split-half NC for r_RT
            "mean_nll":             _safe_mean(nlls_model),
            "hh_mean_nll":          _safe_mean(nlls_hh),
            "mean_w1_ms":           _safe_mean(w1s),
            "mean_auc":             _safe_mean(aucs),
            "mean_kappa":           _safe_mean(kappas),
            "human_human_rt_corr":  rt_corr,
            "per_coh_accs_model":   accs_model,
            "per_coh_accs_human":   accs_human,
            "per_coh_rt_model":     med_rt_model,
            "per_coh_rt_human":     med_rt_human,
            "coherences":           list(per_coh.keys()),
        }

        return {"per_coherence": per_coh, "global": glb}

    # ──────────────────────────────────────────────────────────────────────────
    # 3a. Psychometric + Chronometric curves
    # ──────────────────────────────────────────────────────────────────────────


    # ──────────────────────────────────────────────────────────────────────────
    # 4. Print paper table
    # ──────────────────────────────────────────────────────────────────────────

    def print_paper_table(self, metrics: dict) -> None:
        """
        Print metrics formatted for the paper table.

        Columns: Acc (%) | Macro AUC | κ | r_acc | NLL ↓ | r_RT ↑
        Noise ceiling row: HH_NLL (LOTO) | hh_r_RT (split-half 2B)
        """
        glb = metrics["global"]

        def _f(v, fmt=".3f"):
            return "—" if not np.isfinite(v) else format(v, fmt)

        W = 100
        print("\n" + "=" * W)
        print("PAPER TABLE — NRM (NRM) on RDM task")
        print(f"  Ambiguities: 1B (acc per-coh, no LOSO) | "
              f"2B (r_RT NC = split-half across trials) | "
              f"3 (NLL NC = LOTO)")
        print("=" * W)
        header = (
            f"{'Model':22s}  "
            f"{'Acc(%)':>7}  "
            f"{'AUC':>6}  "
            f"{'Kappa':>7}  "
            f"{'r_acc':>6}  "
            f"{'NLL↓':>7}  "
            f"{'r_RT↑':>6}  "
        )
        print(header)
        print("-" * W)
        # Noise ceiling row
        print(
            f"{'Human (NC)':22s}  "
            f"{'—':>7}  "
            f"{'—':>6}  "
            f"{'—':>7}  "
            f"{'—':>6}  "
            f"{_f(glb['hh_mean_nll']):>7}  "
            f"{_f(glb['hh_r_rt']):>6}  "
        )
        print("-" * W)
        # NRM row
        print(
            f"{'NRM (ours)':22s}  "
            f"{glb['acc']*100:>7.1f}  "
            f"{_f(glb['mean_auc']):>6}  "
            f"{_f(glb['mean_kappa']):>7}  "
            f"{_f(glb['r_acc']):>6}  "
            f"{_f(glb['mean_nll']):>7}  "
            f"{_f(glb['r_rt']):>6}  "
        )
        print("=" * W)

        print("\nPer-coherence breakdown:")
        hdr2 = (
            f"  {'Coh%':>6}  {'N':>5}  "
            f"{'Acc_H':>6}  {'Acc_M':>6}  "
            f"{'RT_H ms':>8}  {'RT_M ms':>8}  "
            f"{'W1 ms':>7}  {'NLL_M':>6}  {'NLL_HH':>7}  "
            f"{'r_RT':>6}  {'AUC':>6}  {'Kappa':>7}"
        )
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for coh in self.coherence_list:
            if coh not in metrics["per_coherence"]:
                continue
            p = metrics["per_coherence"][coh]
            print(
                f"  {coh:6.1f}  {p['n_trials']:>5d}  "
                f"{p['human_acc']:>6.3f}  {p['model_acc']:>6.3f}  "
                f"{p['human_median_rt']*1000:>8.1f}  "
                f"{p['model_median_rt']*1000:>8.1f}  "
                f"{p['w1_ms']:>7.1f}  "
                f"{_f(p['nll_model']):>6}  "
                f"{_f(p['nll_hh']):>7}  "
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
        n_reps:        int = 1,
        save_dir:      str | Path = "outputs_rdm",
        n_rt_samples:  int = 50,
        rt_ll_method:  str = "kde",
    ) -> dict:
        """
        Full evaluation pipeline.

        Parameters
        ----------
        test_loader   : DataLoader for the test set (RDM dataset)
        n_reps        : (unused, kept for API compatibility)
        save_dir      : directory for all outputs
        n_rt_samples  : model RT samples per test trial
        rt_ll_method  : "kde" (default, non-parametric)

        Returns
        -------
        dict: {data, metrics, plot_paths, save_dir}
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: collect ───────────────────────────────────────────────────
        data = self.collect_predictions(test_loader, n_rt_samples=n_rt_samples)

        # ── Step 2: metrics ───────────────────────────────────────────────────
        metrics = self.compute_metrics(data)

        # ── Step 3: plots ─────────────────────────────────────────────────────
        print("\nGenerating RDM plots...")
        plot_paths: dict[str, Path | list[Path]] = {}

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

        # ── Step 4: save CSV summary ──────────────────────────────────────────
        rows = []
        for coh, p in metrics["per_coherence"].items():
            rows.append({
                "coherence_pct":   coh,
                "n_trials":        p["n_trials"],
                "human_acc":       p["human_acc"],
                "model_acc":       p["model_acc"],
                "human_median_rt": p["human_median_rt"],
                "model_median_rt": p["model_median_rt"],
                "w1_ms":           p["w1_ms"],
                "nll_model":       p["nll_model"],
                "nll_hh":          p["nll_hh"],
                "pearson_r":       p.get("pearson_r",  float("nan")),
                "spearman_r":      p.get("spearman_r", float("nan")),
                "ccc":             p.get("ccc",        float("nan")),
                "auc":             p.get("auc",        float("nan")),
                "kappa":           p.get("kappa",      float("nan")),
            })
        pd.DataFrame(rows).to_csv(save_dir / "rdm_metrics_per_coherence.csv", index=False)

        glb = metrics["global"]
        pd.DataFrame([{
            "acc":          glb["acc"],
            "human_acc":    glb["human_acc"],
            "r_acc":        glb["r_acc"],
            "r_rt":         glb["r_rt"],
            "hh_r_rt":      glb["hh_r_rt"],
            "mean_nll":     glb["mean_nll"],
            "hh_mean_nll":  glb["hh_mean_nll"],
            "mean_w1_ms":   glb["mean_w1_ms"],
            "mean_auc":     glb["mean_auc"],
            "mean_kappa":   glb["mean_kappa"],
        }]).to_csv(save_dir / "rdm_metrics_global.csv", index=False)

        print(f"\nAll outputs saved to: {display_path(save_dir)}")

        # ── Step 5: print table ───────────────────────────────────────────────
        self.print_paper_table(metrics)

        return {
            "data":       data,
            "metrics":    metrics,
            "plot_paths": plot_paths,
            "save_dir":   str(save_dir),
        }
