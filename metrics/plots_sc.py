"""Per-discriminability diagnostic figures for the SC evaluator (§3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

_PROB_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


class SCPlotsMixin:
    """Plot methods of :class:`scripts.eval_sc.SCEvaluator`."""

    def plot_psychometric_chronometric(
        self, data: dict, metrics: dict, save_dir: str | Path
    ) -> Path:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        per_prob = metrics["per_prob"]
        probs    = sorted(per_prob.keys())

        h_accs = [per_prob[p]["human_acc"]       for p in probs]
        m_accs = [per_prob[p]["model_acc"]        for p in probs]
        h_rts  = [per_prob[p]["human_median_rt"]  for p in probs]
        m_rts  = [per_prob[p]["model_median_rt"]  for p in probs]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("NRM — SC Psychometric and Chronometric Curves",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(probs, h_accs, "o-", color="#2c7bb6", label="Human", lw=2, ms=7)
        ax.plot(probs, m_accs, "s--", color="#d7191c", label="NRM",   lw=2, ms=7)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlabel("P(natural stimulus)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Psychometric Curve")
        ax.set_ylim(0.3, 1.05)
        ax.set_xticks(probs)
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(probs, [r * 1000 for r in h_rts], "o-", color="#2c7bb6",
                label="Human", lw=2, ms=7)
        ax.plot(probs, [r * 1000 for r in m_rts], "s--", color="#d7191c",
                label="NRM",   lw=2, ms=7)
        ax.set_xlabel("P(natural stimulus)")
        ax.set_ylabel("Median RT (ms)")
        ax.set_title("Chronometric Curve")
        ax.set_xticks(probs)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out = save_dir / "psychometric_chronometric.pdf"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 3b. RT distribution comparison — one file per prob level
    # ──────────────────────────────────────────────────────────────────────────

    def plot_rt_distributions(
        self, data: dict, metrics: dict, save_dir: str | Path
    ) -> list[Path]:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        per_prob = metrics["per_prob"]
        rt_range = np.linspace(0.001, 0.6, 300)

        for prob in self.prob_list:
            if prob not in per_prob:
                continue
            h_rts = data["human_rts_by_prob"].get(prob, np.array([]))
            m_rts = data["rt_samples_by_prob"].get(prob, np.array([]))
            if len(h_rts) < 2 or len(m_rts) < 2:
                continue

            info  = per_prob[prob]
            w1    = info["w1_ms"]
            nll_m = info["nll_model"]
            nll_h = info["nll_hh"]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"RT Distribution — P(natural) = {prob:.1f}\n"
                f"Acc: human={info['human_acc']:.2f}  model={info['model_acc']:.2f}  |  "
                f"W1={w1:.1f}ms  NLL_model={nll_m:.3f}  HH_NLL={nll_h:.3f}",
                fontsize=10,
            )

            ax = axes[0]
            ax.hist(h_rts, bins=30, density=True, alpha=0.4, color="#2c7bb6",
                    label=f"Human (n={len(h_rts)})")
            ax.hist(np.clip(m_rts, 0.001, 0.6), bins=50, density=True, alpha=0.3,
                    color="#d7191c", label=f"NRM samples (n={len(m_rts)})")
            try:
                kde_h = gaussian_kde(np.clip(h_rts, 0.001, 0.6))
                kde_m = gaussian_kde(np.clip(m_rts, 0.001, 0.6))
                ax.plot(rt_range, kde_h(rt_range), color="#2c7bb6", lw=2.5)
                ax.plot(rt_range, kde_m(rt_range), color="#d7191c", lw=2.5, ls="--")
            except Exception:
                pass
            ax.axvline(np.median(h_rts), color="#2c7bb6", ls=":", lw=1.5, alpha=0.8)
            ax.axvline(np.median(m_rts), color="#d7191c", ls=":", lw=1.5, alpha=0.8)
            ax.set_xlabel("RT (s)")
            ax.set_ylabel("Density")
            ax.set_title("RT Density Comparison")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)

            ax = axes[1]
            h_sorted = np.sort(h_rts)
            m_sorted = np.sort(np.clip(m_rts, 0.001, 0.6))
            ax.plot(h_sorted, np.linspace(0, 1, len(h_sorted)),
                    color="#2c7bb6", lw=2.5, label="Human")
            ax.plot(m_sorted, np.linspace(0, 1, len(m_sorted)),
                    color="#d7191c", lw=2.5, ls="--", label="NRM")
            ax.set_xlabel("RT (s)")
            ax.set_ylabel("Cumulative Probability")
            ax.set_title("CDF Comparison")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)

            plt.tight_layout()
            prob_tag = f"{prob:.1f}".replace(".", "_")
            out = save_dir / f"rt_distribution_comparison_prob_{prob_tag}.pdf"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(out)

        print(f"  Saved {len(paths)} RT distribution plots.")
        return paths

    # ──────────────────────────────────────────────────────────────────────────
    # 3c. RT likelihood comparison — one file per prob level
    # ──────────────────────────────────────────────────────────────────────────

    def plot_rt_likelihood(
        self, data: dict, metrics: dict, save_dir: str | Path
    ) -> list[Path]:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        per_prob = metrics["per_prob"]
        rt_range = np.linspace(0.001, 0.6, 400)

        for prob in self.prob_list:
            if prob not in per_prob:
                continue
            h_rts = data["human_rts_by_prob"].get(prob, np.array([]))
            m_rts = data["rt_samples_by_prob"].get(prob, np.array([]))
            if len(h_rts) < 2 or len(m_rts) < 2:
                continue

            info  = per_prob[prob]
            nll_m = info["nll_model"]
            nll_h = info["nll_hh"]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"RT Likelihood — P(natural) = {prob:.1f}  "
                f"(NLL_model={nll_m:.3f}  HH_NLL={nll_h:.3f}  "
                f"delta={nll_h - nll_m:+.3f})",
                fontsize=10,
            )

            ax = axes[0]
            try:
                kde_m = gaussian_kde(np.clip(m_rts, 0.001, 0.6))
                ax.fill_between(rt_range, kde_m(rt_range), alpha=0.25, color="#d7191c",
                                label="Model predictive density")
                ax.plot(rt_range, kde_m(rt_range), color="#d7191c", lw=2.5)
            except Exception:
                pass
            ax.hist(h_rts, bins=30, density=True, alpha=0.5, color="#2c7bb6",
                    label=f"Human empirical (n={len(h_rts)})")
            ax.set_xlabel("RT (s)")
            ax.set_ylabel("Density")
            ax.set_title("Model Predictive Density vs Human RTs")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
            ax.text(0.98, 0.95, f"NLL_model = {nll_m:.3f}\nHH_NLL = {nll_h:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

            ax = axes[1]
            try:
                kde_m_r  = gaussian_kde(np.clip(m_rts, 0.001, 0.6))
                h_rts_c  = np.clip(h_rts, 0.001, 0.6)
                log_liks = np.log(np.maximum(kde_m_r.evaluate(h_rts_c), 1e-10))
                hh_lls   = []
                for i, rt_i in enumerate(h_rts_c):
                    rest = np.concatenate([h_rts_c[:i], h_rts_c[i + 1:]])
                    if len(rest) >= 3:
                        try:
                            kde_hh = gaussian_kde(rest)
                            d = float(kde_hh.evaluate(np.array([rt_i]))[0])
                            hh_lls.append(np.log(max(d, 1e-10)))
                        except Exception:
                            hh_lls.append(float("nan"))
                    else:
                        hh_lls.append(float("nan"))
                hh_lls    = np.array(hh_lls)
                advantage = log_liks - hh_lls
                valid     = np.isfinite(advantage)
                colors_adv = np.where(advantage[valid] >= 0, "#d7191c", "#2c7bb6")
                ax.scatter(h_rts_c[valid], advantage[valid], c=colors_adv,
                           alpha=0.4, s=12)
                ax.axhline(0, color="black", lw=1.2, ls="--", label="Model = HH")
                ax.set_xlabel("Human RT (s)")
                ax.set_ylabel("log p(RT|model) − log p(RT|HH)")
                ax.set_title("LL advantage: Model vs HH (per trial)")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.2)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                        ha="center", fontsize=8)

            plt.tight_layout()
            prob_tag = f"{prob:.1f}".replace(".", "_")
            out = save_dir / f"rt_likelihood_comparison_prob_{prob_tag}.pdf"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(out)

        print(f"  Saved {len(paths)} RT likelihood plots.")
        return paths

    # ──────────────────────────────────────────────────────────────────────────
    # 3d. Signed RT histograms — one panel per prob level
    # ──────────────────────────────────────────────────────────────────────────

    def plot_signed_rt_histograms(
        self, data: dict, metrics: dict, save_dir: str | Path
    ) -> Path:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        per_prob = metrics["per_prob"]
        probs    = [p for p in self.prob_list if p in per_prob]
        n_prob   = len(probs)

        fig, axes = plt.subplots(1, n_prob, figsize=(4 * n_prob, 4), sharey=False)
        if n_prob == 1:
            axes = [axes]
        fig.suptitle("Signed RT Distributions (NRM vs Human)", fontsize=13, fontweight="bold")

        for ax, prob in zip(axes, probs):
            info     = per_prob[prob]
            h_rts    = data["human_rts_by_prob"].get(prob, np.array([]))
            coh_info = data["model_info_by_prob"].get(prob, {})
            h_correct = (
                np.array(coh_info.get("human_choices", []))
                == np.array(coh_info.get("true_labels",   []))
            ).astype(float)
            signed_h = h_rts * (2.0 * h_correct - 1.0)

            m_rts_pool = data["rt_samples_by_prob"].get(prob, np.array([]))
            n_trials   = len(h_correct)
            if len(m_rts_pool) >= n_trials:
                m_rts_1p = m_rts_pool[::max(1, len(m_rts_pool) // n_trials)][:n_trials]
            else:
                m_rts_1p = m_rts_pool

            m_pred_ch = np.array(coh_info.get("pred_choices", []))
            m_true_lb = np.array(coh_info.get("true_labels",  []))
            n_use = min(len(m_rts_1p), len(m_pred_ch), len(m_true_lb))
            if n_use > 0:
                m_correct = (m_pred_ch[:n_use] == m_true_lb[:n_use]).astype(float)
                signed_m  = m_rts_1p[:n_use] * (2.0 * m_correct - 1.0)
            else:
                signed_m  = np.array([])

            bins_s = np.linspace(-0.55, 0.55, 41)
            ax.hist(signed_h, bins=bins_s, density=True, alpha=0.5, color="#2c7bb6",
                    label="Human")
            if len(signed_m) > 0:
                ax.hist(signed_m, bins=bins_s, density=True, alpha=0.45, color="#d7191c",
                        label="NRM")
            ax.axvline(0, color="black", lw=1.0, ls="-")
            ax.set_title(
                f"P={prob:.1f}\nacc_h={info['human_acc']:.2f}  acc_m={info['model_acc']:.2f}",
                fontsize=8,
            )
            ax.set_xlabel("Signed RT (s)", fontsize=8)
            ax.set_ylabel("Density",       fontsize=8)
            ax.legend(fontsize=6)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=7)

        plt.tight_layout()
        out = save_dir / "signed_rt_distributions.pdf"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 3e. Summary overview
    # ──────────────────────────────────────────────────────────────────────────

    def plot_summary_overview(
        self, data: dict, metrics: dict, save_dir: str | Path
    ) -> Path:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        per_prob = metrics["per_prob"]
        glb      = metrics["global"]
        probs    = sorted(per_prob.keys())
        x_pos    = np.arange(len(probs))

        h_accs  = [per_prob[p]["human_acc"]       for p in probs]
        m_accs  = [per_prob[p]["model_acc"]        for p in probs]
        h_rts   = [per_prob[p]["human_median_rt"]  for p in probs]
        m_rts   = [per_prob[p]["model_median_rt"]  for p in probs]
        w1s     = [per_prob[p]["w1_ms"]            for p in probs]
        nlls_m  = [per_prob[p]["nll_model"]        for p in probs]
        nlls_h  = [per_prob[p]["nll_hh"]           for p in probs]

        fig = plt.figure(figsize=(18, 14))
        gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.4)

        # Psychometric
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(probs, h_accs, "o-", color="#2c7bb6", label="Human", lw=2, ms=7)
        ax1.plot(probs, m_accs, "s--", color="#d7191c", label="NRM",  lw=2, ms=7)
        ax1.axhline(0.5, color="gray", ls=":", lw=1)
        ax1.set_xlabel("P(natural)")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Psychometric Curve")
        ax1.set_ylim(0.3, 1.05)
        ax1.set_xticks(probs)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # Chronometric
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(probs, [r * 1000 for r in h_rts], "o-", color="#2c7bb6",
                 label="Human", lw=2, ms=7)
        ax2.plot(probs, [r * 1000 for r in m_rts], "s--", color="#d7191c",
                 label="NRM",   lw=2, ms=7)
        ax2.set_xlabel("P(natural)")
        ax2.set_ylabel("Median RT (ms)")
        ax2.set_title("Chronometric Curve")
        ax2.set_xticks(probs)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # NLL bar chart
        ax3 = fig.add_subplot(gs[0, 2])
        w   = 0.35
        ax3.bar(x_pos - w / 2, nlls_m, w, color="#d7191c", alpha=0.8, label="NRM")
        ax3.bar(x_pos + w / 2, nlls_h, w, color="#2c7bb6", alpha=0.8, label="Human (HH NLL)")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f"{p:.1f}" for p in probs], fontsize=7)
        ax3.set_xlabel("P(natural)")
        ax3.set_ylabel("NLL (lower = better)")
        ax3.set_title("NLL per Naturalness Level")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.2, axis="y")

        # W1 bar chart
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.bar(x_pos, w1s, color="#7b2d8b", alpha=0.8)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"{p:.1f}" for p in probs], fontsize=7)
        ax4.set_xlabel("P(natural)")
        ax4.set_ylabel("Wasserstein-1 (ms)")
        ax4.set_title("W1 Distance per Naturalness Level")
        ax4.grid(alpha=0.2, axis="y")

        # Accuracy gap
        ax5 = fig.add_subplot(gs[1, 1])
        gaps    = [m - h for m, h in zip(m_accs, h_accs)]
        colors_g = ["#d7191c" if g >= 0 else "#2c7bb6" for g in gaps]
        ax5.bar(x_pos, gaps, color=colors_g, alpha=0.8)
        ax5.axhline(0, color="black", lw=1)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([f"{p:.1f}" for p in probs], fontsize=7)
        ax5.set_xlabel("P(natural)")
        ax5.set_ylabel("Accuracy gap (Model − Human)")
        ax5.set_title("Accuracy Gap per Naturalness Level")
        ax5.grid(alpha=0.2, axis="y")

        # Global metrics text
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        nc  = glb.get("nc", {})
        txt = (
            f"Global Metrics (NRM)\n\n"
            f"  Acc:           {glb['acc']*100:.1f}%\n"
            f"  Human Acc:     {glb['human_acc']*100:.1f}%\n"
            f"  r_acc:         {glb['r_acc']:.3f}\n"
            f"  r_RT:          {glb['r_rt']:.3f}\n"
            f"  CCC_RT:        {glb['ccc_rt']:.3f}\n"
            f"  Mean NLL:      {glb['mean_nll']:.3f}\n"
            f"  HH NLL (NC):   {glb['hh_mean_nll']:.3f}\n"
            f"  Mean W1:       {glb['mean_w1_ms']:.1f} ms\n"
            f"\nNoise Ceiling\n"
            f"  HH r_RT:       {nc.get('r_rt', float('nan')):.3f}\n"
            f"  HH r_acc:      {nc.get('r_acc', float('nan')):.3f}\n"
            f"  HH Acc:        {nc.get('acc', float('nan'))*100:.1f}%\n"
        )
        ax6.text(0.05, 0.95, txt, transform=ax6.transAxes, va="top",
                 fontsize=9, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        # RT scatter (per-prob median)
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter([r * 1000 for r in h_rts], [r * 1000 for r in m_rts],
                    c=_PROB_COLORS[:len(probs)], s=80, zorder=3)
        for i, p in enumerate(probs):
            ax7.annotate(f"{p:.1f}", (h_rts[i]*1000, m_rts[i]*1000),
                         fontsize=7, ha="left", va="bottom")
        lim = [
            min(min(h_rts), min(m_rts)) * 900,
            max(max(h_rts), max(m_rts)) * 1100,
        ]
        ax7.plot(lim, lim, "k--", lw=1, alpha=0.5)
        ax7.set_xlabel("Human Median RT (ms)")
        ax7.set_ylabel("Model Median RT (ms)")
        ax7.set_title(f"RT Correlation (r={glb['r_rt']:.3f})")
        ax7.grid(alpha=0.2)

        # Accuracy scatter
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.scatter(h_accs, m_accs, c=_PROB_COLORS[:len(probs)], s=80, zorder=3)
        for i, p in enumerate(probs):
            ax8.annotate(f"{p:.1f}", (h_accs[i], m_accs[i]), fontsize=7,
                         ha="left", va="bottom")
        al = [min(min(h_accs), min(m_accs)) - 0.02,
              max(max(h_accs), max(m_accs)) + 0.02]
        ax8.plot(al, al, "k--", lw=1, alpha=0.5)
        ax8.set_xlabel("Human Accuracy")
        ax8.set_ylabel("Model Accuracy")
        ax8.set_title(f"Accuracy Correlation (r={glb['r_acc']:.3f})")
        ax8.grid(alpha=0.2)

        # Per-trial RT scatter
        ax9 = fig.add_subplot(gs[2, 2])
        N_sub = min(3000, len(data["true_rts"]))
        idx   = np.random.choice(len(data["true_rts"]), N_sub, replace=False)
        ax9.scatter(data["true_rts"][idx] * 1000,
                    data["pred_rts"][idx]  * 1000,
                    alpha=0.15, s=4, color="steelblue")
        lim9 = [0, 550]
        ax9.plot(lim9, lim9, "r--", lw=1)
        ax9.set_xlim(*lim9); ax9.set_ylim(*lim9)
        ax9.set_xlabel("Human RT (ms)")
        ax9.set_ylabel("Model RT (ms)")
        ax9.set_title("Per-trial RT scatter")
        ax9.grid(alpha=0.2)

        out = save_dir / "summary_overview.pdf"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")
        return out
