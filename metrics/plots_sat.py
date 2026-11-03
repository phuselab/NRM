"""SAT diagnostic figures: accuracy-vs-deadline curves, steepness and RT plots (§3)."""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix

from datasets.sat.loader import CATEGORIES
from metrics.relational_metrics import _CHANCE
from scripts.results_io import display_path


def _ll_outlier_mask(hh_arr, mod_arr, k=3.0):
    """
    Tukey IQR-based outlier detection on the combined (HH LL, Model LL) distribution.

    A point is an outlier if EITHER its HH LL or its Model LL lies outside
    [Q1 − k·IQR, Q3 + k·IQR] computed on the union of both arrays.
    Non-finite values (±inf, NaN) are always treated as outliers.

    Returns
    -------
    valid : np.ndarray[bool]  — True for inliers
    lower : float             — lower fence
    upper : float             — upper fence
    n_out : int               — number of outliers
    """
    finite_vals = np.concatenate([
        hh_arr[np.isfinite(hh_arr)],
        mod_arr[np.isfinite(mod_arr)],
    ])
    if len(finite_vals) < 4:                       # too few points to estimate IQR
        valid = np.isfinite(hh_arr) & np.isfinite(mod_arr)
        return valid, -np.inf, np.inf, int((~valid).sum())

    q1, q3 = np.nanpercentile(finite_vals, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        # Degenerate distribution — fall back to median ± k*std
        med = np.nanmedian(finite_vals)
        std = np.nanstd(finite_vals) or 1.0
        lower, upper = med - k * std, med + k * std
    else:
        lower = q1 - k * iqr
        upper = q3 + k * iqr

    valid = (
        np.isfinite(hh_arr)  & np.isfinite(mod_arr) &
        (hh_arr  >= lower)   & (hh_arr  <= upper) &
        (mod_arr >= lower)   & (mod_arr <= upper)
    )
    return valid, float(lower), float(upper), int((~valid).sum())


class Plotter:
    """Handle all plotting functionality."""
    
    def __init__(self, save_dir='outputs'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
    
    def plot_confusion_matrices(self, true_labels, human_choices, model_choices, 
                               human_acc, model_acc, model_human_agreement, n_choices):
        """Create confusion matrix comparison plot."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Human vs True Labels
        cm_human = confusion_matrix(true_labels, human_choices, 
                                    labels=range(n_choices))
        sns.heatmap(cm_human, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=range(n_choices), yticklabels=range(n_choices))
        axes[0].set_title(f'Humans vs True Labels\n(Accuracy: {human_acc:.4f})')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Human Choice')
        
        # Model vs True Labels
        cm_model = confusion_matrix(true_labels, model_choices, 
                                   labels=range(n_choices))
        sns.heatmap(cm_model, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                   xticklabels=range(n_choices), yticklabels=range(n_choices))
        axes[1].set_title(f'Model vs True Labels\n(Accuracy: {model_acc:.4f})')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Model Choice')
        
        # Model vs Human
        cm_model_human = confusion_matrix(human_choices, model_choices, 
                                         labels=range(n_choices))
        sns.heatmap(cm_model_human, annot=True, fmt='d', cmap='Purples', ax=axes[2],
                   xticklabels=range(n_choices), yticklabels=range(n_choices))
        axes[2].set_title(f'Model vs Human Choices\n(Agreement: {model_human_agreement:.4f})')
        axes[2].set_ylabel('Human Choice')
        axes[2].set_xlabel('Model Choice')
        
        plt.tight_layout()
        save_path = self.save_dir / 'confusion_matrices.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_rt_correlations(self, true_rts, pred_rts, rt_metrics, 
                            human_human_data=None):
        """Create RT correlation plots."""
        if human_human_data is not None and human_human_data['rt_pairs'] is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Human-Human RT
            rt_pairs = human_human_data['rt_pairs']
            axes[0].scatter(rt_pairs[:, 0], rt_pairs[:, 1], alpha=0.5, s=20, color='blue')
            min_val = min(rt_pairs[:, 0].min(), rt_pairs[:, 1].min())
            max_val = max(rt_pairs[:, 0].max(), rt_pairs[:, 1].max())
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', 
                        label='Perfect agreement')
            axes[0].set_xlabel('Human 1 RT (s)')
            axes[0].set_ylabel('Human 2 RT (s)')
            axes[0].set_title(f"Human-Human RT Correlation\n"
                            f"(r={human_human_data['rt_correlation']['r']:.3f}, "
                            f"n={len(rt_pairs)} pairs)")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Model-Human RT
            axes[1].scatter(true_rts, pred_rts, alpha=0.5, s=10, color='green')
            axes[1].plot([true_rts.min(), true_rts.max()], 
                        [true_rts.min(), true_rts.max()], 
                        'r--', label='Perfect match')
            axes[1].set_xlabel('Human RT (s)')
            axes[1].set_ylabel('Model RT (s)')
            axes[1].set_title(f"Model-Human RT Correlation\n"
                            f"(r={rt_metrics['pearson_r']:.3f}, n={len(true_rts)})")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Comparison
            hh_r = human_human_data['rt_correlation']['r']
            mh_r = rt_metrics['pearson_r']
            
            correlations = ['Human-Human\nRT', 'Model-Human\nRT']
            corr_values = [hh_r, mh_r]
            colors = ['blue', 'green']
            
            bars = axes[2].bar(correlations, corr_values, color=colors, alpha=0.7)
            axes[2].set_ylabel('Pearson Correlation')
            axes[2].set_title('RT Correlation Comparison')
            axes[2].set_ylim([0, 1])
            axes[2].axhline(0, color='black', linewidth=0.5)
            axes[2].grid(True, alpha=0.3, axis='y')
            
            for bar, corr in zip(bars, corr_values):
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Add interpretation
            diff = mh_r - hh_r
            if diff > 0:
                text = f'Model exceeds\nhuman-human\ncorrelation\nby {diff:.3f}'
                color = 'lightgreen'
            else:
                text = f'Model is {abs(diff):.3f}\nbelow human-human\ncorrelation'
                color = 'lightyellow'
            
            axes[2].text(0.5, 0.5, text, ha='center', va='center',
                        transform=axes[2].transAxes,
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
            
        else:
            # Just Model-Human
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.scatter(true_rts, pred_rts, alpha=0.5, s=10, color='green')
            ax.plot([true_rts.min(), true_rts.max()], 
                   [true_rts.min(), true_rts.max()], 
                   'r--', label='Perfect match')
            ax.set_xlabel('Human RT (s)')
            ax.set_ylabel('Model RT (s)')
            ax.set_title(f"Model-Human RT Correlation\n(r={rt_metrics['pearson_r']:.3f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / 'rt_correlations.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_error_analysis(self, true_labels, human_choices, model_choices, 
                           true_rts, pred_rts, agreement_breakdown, rt_metrics, 
                           model_human_agreement, n_choices):
        """2×2 error-analysis figure."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Agreement breakdown
        agreement_data = {
            'All Correct\n(Model=Human=True)': agreement_breakdown['all_correct']['count'],
            'Model Correct\nHuman Wrong': agreement_breakdown['model_correct_human_wrong']['count'],
            'Human Correct\nModel Wrong': agreement_breakdown['human_correct_model_wrong']['count'],
            'Both Wrong\n(different)': agreement_breakdown['both_wrong_different']['count'],
            'Both Wrong\n(same error)': agreement_breakdown['both_wrong_same_error']['count'],
        }
        
        colors = ['green', 'orange', 'red', 'darkred', 'purple']
        bars = axes[0, 0].bar(range(len(agreement_data)), list(agreement_data.values()), 
                             color=colors, alpha=0.7)
        axes[0, 0].set_xticks(range(len(agreement_data)))
        axes[0, 0].set_xticklabels(list(agreement_data.keys()), rotation=45, ha='right')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Agreement Pattern Breakdown')
        
        # Add percentages
        n_total = len(model_choices)
        for bar, count in zip(bars, agreement_data.values()):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{count}\n({count/n_total*100:.1f}%)',
                          ha='center', va='bottom', fontsize=9)
        
        # 2. RT scatter
        axes[0, 1].scatter(true_rts, pred_rts, alpha=0.5, s=10, label='Model RTs')
        axes[0, 1].plot([true_rts.min(), true_rts.max()], 
                       [true_rts.min(), true_rts.max()], 
                       'r--', label='Perfect match')
        axes[0, 1].set_xlabel('Human RT (s)')
        axes[0, 1].set_ylabel('Model RT (s)')
        axes[0, 1].set_title(f"RT Correlation: Model vs Human\n"
                           f"(r={rt_metrics['pearson_r']:.3f})")
        axes[0, 1].legend()
        
        # 3. RT distributions
        axes[1, 0].hist(true_rts, bins=50, alpha=0.6, label='Human RTs', density=True)
        axes[1, 0].hist(pred_rts, bins=50, alpha=0.6, label='Model RTs', density=True)
        axes[1, 0].set_xlabel('Reaction Time (s)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('RT Distributions')
        axes[1, 0].legend()
        
        # 4. Agreement by class
        agreement_by_class = []
        for class_idx in range(n_choices):
            mask = true_labels == class_idx
            if mask.sum() > 0:
                agreement = (model_choices[mask] == human_choices[mask]).sum() / mask.sum()
                agreement_by_class.append(agreement)
            else:
                agreement_by_class.append(0)
        
        axes[1, 1].bar(range(n_choices), agreement_by_class, alpha=0.7, color='teal')
        axes[1, 1].set_xlabel('True Label')
        axes[1, 1].set_ylabel('Model-Human Agreement')
        axes[1, 1].set_title('Model-Human Agreement by Class')
        axes[1, 1].set_xticks(range(n_choices))
        axes[1, 1].axhline(model_human_agreement, color='red', linestyle='--', 
                          label=f'Overall: {model_human_agreement:.3f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        save_path = self.save_dir / 'error_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path


class SATPlotter:
    """
    Plot SAT-style (16 subplots + overall) for NRM vs humans.
 
    Produces two figures with the same layout but a different X axis:
        1. sat_curves_rt_vs_acc.png   — X = actual mean RT (s)
        2. sat_curves_beep_vs_acc.png — X = nominal beep (s)
 
    In both:
        - 16 subplots (4×4), one per category
        - Human curve in blue with error bars on X and Y
        - NRM curve in red with errorbars on X and Y
        - Overall subplot top left (or an extra row)
    """
 
    def __init__(self, save_dir='outputs'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
 
    # -------------------------------------------------------------------------
    # Main plot (16 subplots, one per category)
    # -------------------------------------------------------------------------
 
    def _make_16_subplot_figure(self, human_by_cat, model_by_cat,
                                human_overall, model_overall,
                                x_mode, use_sem, n_choices=16):
        """
        Creates a figure with n_choices subplots + 1 overall subplot (17 panels,
        a 3×6 or 4×5 grid depending on n_choices).
 
        x_mode: 'rt'   → X = actual mean RT
                'beep' → X = nominal beep
 
        Returns:
            matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
 
        err_key = 'sem' if use_sem else 'std'
        n_cols  = 4
        # +1 for the overall panel, rounded up to a multiple of n_cols
        n_panels = n_choices + 1
        n_rows   = (n_panels + n_cols - 1) // n_cols
 
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 4.0, n_rows * 3.2),
                                 squeeze=False)
        axes_flat = axes.flatten()
 
        def _plot_one(ax, human_data, model_data, title):
            """Draw one (human, model) pair on ax."""
 
            plotted_something = False
 
            if human_data is not None:
                beeps = human_data['beeps_s']
                hd    = human_data['per_beep']
 
                if x_mode == 'rt':
                    xs = np.array([hd[b]['rt_mean']  for b in beeps])
                else:
                    xs = np.array(beeps)
 
                acc  = np.array([hd[b]['acc_mean'] for b in beeps]) * 100
                xerr = np.array([hd[b][f'rt_{err_key}'] for b in beeps]) \
                       if x_mode == 'rt' else None
                yerr = np.array([hd[b][f'acc_{err_key}'] for b in beeps]) * 100
 
                ax.errorbar(
                    xs, acc,
                    xerr=xerr, yerr=yerr,
                    fmt='o-', color='royalblue', alpha=0.85,
                    label='Human', capsize=3, capthick=1.0,
                    linewidth=1.6, markersize=5, zorder=3,
                )
                plotted_something = True
 
            if model_data is not None:
                beeps = model_data['beeps_s']
                md    = model_data['per_beep']
 
                if x_mode == 'rt':
                    xs = np.array([md[b]['rt_mean']  for b in beeps])
                else:
                    xs = np.array(beeps)
 
                acc  = np.array([md[b]['acc_mean'] for b in beeps]) * 100
                xerr = np.array([md[b][f'rt_{err_key}'] for b in beeps]) \
                       if x_mode == 'rt' else None
                yerr = np.array([md[b][f'acc_{err_key}'] for b in beeps]) * 100
 
                ax.errorbar(
                    xs, acc,
                    xerr=xerr, yerr=yerr,
                    fmt='s--', color='tomato', alpha=0.85,
                    label='NRM', capsize=3, capthick=1.0,
                    linewidth=1.6, markersize=5, zorder=3,
                )
                plotted_something = True
 
            if not plotted_something:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', color='gray', fontsize=9)
 
            # Vertical lines at the nominal beeps (x_mode='rt' only)
            if x_mode == 'rt' and human_data is not None:
                for b in human_data['beeps_s']:
                    ax.axvline(b, color='gray', linestyle=':', alpha=0.3,
                               linewidth=0.7)
 
            ax.set_ylim([0, 105])
            ax.set_title(title, fontsize=9, pad=3)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if x_mode == 'beep' and (human_data or model_data):
                ref = human_data or model_data
                ax.set_xticks(ref['beeps_s'])
                ax.tick_params(axis='x', labelsize=6, rotation=30)
            return plotted_something
 
        # --- Subplot 0: overall ---
        _plot_one(axes_flat[0], human_overall, model_overall, 'Overall')
        axes_flat[0].legend(fontsize=7, loc='lower right')
 
        # --- Subplots 1..16: categories ---
        for cat_idx in range(n_choices):
            ax   = axes_flat[cat_idx + 1]
            h_d  = human_by_cat.get(cat_idx) if human_by_cat else None
            m_d  = model_by_cat.get(cat_idx) if model_by_cat else None
            name = (h_d['category_name'] if h_d
                    else (m_d['category_name'] if m_d else str(cat_idx)))
            _plot_one(ax, h_d, m_d, name)
 
        # Hide empty panels
        for i in range(n_choices + 1, len(axes_flat)):
            axes_flat[i].set_visible(False)
 
        # Shared axis labels
        x_label = ('Reaction time (s)'
                   if x_mode == 'rt' else 'Time limit / beep (s)')
        fig.supxlabel(x_label,    fontsize=10, y=0.01)
        fig.supylabel('Accuracy (%)', fontsize=10, x=0.01)
 
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
        return fig
 
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
 
    def plot_sat_by_category(self, human_by_cat, model_by_cat,
                             human_overall, model_overall,
                             use_sem=True, n_choices=16):
        """
        Generate both SAT plots (actual RT and nominal beep).
 
        Args:
            human_by_cat:   output of calculate_human_sat_curves_by_category
            model_by_cat:   output of calculate_model_sat_curves_by_category
            human_overall:  output of calculate_human_sat_curves
            model_overall:  overall aggregate of the model
            use_sem:        True → SEM error bars, False → std
            n_choices:      number of categories (default 16)
 
        Returns:
            dict {'rt_vs_acc': Path, 'beep_vs_acc': Path}
        """
        paths = {}
 
        # Plot 1: X = actual RT
        fig1 = self._make_16_subplot_figure(
            human_by_cat=human_by_cat,
            model_by_cat=model_by_cat,
            human_overall=human_overall,
            model_overall=model_overall,
            x_mode='rt',
            use_sem=use_sem,
            n_choices=n_choices,
        )
        p1 = self.save_dir / 'sat_curves_rt_vs_acc.png'
        fig1.savefig(p1, dpi=200, bbox_inches='tight')
        plt.close(fig1)
        paths['rt_vs_acc'] = p1
        print(f"  [SAT] Saved {display_path(p1)}")
 
        # Plot 2: X = nominal beep
        fig2 = self._make_16_subplot_figure(
            human_by_cat=human_by_cat,
            model_by_cat=model_by_cat,
            human_overall=human_overall,
            model_overall=model_overall,
            x_mode='beep',
            use_sem=use_sem,
            n_choices=n_choices,
        )
        p2 = self.save_dir / 'sat_curves_beep_vs_acc.png'
        fig2.savefig(p2, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        paths['beep_vs_acc'] = p2
        print(f"  [SAT] Saved {display_path(p2)}")
 
        return paths
 
    # -------------------------------------------------------------------------
    # Summary CSV table
    # -------------------------------------------------------------------------
 
    def save_sat_table(self, human_by_cat, model_by_cat,
                       human_overall, model_overall):
        """
        Save a CSV table with columns:
        category | category_name | beep_s |
        human_rt_mean | human_rt_std | human_acc_mean | human_acc_std |
        model_rt_mean | model_rt_std | model_acc_mean | model_acc_std
 
        Also prints a summary to the console.
 
        Returns:
            Path of the CSV
        """
        import numpy as np
        rows = []
 
        def _add_rows(cat_id, cat_name, h_data, m_data):
            all_beeps = set()
            if h_data:
                all_beeps.update(h_data['beeps_s'])
            if m_data:
                all_beeps.update(m_data['beeps_s'])
            for beep in sorted(all_beeps):
                row = {'category': cat_id, 'category_name': cat_name,
                       'beep_s': beep}
                if h_data and beep in h_data['per_beep']:
                    hb = h_data['per_beep'][beep]
                    row.update({
                        'human_rt_mean':  hb['rt_mean'],
                        'human_rt_std':   hb['rt_std'],
                        'human_rt_sem':   hb['rt_sem'],
                        'human_acc_mean': hb['acc_mean'],
                        'human_acc_std':  hb['acc_std'],
                        'human_acc_sem':  hb['acc_sem'],
                        'human_n_images': hb['n_images'],
                        'human_n_trials': hb.get('n_trials', np.nan),
                    })
                if m_data and beep in m_data['per_beep']:
                    mb = m_data['per_beep'][beep]
                    row.update({
                        'model_rt_mean':  mb['rt_mean'],
                        'model_rt_std':   mb['rt_std'],
                        'model_rt_sem':   mb['rt_sem'],
                        'model_acc_mean': mb['acc_mean'],
                        'model_acc_std':  mb['acc_std'],
                        'model_acc_sem':  mb['acc_sem'],
                        'model_n_images': mb['n_images'],
                    })
                rows.append(row)
 
        # Overall as category -1
        _add_rows(-1, 'overall', human_overall, model_overall)
 
        # Per category
        all_cats = set()
        if human_by_cat:
            all_cats.update(human_by_cat.keys())
        if model_by_cat:
            all_cats.update(model_by_cat.keys())
 
        for cat in sorted(all_cats):
            h_d = human_by_cat.get(cat) if human_by_cat else None
            m_d = model_by_cat.get(cat) if model_by_cat else None
            name = (h_d['category_name'] if h_d
                    else (m_d['category_name'] if m_d else str(cat)))
            _add_rows(cat, name, h_d, m_d)
 
        df = pd.DataFrame(rows)
        csv_path = self.save_dir / 'sat_curves_summary.csv'
        df.to_csv(csv_path, index=False)
 
        # Console summary (overall only)
        overall_rows = df[df['category'] == -1]
        if len(overall_rows) > 0:
            print("\n" + "="*100)
            print("SAT SUMMARY — OVERALL")
            print(f"{'Beep(s)':>8} | {'Human RT±std':>18} | {'Human Acc±std':>16} | "
                  f"{'Model RT±std':>18} | {'Model Acc±std':>16}")
            print("-"*100)
            for _, r in overall_rows.iterrows():
                hrt  = (f"{r.get('human_rt_mean', float('nan')):.3f}"
                        f"±{r.get('human_rt_std',  float('nan')):.3f}")
                hacc = (f"{r.get('human_acc_mean', float('nan'))*100:.1f}"
                        f"±{r.get('human_acc_std',  float('nan'))*100:.1f}%")
                mrt  = (f"{r.get('model_rt_mean', float('nan')):.3f}"
                        f"±{r.get('model_rt_std',  float('nan')):.3f}")
                macc = (f"{r.get('model_acc_mean', float('nan'))*100:.1f}"
                        f"±{r.get('model_acc_std',  float('nan'))*100:.1f}%")
                print(f"{r['beep_s']:>8.3f} | {hrt:>18} | {hacc:>16} | "
                      f"{mrt:>18} | {macc:>16}")
            print("="*100)
 
        print(f"  [SAT] Table saved to {display_path(csv_path)}")
        return csv_path


def plot_sat_by_level(
        human_curves: dict,
        model_curves: dict,
        x_mode: str = 'beep',
        use_sem: bool = True,
        save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Equivalent to Figure 3 of the paper:
      Rows: Human | NRM|Rows: Human | NRM
      Each curve traces accuracy vs time for one perturbation level.
      Includes the chance point (t=0, acc=6.25%) at the start of every curve.
    """
    ordered_lvls = human_curves['ordered_levels']
    display      = human_curves['display_labels']
    colors       = human_curves['colors']
    scenario     = human_curves.get('scenario', '')
    err_key      = 'sem' if use_sem else 'std'
    chance_pct   = _CHANCE * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    titles   = ['Human', 'NRM']
    sources  = [human_curves['overall'], model_curves['overall']]

    for ax, title, source in zip(axes, titles, sources):
        for level in ordered_lvls:
            curves = source.get(level)
            if curves is None:
                continue
            beeps = curves['beeps_s']
            pd_   = curves['per_beep']

            xs   = np.array([pd_[b]['rt_mean'] if x_mode == 'rt' else b
                              for b in beeps])
            acc  = np.array([pd_[b]['acc_mean'] for b in beeps]) * 100.0
            yerr = np.array([pd_[b][f'acc_{err_key}'] for b in beeps]) * 100.0
            xerr = (np.array([pd_[b][f'rt_{err_key}'] for b in beeps])
                    if x_mode == 'rt' else None)

            col = colors.get(level, 'gray')
            lbl = display.get(level, level)

            # Prepend chance point at t=0
            xs_full   = np.concatenate([[0.0], xs])
            acc_full  = np.concatenate([[chance_pct], acc])
            yerr_full = np.concatenate([[0.0], yerr])
            xerr_full = (np.concatenate([[0.0], xerr]) if xerr is not None else None)

            ax.errorbar(
                xs_full, acc_full,
                xerr=xerr_full, yerr=yerr_full,
                fmt='o-', color=col, label=lbl,
                capsize=3, capthick=1.0, linewidth=1.8, markersize=5,
            )

        ax.axhline(chance_pct, color='gray', ls='--', lw=0.8, alpha=0.5,
                   label='Chance (6.25%)')
        ax.set_ylim([0, 105])
        ax.set_ylabel('Accuracy (%)' if ax is axes[0] else '')
        x_label = ('Reaction time (s)' if x_mode == 'rt' else 'Time / Beep (s)')
        ax.set_xlabel(x_label)
        ax.set_title(f'{title}  [{scenario}]', fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'SAT Curves — {scenario}', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  [SAT Fig3] Saved {display_path(save_path)}")
    return fig


def plot_sat_by_level_and_category(
        human_curves: dict,
        model_curves: dict,
        x_mode: str = 'beep',
        use_sem: bool = True,
        n_choices: int = 16,
        save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Figure with 17 subplots (overall + 16 categories).
    Each subplot: Human (solid) and Model (dashed) per level.
    """
    ordered_lvls = human_curves['ordered_levels']
    display      = human_curves['display_labels']
    colors_map   = human_curves['colors']
    scenario     = human_curves.get('scenario', '')
    err_key      = 'sem' if use_sem else 'std'
    chance_pct   = _CHANCE * 100.0

    n_cols   = 4
    n_panels = n_choices + 1
    n_rows   = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.0, n_rows * 3.2),
                             squeeze=False)
    axes_flat = axes.flatten()

    def _plot_one(ax, h_by_lev, m_by_lev, title):
        legend_handles = []
        for level in ordered_lvls:
            col  = colors_map.get(level, 'gray')
            lbl  = display.get(level, level)
            for (source, ls, marker) in [(h_by_lev, '-', 'o'), (m_by_lev, '--', 's')]:
                curves = source.get(level) if source else None
                if curves is None:
                    continue
                beeps = curves['beeps_s']
                pd_   = curves['per_beep']
                xs   = np.array([pd_[b]['rt_mean'] if x_mode == 'rt' else b for b in beeps])
                acc  = np.array([pd_[b]['acc_mean'] for b in beeps]) * 100.0
                yerr = np.array([pd_[b][f'acc_{err_key}'] for b in beeps]) * 100.0
                xerr = (np.array([pd_[b][f'rt_{err_key}'] for b in beeps])
                        if x_mode == 'rt' else None)
                xs_f   = np.concatenate([[0.0], xs])
                acc_f  = np.concatenate([[chance_pct], acc])
                yerr_f = np.concatenate([[0.0], yerr])
                xerr_f = (np.concatenate([[0.0], xerr]) if xerr is not None else None)
                h, = ax.plot(xs_f, acc_f, ls=ls, marker=marker, color=col,
                              linewidth=1.4, markersize=4, alpha=0.85)
                ax.errorbar(xs_f, acc_f, xerr=xerr_f, yerr=yerr_f, fmt='none',
                            color=col, capsize=2, capthick=0.7, alpha=0.6)
            legend_handles.append(
                mlines.Line2D([], [], color=col, label=lbl, linewidth=1.4)
            )
        ax.axhline(chance_pct, color='gray', ls=':', lw=0.7, alpha=0.4)
        ax.set_ylim([0, 105])
        ax.set_title(title, fontsize=8, pad=3)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        return legend_handles

    h_ov = human_curves['overall']
    m_ov = model_curves['overall']
    leg_h = _plot_one(axes_flat[0], h_ov, m_ov, f'Overall — {scenario}')
    solid  = mlines.Line2D([], [], color='black', ls='-',  label='Human',     linewidth=1.4)
    dashed = mlines.Line2D([], [], color='black', ls='--', label='NRM', linewidth=1.4)
    axes_flat[0].legend(handles=leg_h + [solid, dashed], fontsize=6, loc='lower right')

    h_by_cat = human_curves['by_category']
    m_by_cat = model_curves['by_category']
    for cat_idx in range(n_choices):
        ax  = axes_flat[cat_idx + 1]
        h_d = {lv: h_by_cat.get(lv, {}).get(cat_idx) for lv in ordered_lvls}
        m_d = {lv: m_by_cat.get(lv, {}).get(cat_idx) for lv in ordered_lvls}
        nm  = CATEGORIES[cat_idx] if cat_idx < len(CATEGORIES) else str(cat_idx)
        _plot_one(ax, h_d, m_d, nm)

    for i in range(n_choices + 1, len(axes_flat)):
        axes_flat[i].set_visible(False)

    x_label = 'Reaction time (s)' if x_mode == 'rt' else 'Time / Beep (s)'
    fig.supxlabel(x_label,        fontsize=9, y=0.01)
    fig.supylabel('Accuracy (%)', fontsize=9, x=0.01)
    fig.suptitle(
        f'Human (─) andRows: Human | NRM (- -) SAT by level — {scenario}',
        fontsize=10, fontweight='bold',
    )
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  [SAT level] Saved {display_path(save_path)}")
    return fig


def _boxplot_panel(ax, data_dict: dict, ylabel: str, title: str,
                   colors: dict | None = None, ylim: tuple | None = None,
                   hline: float | None = None):
    """Helper: boxplot with an overlaid scatter.

    NaN values are dropped before the boxplot (a NaN makes the median NaN ->
    box invisible in matplotlib) but are reported as a warning in the title.
    The scatter is drawn over the finite values only.
    """
    labels = list(data_dict.keys())
    raw    = [data_dict[lbl] for lbl in labels]
    if colors is None:
        colors = {}

    # Drop NaN for the boxplot; keep count of how many were removed
    clean: list[list[float]] = []
    n_nan_total = 0
    for d in raw:
        d_arr   = np.array(d, dtype=float)
        finite  = d_arr[np.isfinite(d_arr)].tolist()
        n_nan_total += int(np.sum(~np.isfinite(d_arr)))
        clean.append(finite)

    # Replace each label with one carrying the valid count
    labels_with_n = [
        f'{lbl}\n(N={len(c)})' if len(c) != len(raw[i]) else f'{lbl}\n(N={len(c)})'
        for i, (lbl, c) in enumerate(zip(labels, clean))
    ]

    default_colors = ['#4472C4', '#ED7D31', '#A9D18E', '#FF0000', '#7030A0']

    # Draw the boxplot only for groups with ≥ 2 valid points
    # Groups with 0-1 points are skipped by the boxplot but shown as scatter
    box_indices = [i for i, c in enumerate(clean) if len(c) >= 2]
    if box_indices:
        # No tick labels here: `labels=` was removed in matplotlib 3.9 and the
        # ticks are set below for *all* groups, including those with <2 points
        # that the boxplot skips.
        bp = ax.boxplot(
            [clean[i] for i in box_indices],
            positions=[i + 1 for i in box_indices],
            patch_artist=True, notch=False, widths=0.5,
            medianprops=dict(color='darkorange', linewidth=2.5),
            whiskerprops=dict(linewidth=1.3),
            capprops=dict(linewidth=1.3),
            flierprops=dict(marker='x', markersize=4, alpha=0.4),
            zorder=2,
        )
        for j, (patch, bi) in enumerate(zip(bp['boxes'], box_indices)):
            lbl = labels[bi]
            col = colors.get(lbl, default_colors[bi % len(default_colors)])
            patch.set_facecolor(col)
            patch.set_alpha(0.70)

    # Always place an xtick at every position, including 0-1 point groups
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels_with_n, fontsize=9)

    # Scatter (on the finite values, not raw, so NaN is never plotted)
    rng = np.random.default_rng(0)
    for i, d_clean in enumerate(clean, start=1):
        if len(d_clean) == 0:
            continue
        jitter = rng.normal(0, 0.06, len(d_clean))
        ax.scatter(i + jitter, d_clean, s=14, color='black', alpha=0.50, zorder=3)

    # Note NaN in the title, if any
    title_full = title
    if n_nan_total > 0:
        title_full += f'\n[{n_nan_total} NaN values excluded]'

    if ylim is not None:
        ax.set_ylim(ylim)
    if hline is not None:
        ax.axhline(hline, color='gray', ls='--', lw=1.0, alpha=0.6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title_full, fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')


def plot_satbench_figures(
        metrics: dict,
        human_curves: dict,
        model_curves: dict,
        save_dir: Path,
        prefix: str = 'satbench',
        n_choices: int = 16,
) -> dict[str, Path]:
    """
    Generate the equivalents of Figures 4-7 of the SATBench paper (Section 5).

    Figure 4 — RMSE Boxplot   : eRMSE distribution over observers
    Figure 5 — Corr Boxplot   : Spearman distribution over observers
    Figure 6 — Cat Barplot    : median correlation per category
    Figure 7 — Steepness      : kappa vs perturbation level (log-scale)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    scenario     = metrics.get('scenario', 'SAT')
    ordered_lvls = metrics['ordered_levels']
    display      = metrics['display_labels']

    # ── Figure 4: RMSE Boxplot ────────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(6, 5))

    model_ermse = metrics['ermse']['model_per_obs']
    human_ermse = metrics['ermse']['human_per_obs']
    boxdata4: dict[str, list[float]] = {}
    if human_ermse:
        boxdata4['Human\n(within)'] = human_ermse
    if model_ermse:
        boxdata4['NRM'] = model_ermse

    if boxdata4:
        _boxplot_panel(
            ax4, boxdata4,
            ylabel='eRMSE with human data (%)',
            title=f'Figure 4 equivalent — Curve-fit Error\n{scenario}',
            colors={'Human\n(within)': '#4472C4', 'NRM': '#ED7D31'},
            ylim=(0, None),
        )
        # Dotted horizontal line at median of Human (following paper style)
        if human_ermse:
            med_h = float(np.nanmedian(human_ermse))
            ax4.axhline(med_h, color='#4472C4', ls=':', lw=1.5, alpha=0.7,
                        label=f'Human median: {med_h:.2f}%')
            ax4.legend(fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No per-observer data available\n'
                 '(no srids found in data)',
                 ha='center', va='center', transform=ax4.transAxes,
                 fontsize=9, color='gray',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_title(f'Figure 4 equivalent — Curve-fit Error\n{scenario}')

    plt.tight_layout()
    p4 = save_dir / f'{prefix}_fig4_rmse_boxplot.png'
    fig4.savefig(p4, dpi=200, bbox_inches='tight')
    plt.close(fig4)
    saved['fig4_rmse_boxplot'] = p4
    print(f"  [SAT Fig4] Saved {display_path(p4)}")

    # ── Figure 5: Correlation Boxplot ─────────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(6, 5))

    model_sp = metrics['spearman']['model_per_obs']
    human_sp = metrics['spearman']['human_per_obs']
    boxdata5: dict[str, list[float]] = {}
    if human_sp:
        boxdata5['Human\n(within)'] = human_sp
    if model_sp:
        boxdata5['NRM'] = model_sp

    if boxdata5:
        _boxplot_panel(
            ax5, boxdata5,
            ylabel='Category-wise Spearman ρ',
            title=f'Figure 5 equivalent — Category-wise Correlation\n{scenario}',
            colors={'Human\n(within)': '#4472C4', 'NRM': '#ED7D31'},
            ylim=(-1.15, 1.15),
            hline=0.0,
        )
        if human_sp:
            med_h = float(np.nanmedian(human_sp))
            ax5.axhline(med_h, color='#4472C4', ls=':', lw=1.5, alpha=0.7,
                        label=f'Human median: {med_h:.3f}')
            ax5.legend(fontsize=8)
    else:
        ax5.text(0.5, 0.5, 'No per-observer data available',
                 ha='center', va='center', transform=ax5.transAxes,
                 fontsize=9, color='gray',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax5.set_title(f'Figure 5 equivalent — Category-wise Correlation\n{scenario}')

    plt.tight_layout()
    p5 = save_dir / f'{prefix}_fig5_corr_boxplot.png'
    fig5.savefig(p5, dpi=200, bbox_inches='tight')
    plt.close(fig5)
    saved['fig5_corr_boxplot'] = p5
    print(f"  [SAT Fig5] Saved {display_path(p5)}")

    # ── Figure 6: Category Barplot ────────────────────────────────────────
    fig6, ax6 = plt.subplots(figsize=(12, 5))

    cat_med_m = metrics['spearman_cat_median']
    cat_med_h = metrics.get('spearman_cat_median_human', {})

    x      = np.arange(n_choices)
    width  = 0.38
    vals_m = [cat_med_m.get(c, float('nan')) for c in range(n_choices)]
    vals_h = [cat_med_h.get(c, float('nan')) for c in range(n_choices)]

    # Bar width set to 0 when no per-observer data
    has_obs = any(not np.isnan(v) for v in vals_m)
    if has_obs:
        bars_h = ax6.bar(x - width / 2, vals_h, width, label='Human (within)',
                         color='#4472C4', alpha=0.75, edgecolor='black', linewidth=0.4)
        bars_m = ax6.bar(x + width / 2, vals_m, width, label='NRM',
                         color='#ED7D31', alpha=0.75, edgecolor='black', linewidth=0.4)
        # Clip negative bars for visualisation (following paper)
        for bar in bars_h:
            if bar.get_height() < 0:
                bar.set_height(0)
                bar.set_linestyle('--')
        for bar in bars_m:
            if bar.get_height() < 0:
                bar.set_height(0)
                bar.set_linestyle('--')
    else:
        ax6.text(0.5, 0.5, 'No per-observer data available',
                 ha='center', va='center', transform=ax6.transAxes,
                 fontsize=10, color='gray')

    ax6.axhline(0, color='black', lw=0.8)
    ax6.set_xticks(x)
    cat_labels = [CATEGORIES[c] if c < len(CATEGORIES) else str(c) for c in range(n_choices)]
    ax6.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=9)
    ax6.set_ylabel('Median Spearman ρ (across observers)', fontsize=9)
    ax6.set_ylim([-0.1, 1.05])
    ax6.set_title(f'Figure 6 equivalent — Category-wise Correlation per Category\n{scenario}',
                  fontsize=10)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    p6 = save_dir / f'{prefix}_fig6_cat_barplot.png'
    fig6.savefig(p6, dpi=200, bbox_inches='tight')
    plt.close(fig6)
    saved['fig6_cat_barplot'] = p6
    print(f"  [SAT Fig6] Saved {display_path(p6)}")

    # ── Figure 7: Steepness Line Plot (log-scale) ─────────────────────────
    fig7, ax7 = plt.subplots(figsize=(8, 5))

    steepness = metrics['steepness']

    def _level_numeric(lv: str) -> float:
        try:
            return float(lv)
        except ValueError:
            return {'c': 0.0, 'g': 1.0}.get(lv, float('nan'))

    x_vals  = [_level_numeric(lv) for lv in ordered_lvls]
    x_lbls  = [display.get(lv, lv) for lv in ordered_lvls]

    h_kap   = [steepness.get(lv, {}).get('human', (float('nan'), float('nan')))[0]
               for lv in ordered_lvls]
    m_kap   = [steepness.get(lv, {}).get('model', (float('nan'), float('nan')))[0]
               for lv in ordered_lvls]
    h_sem   = [steepness.get(lv, {}).get('human', (float('nan'), float('nan')))[1]
               for lv in ordered_lvls]
    m_sem   = [steepness.get(lv, {}).get('model', (float('nan'), float('nan')))[1]
               for lv in ordered_lvls]

    h_pts = [(x, k, s) for x, k, s in zip(x_vals, h_kap, h_sem)
             if not np.isnan(k) and k > 0]
    m_pts = [(x, k, s) for x, k, s in zip(x_vals, m_kap, m_sem)
             if not np.isnan(k) and k > 0]

    if h_pts:
        hx, hk, hs = zip(*h_pts)
        ax7.errorbar(hx, hk, yerr=hs, fmt='o-', color='#4472C4',
                     label='Human', capsize=5, linewidth=2.0, markersize=8)
    if m_pts:
        mx, mk, ms = zip(*m_pts)
        ax7.errorbar(mx, mk, yerr=ms, fmt='s--', color='#ED7D31',
                     label='NRM', capsize=5, linewidth=2.0, markersize=8)

    if not h_pts and not m_pts:
        ax7.text(0.5, 0.5, 'Weibull fit failed (too few points or divergent fit)',
                 ha='center', va='center', transform=ax7.transAxes,
                 fontsize=9, color='gray',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    try:
        ax7.set_yscale('log')
    except Exception:
        pass

    ax7.set_xticks(x_vals)
    ax7.set_xticklabels(x_lbls, fontsize=10)
    ax7.set_xlabel('Perturbation level', fontsize=10)
    ax7.set_ylabel('Steepness κ (log scale)', fontsize=10)
    ax7.set_title(f'Figure 7 equivalent — SAT Curve Steepness\n{scenario}', fontsize=10)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.25, which='both')
    plt.tight_layout()
    p7 = save_dir / f'{prefix}_fig7_steepness.png'
    fig7.savefig(p7, dpi=200, bbox_inches='tight')
    plt.close(fig7)
    saved['fig7_steepness'] = p7
    print(f"  [SAT Fig7] Saved {display_path(p7)}")

    return saved


def create_ll_comparison_plot(human_human_results, model_results, comparison,
                               save_path='ll_comparison.png'):
    """
    Human-human vs model LL comparison plot.
    Supports int keys (non-SAT) and tuple keys (SAT conditioned).
    Does not use external LaTeX.
    """
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    import matplotlib.pyplot as plt

    conditioned = human_human_results.get('conditioned', False)
    balanced    = model_results.get('balanced', False)
    suffix = ''
    if conditioned:
        suffix = ' (cond. per beep' + (', balanced)' if balanced else ')')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Bar chart
    if comparison['n_common_stimuli'] > 0:
        lls    = [comparison['human_human_mean_ll'], comparison['model_mean_ll']]
        colors = ['royalblue', 'forestgreen']
        bars   = axes[0, 0].bar(['Human-Human\n(LOSO)', 'Model'], lls,
                                color=colors, alpha=0.75)
        axes[0, 0].set_ylabel('Mean Log-Likelihood')
        axes[0, 0].set_title('RT Predictive Performance' + suffix)
        axes[0, 0].axhline(0, color='black', linewidth=0.7)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for bar, ll in zip(bars, lls):
            offset = 0.01 * abs(ll) if ll else 0.02
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2., ll + offset,
                            f'{ll:.3f}', ha='center', va='bottom', fontweight='bold')
        adv = comparison['model_advantage']
        txt = (f'Model better\n(+{adv:.3f} LL)' if adv > 0
               else f'HH better\n({adv:.3f} LL)')
        fc  = 'lightgreen' if adv > 0 else 'lightblue'
        axes[0, 0].text(0.5, 0.15, txt, ha='center', va='center',
                        transform=axes[0, 0].transAxes,
                        bbox=dict(boxstyle='round', facecolor=fc, alpha=0.5), fontsize=11)

    # 2. Scatter
    if comparison['n_common_stimuli'] > 0:
        keys    = list(comparison['per_stimulus_advantage'].keys())
        hh_lls  = [human_human_results['per_stimulus_mean_ll'][k] for k in keys]
        mod_lls = [model_results['per_stimulus_mean_ll'][k]        for k in keys]
        axes[0, 1].scatter(hh_lls, mod_lls, alpha=0.55, s=40, color='steelblue')
        lo = min(min(hh_lls), min(mod_lls))
        hi = max(max(hh_lls), max(mod_lls))
        axes[0, 1].plot([lo, hi], [lo, hi], 'r--', lw=1.8, label='Equal performance')
        axes[0, 1].fill_between([lo, hi], [lo, hi], hi, alpha=0.08, color='green',
                                 label='Model better')
        axes[0, 1].fill_between([lo, hi], lo, [lo, hi], alpha=0.08, color='royalblue',
                                 label='HH better')
        axes[0, 1].set_xlabel('Human-Human LL (LOSO)')
        axes[0, 1].set_ylabel('Model LL')
        axes[0, 1].set_title(
            f'Per-Stimulus LL Comparison\n({comparison["n_common_stimuli"]} stimuli{suffix})')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Distribution of advantages
    if comparison['n_common_stimuli'] > 0:
        adv_list = list(comparison['per_stimulus_advantage'].values())
        mean_adv = float(np.mean(adv_list))
        axes[1, 0].hist(adv_list, bins=30, alpha=0.72,
                        color='mediumpurple', edgecolor='black', lw=0.5)
        axes[1, 0].axvline(0, color='red', ls='--', lw=2, label='No difference')
        axes[1, 0].axvline(mean_adv, color='orange', ls='-', lw=2,
                            label=f'Mean: {mean_adv:.3f}')
        axes[1, 0].set_xlabel('Model Advantage (Model LL - HH LL)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Model Advantages' + suffix)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Pie
    if comparison['n_common_stimuli'] > 0:
        counts = [comparison['model_wins'],
                  comparison['human_human_wins'],
                  comparison['ties']]
        labels = ['Model\nBetter', 'Human-Human\nBetter', 'Ties']
        cpie   = ['forestgreen', 'royalblue', 'silver']
        trip   = [(c, l, cl) for c, l, cl in zip(counts, labels, cpie) if c > 0]
        if trip:
            c_nz, l_nz, cl_nz = zip(*trip)
            _, _, ats = axes[1, 1].pie(c_nz, labels=l_nz, colors=cl_nz,
                                        autopct='%1.1f%%', startangle=90)
            for at in ats:
                at.set_color('white'); at.set_fontweight('bold'); at.set_fontsize(10)
        wr = comparison['win_rate'] * 100
        axes[1, 1].set_title(f'Win/Loss Summary\n(Model win rate: {wr:.1f}%{suffix})')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def create_ll_comparison_plot_per_beep(human_human_results, model_results,
                                        save_path='ll_comparison_per_beep.png'):
    """
    Separate plot per beep (SAT conditioned only).

    For each unique beep: scatter of HH-LL vs Model-LL of the stimuli
    at that beep + identity line + statistics.
    This shows whether the misalignment is uniform or concentrated.

    DEPRECATED for direct use: prefer create_ll_comparison_plots_per_beep_separate
    which writes a separate file with the full 2x2 layout for every beep.
    """
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    import matplotlib.pyplot as plt

    if not human_human_results.get('conditioned', False):
        warnings.warn("create_ll_comparison_plot_per_beep is for conditioned results.")
        return None

    hh_mean  = human_human_results['per_stimulus_mean_ll']
    mod_mean = model_results['per_stimulus_mean_ll']
    balanced = model_results.get('balanced', False)

    # Group keys by beep
    beep_to_keys = defaultdict(list)
    for key in set(hh_mean.keys()) & set(mod_mean.keys()):
        beep_to_keys[key[1]].append(key)

    beeps_sorted = sorted(beep_to_keys.keys())
    n_beeps      = len(beeps_sorted)
    if n_beeps == 0:
        return None

    n_cols   = min(n_beeps, 3)
    n_rows   = (n_beeps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5.5 * n_cols, 5.2 * n_rows),
                              squeeze=False)
    axes_flat = axes.flatten()

    for bi, beep in enumerate(beeps_sorted):
        ax   = axes_flat[bi]
        keys = beep_to_keys[beep]

        hh_lls  = [hh_mean[k]  for k in keys]
        mod_lls = [mod_mean[k] for k in keys]

        ax.scatter(hh_lls, mod_lls, alpha=0.65, s=45, color='steelblue', zorder=3)

        lo = min(min(hh_lls), min(mod_lls))
        hi = max(max(hh_lls), max(mod_lls))
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.8, label='Equal perf.')
        ax.fill_between([lo, hi], [lo, hi], hi, alpha=0.08, color='green')
        ax.fill_between([lo, hi], lo, [lo, hi], alpha=0.08, color='royalblue')

        adv_list = [mod_mean[k] - hh_mean[k] for k in keys]
        mw       = sum(1 for v in adv_list if v > 0)
        hw       = sum(1 for v in adv_list if v < 0)
        mean_a   = float(np.mean(adv_list))

        bal_txt = ', balanced' if balanced else ''
        ax.set_xlabel('Human-Human LL (LOSO)', fontsize=9)
        ax.set_ylabel('Model LL', fontsize=9)
        ax.set_title(
            f'Beep = {beep:.3f} s  |  n = {len(keys)}{bal_txt}\n'
            f'Model wins: {mw}/{len(keys)}  |  HH wins: {hw}/{len(keys)}\n'
            f'Mean advantage: {mean_a:.3f}',
            fontsize=9,
        )
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    for j in range(n_beeps, len(axes_flat)):
        axes_flat[j].set_visible(False)

    bal_label = ' (balanced)' if balanced else ''
    fig.suptitle(
        f'Per-Stimulus LL Comparison per Beep{bal_label}\n'
        f'(conditioned on (image, beep))',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def create_ll_comparison_plots_per_beep_separate(
        human_human_results, model_results,
        save_dir='.', filename_prefix='rt_likelihood_comparison_beep'):
    """
    SAT-only: writes a separate PNG file for every beep value.

    Each file has the full 2x2 layout: mean-LL bar chart, per-observation
    scatter, advantage histogram, win/loss pie.

    No grouping by image: every point in the scatter is
    ONE single human observation (X = HH-LOSO LL, Y = Model LL of the
    same observation). The bar chart and the summary statistics use
    the mean over all observations at the beep, without pre-aggregating by image.

    Returns
    -------
    list[str]
        Paths of the saved files (one per beep, in increasing beep order).
    """
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    if not human_human_results.get('conditioned', False):
        warnings.warn(
            "create_ll_comparison_plots_per_beep_separate needs "
            "conditioned (SAT) results. No file written."
        )
        return []

    # Use the raw per-observation LLs, not the per-stimulus means
    hh_per_ll  = human_human_results['per_stimulus_ll']   # (img,beep) -> list[float]
    mod_per_ll = model_results['per_stimulus_ll']          # (img,beep) -> list[float]
    balanced   = model_results.get('balanced', False)
    method     = human_human_results.get('method', '')

    # Group the shared keys by beep
    common_keys  = set(hh_per_ll.keys()) & set(mod_per_ll.keys())
    beep_to_keys = defaultdict(list)
    for key in common_keys:
        beep_to_keys[key[1]].append(key)

    beeps_sorted = sorted(beep_to_keys.keys())
    if not beeps_sorted:
        return []

    save_dir_path = _Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    saved_paths   = []

    bal_label  = ', balanced' if balanced else ''
    meth_label = f', {method}' if method else ''

    for bi, beep in enumerate(beeps_sorted):
        keys = beep_to_keys[beep]

        # Collect every individual observation at this beep
        # (no grouping by image)
        hh_obs_all    = []
        model_obs_all = []
        for key in keys:
            hh_key    = hh_per_ll[key]
            model_key = mod_per_ll[key]
            n_pairs   = min(len(hh_key), len(model_key))
            hh_obs_all.extend(hh_key[:n_pairs])
            model_obs_all.extend(model_key[:n_pairs])

        n_obs    = len(hh_obs_all)
        n_images = len(keys)

        if n_obs == 0:
            continue

        hh_arr  = np.array(hh_obs_all,    dtype=float)
        mod_arr = np.array(model_obs_all, dtype=float)

        # ── Outlier detection (Tukey IQR k=3) on combined LL distribution ──
        # Outliers are excluded from ALL sub-plots; bar chart and pie chart
        # also reflect only inlier observations so that extreme degenerate
        # lognormal fits (σ→0 → LL→−∞) do not dominate the summary.
        valid_mask, ll_lower, ll_upper, n_outliers = _ll_outlier_mask(
            hh_arr, mod_arr, k=3.0)
        hh_plot  = hh_arr[valid_mask]
        mod_plot = mod_arr[valid_mask]
        n_plot   = int(valid_mask.sum())

        hh_mean_val  = float(np.nanmean(hh_plot))   if n_plot > 0 else float('nan')
        mod_mean_val = float(np.nanmean(mod_plot))  if n_plot > 0 else float('nan')
        adv_arr      = mod_plot - hh_plot
        mean_adv     = float(np.nanmean(adv_arr))   if n_plot > 0 else float('nan')
        mw           = int(np.sum(adv_arr > 0))
        hw           = int(np.sum(adv_arr < 0))
        ties         = int(np.sum(adv_arr == 0))
        win_rate     = mw / n_plot if n_plot > 0 else 0.0

        outlier_note = (
            f' [{n_outliers} outliers excl., Tukey IQR k=3: ({ll_lower:.1f}, {ll_upper:.1f})]'
            if n_outliers > 0 else ''
        )
        beep_title = (
            f'Beep = {beep:.3f} s  |  n_obs = {n_obs}  |  n_images = {n_images}'
            f'{bal_label}{meth_label}'
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # ---- 1. Mean-LL bar chart (inliers only) ----
        lls    = [hh_mean_val, mod_mean_val]
        colors = ['royalblue', 'forestgreen']
        bars   = axes[0, 0].bar(['Human-Human\n(LOSO)', 'Model'], lls,
                                color=colors, alpha=0.75)
        axes[0, 0].set_ylabel('Mean Log-Likelihood')
        axes[0, 0].set_title(
            f'RT Predictive Performance (inliers n={n_plot})\n{beep_title}')
        axes[0, 0].axhline(0, color='black', linewidth=0.7)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for bar, ll in zip(bars, lls):
            if not np.isfinite(ll):
                continue
            offset = 0.01 * abs(ll) if ll else 0.02
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2., ll + offset,
                f'{ll:.3f}', ha='center', va='bottom', fontweight='bold')
        if np.isfinite(mean_adv):
            txt = (f'Model better\n(+{mean_adv:.3f} LL)' if mean_adv > 0
                   else f'HH better\n({mean_adv:.3f} LL)')
            fc  = 'lightgreen' if mean_adv > 0 else 'lightblue'
            axes[0, 0].text(0.5, 0.15, txt, ha='center', va='center',
                            transform=axes[0, 0].transAxes,
                            bbox=dict(boxstyle='round', facecolor=fc, alpha=0.5),
                            fontsize=11)
        if n_outliers > 0:
            axes[0, 0].text(0.5, 0.01, outlier_note.strip('[] '),
                            ha='center', va='bottom', transform=axes[0, 0].transAxes,
                            fontsize=7, color='dimgray', style='italic')

        # ---- 2. Per-observation scatter (inliers only) ----
        if n_plot > 0:
            axes[0, 1].scatter(hh_plot, mod_plot, alpha=0.35, s=18, color='steelblue')
            lo = float(np.nanmin(np.concatenate([hh_plot, mod_plot])))
            hi = float(np.nanmax(np.concatenate([hh_plot, mod_plot])))
            axes[0, 1].plot([lo, hi], [lo, hi], 'r--', lw=1.8, label='Equal performance')
            axes[0, 1].fill_between([lo, hi], [lo, hi], hi,
                                     alpha=0.08, color='green', label='Model better')
            axes[0, 1].fill_between([lo, hi], lo, [lo, hi],
                                     alpha=0.08, color='royalblue', label='HH better')
        axes[0, 1].set_xlabel('Human-Human LL (LOSO, per observation)')
        axes[0, 1].set_ylabel('Model LL (per observation)')
        axes[0, 1].set_title(
            f'Per-Observation LL Comparison (inliers n={n_plot})\n'
            f'{beep_title}{outlier_note}')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # ---- 3. Per-observation advantage histogram (inliers only) ----
        if len(adv_arr) > 0:
            axes[1, 0].hist(adv_arr, bins=min(40, max(5, n_plot // 10)),
                            alpha=0.72, color='mediumpurple',
                            edgecolor='black', lw=0.5)
            axes[1, 0].axvline(0, color='red', ls='--', lw=2, label='No difference')
            if np.isfinite(mean_adv):
                axes[1, 0].axvline(mean_adv, color='orange', ls='-', lw=2,
                                    label=f'Mean: {mean_adv:.3f}')
        axes[1, 0].set_xlabel('Model Advantage (Model LL - HH LL, per observation)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(
            f'Distribution of Advantages (inliers n={n_plot})\n'
            f'{beep_title}{outlier_note}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # ---- 4. Per-observation win/loss pie (inliers only) ----
        counts = [mw, hw, ties]
        labels = ['Model\nBetter', 'Human-Human\nBetter', 'Ties']
        cpie   = ['forestgreen', 'royalblue', 'silver']
        trip   = [(c, l, cl) for c, l, cl in zip(counts, labels, cpie) if c > 0]
        if trip:
            c_nz, l_nz, cl_nz = zip(*trip)
            _, _, ats = axes[1, 1].pie(c_nz, labels=l_nz, colors=cl_nz,
                                        autopct='%1.1f%%', startangle=90)
            for at in ats:
                at.set_color('white')
                at.set_fontweight('bold')
                at.set_fontsize(10)
        axes[1, 1].set_title(
            f'Win/Loss (inliers n={n_plot}, win rate: {win_rate*100:.1f}%)\n'
            f'{beep_title}{outlier_note}')

        plt.tight_layout()
        fname = save_dir_path / f'{filename_prefix}_{bi+1:02d}_{beep:.3f}s.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(str(fname))

    return saved_paths


def create_ll_comparison_plots_beep_from_pooled(
        pooled_results,
        save_dir='.',
        filename_prefix='rt_likelihood_comparison_beep'):
    """
    SAT-only: writes one PNG file per beep with the classic 2x2 layout
    (LL bar, per-observation scatter, advantage histogram, pie)
    using data pooled per beep (no grouping by image).

    Difference from conditioning on (image, beep)
    ---------------------------------------------
    Conditioning on (image, beep) makes every scatter point the *mean* LL
    of one stimulus (image × beep) — with N_human = 2-5 per cell → unstable fits.
    Here every point = one single human RT observation, with:
    - HH-LOSO: fit on N-1 human RTs at the same beep (N ~ 100-300, stable σ)
    - Model: fit on ALL model samples at that beep (N_model ~ N × 100)
    The bimodal structure from σ → 0 disappears; the comparison reflects the real
    difference between the distributions, not artefacts of sparse cells.

    Layout 2x2
    ----------
    [0,0]  Bar chart: mean LL (HH-LOSO pooled vs Model pooled)
    [0,1]  Scatter:   HH LL vs Model LL per single RT observation
    [1,0]  Histogram: distribution of the model advantage per observation
    [1,1]  Pie:       win/loss count per observation

    Returns
    -------
    list[str]: paths of the saved files.
    """
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    if not pooled_results:
        return []

    save_dir_path = _Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    saved_paths   = []

    beeps_sorted = sorted(pooled_results.keys())

    for bi, beep in enumerate(beeps_sorted):
        res      = pooled_results[beep]
        n_obs    = res['n_obs']
        n_model  = res.get('n_model_samples', 0)
        method   = res.get('method', '')
        hh_mean  = res['hh_mean']
        mod_mean = res['model_mean']
        adv      = res['model_advantage']
        ks_stat  = res.get('ks_stat', np.nan)

        # Valid pairs (no NaN on either side)
        pairs    = [(h, m) for h, m in zip(res['hh_lls'], res['model_lls'])
                    if not np.isnan(h) and not np.isnan(m)]
        hh_sc    = [p[0] for p in pairs]
        mod_sc   = [p[1] for p in pairs]
        adv_obs  = [m - h for h, m in pairs]

        mw = sum(1 for v in adv_obs if v > 0)
        hw = sum(1 for v in adv_obs if v < 0)
        ti = sum(1 for v in adv_obs if v == 0)
        wr = mw / len(adv_obs) if adv_obs else 0.0

        meth_label  = f', {method}' if method else ''
        n_pairs_str = f'{len(pairs)} observations'
        beep_title  = (
            f'Beep = {beep:.3f} s  |  N_human = {n_obs}  |  '
            f'N_model = {n_model}{meth_label}'
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # ---- [0,0] Bar chart mean LL ----
        ax    = axes[0, 0]
        lls   = [hh_mean, mod_mean]
        cols  = ['royalblue', 'forestgreen']
        bars  = ax.bar(
            [f'Human-Human\n(LOSO pooled)\nN={n_obs}',
             f'Model\n(pooled)\nN={n_model}'],
            lls, color=cols, alpha=0.75)
        ax.axhline(0, color='black', lw=0.7)
        ax.set_ylabel('Mean Log-Likelihood')
        ax.set_title(f'RT Predictive Performance — Beep Pooled\n{beep_title}')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, ll in zip(bars, lls):
            if np.isnan(ll):
                continue
            sign   = np.sign(ll) if ll != 0 else 1
            offset = 0.02 * sign
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    ll + offset, f'{ll:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        if not np.isnan(adv):
            txt = (f'Model better\n(Δ = {adv:+.3f})'
                   if adv > 0 else f'HH better\n(Δ = {adv:+.3f})')
            fc  = 'lightgreen' if adv > 0 else 'lightblue'
            ax.text(0.5, 0.88, txt, ha='center', va='top',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor=fc, alpha=0.5),
                    fontsize=10)
        if not np.isnan(ks_stat):
            ax.text(0.02, 0.02, f'KS = {ks_stat:.3f}', transform=ax.transAxes,
                    fontsize=9, color='gray')

        # ---- [0,1] Per-observation scatter ----
        ax = axes[0, 1]
        if hh_sc and mod_sc:
            ax.scatter(hh_sc, mod_sc, alpha=0.25, s=12, color='steelblue',
                       zorder=3, label=f'{n_pairs_str}')
            lo = min(min(hh_sc), min(mod_sc))
            hi = max(max(hh_sc), max(mod_sc))
            ax.plot([lo, hi], [lo, hi], 'r--', lw=1.8, label='Equal performance')
            ax.fill_between([lo, hi], [lo, hi], hi,
                             alpha=0.07, color='forestgreen', label='Model better')
            ax.fill_between([lo, hi], lo, [lo, hi],
                             alpha=0.07, color='royalblue', label='HH better')
        ax.set_xlabel('Human-Human LL (LOSO pooled)')
        ax.set_ylabel('Model LL (pooled)')
        ax.set_title(
            f'Per-Observation LL Comparison\n'
            f'({n_pairs_str})\n{beep_title}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- [1,0] Advantage histogram ----
        ax = axes[1, 0]
        if adv_obs:
            n_bins   = min(40, max(8, len(adv_obs) // 10))
            mean_adv = float(np.mean(adv_obs))
            ax.hist(adv_obs, bins=n_bins, alpha=0.72,
                    color='mediumpurple', edgecolor='black', lw=0.5)
            ax.axvline(0, color='red', ls='--', lw=2, label='No difference')
            ax.axvline(mean_adv, color='orange', ls='-', lw=2,
                       label=f'Mean: {mean_adv:.3f}')
            ax.set_xlabel('Model Advantage (Model LL − HH LL)')
            ax.set_ylabel('Count')
            ax.set_title(
                f'Distribution of Per-Observation Advantages\n{beep_title}')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        # ---- [1,1] Pie win/loss ----
        ax = axes[1, 1]
        counts = [mw, hw, ti]
        labels = ['Model\nBetter', 'HH\nBetter', 'Ties']
        cpie   = ['forestgreen', 'royalblue', 'silver']
        trip   = [(c, l, cl) for c, l, cl in zip(counts, labels, cpie) if c > 0]
        if trip:
            c_nz, l_nz, cl_nz = zip(*trip)
            _, _, ats = ax.pie(c_nz, labels=l_nz, colors=cl_nz,
                               autopct='%1.1f%%', startangle=90)
            for at in ats:
                at.set_color('white')
                at.set_fontweight('bold')
                at.set_fontsize(10)
        ax.set_title(
            f'Win/Loss per Observation\n(Model win rate: {wr*100:.1f}%)\n{beep_title}')

        plt.tight_layout()
        fname = save_dir_path / f'{filename_prefix}_{bi+1:02d}_{beep:.3f}s.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(str(fname))

    return saved_paths


def create_ll_comparison_plots_beep_pooled_separate(
        pooled_results,
        save_dir='.',
        filename_prefix='rt_likelihood_comparison_beep_pooled'):
    """
    SAT-only: a separate PNG file per beep with a 2x2 layout.

    Layout for every beep:
    [0,0] KDE overlay — human vs model RT distributions (with the beep line)
    [0,1] Q-Q plot    — model quantiles vs human quantiles
    [1,0] LL bar      — mean LL HH-LOSO vs Model (one scalar per beep)
    [1,1] RT stats    — mean and std RT for humans and model

    Why there is NO per-observation scatter
    ---------------------------------------
    When pooled, the model is fitted ONCE on all available samples; every
    human rt_i gets the same model fit, so the model LLs vary only because
    rt_i varies (f(rt_i) is monotone in the lognormal). Likewise for HH-LOSO
    with N ~ 200: removing 1 point changes the fit negligibly. The scatter would
    essentially show two 1D curves of the same variable, not a cloud of points
    informative about the comparison. The distributions (KDE, Q-Q) and the
    scalar values (LL, RT) are compared directly instead.

    Returns
    -------
    list[str]: paths of the saved files.
    """
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    if not pooled_results:
        return []

    save_dir_path = _Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    saved_paths   = []

    beeps_sorted = sorted(pooled_results.keys())

    for bi, beep in enumerate(beeps_sorted):
        res       = pooled_results[beep]
        n_obs     = res['n_obs']
        n_model   = res.get('n_model_samples', 0)
        method    = res.get('method', '')
        hh_mean   = res['hh_mean']
        mod_mean  = res['model_mean']
        adv       = res['model_advantage']
        ks_stat   = res.get('ks_stat', np.nan)
        ks_pval   = res.get('ks_pval', np.nan)
        h_pct     = np.array(res.get('human_percentiles', []), dtype=float)
        m_pct     = np.array(res.get('model_percentiles', []), dtype=float)
        h_mean_rt = res.get('human_mean_rt', np.nan)
        h_std_rt  = res.get('human_std_rt',  np.nan)
        m_mean_rt = res.get('model_mean_rt', np.nan)
        m_std_rt  = res.get('model_std_rt',  np.nan)

        human_rts   = np.array(res['rts'],            dtype=float)
        model_rts   = np.array(res.get('model_samples', []), dtype=float)

        meth_label = f', {method}' if method else ''
        beep_title = (
            f'Beep = {beep:.3f} s  |  N_human = {n_obs}  |  '
            f'N_model = {n_model}{meth_label}'
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # ---- [0,0] KDE overlay ----
        ax = axes[0, 0]
        lo_rt = max(0.001, min(human_rts.min(),
                               model_rts.min() if len(model_rts) else human_rts.min()) * 0.85)
        hi_rt = max(human_rts.max(),
                    model_rts.max() if len(model_rts) else human_rts.max()) * 1.10
        x_grid = np.linspace(lo_rt, hi_rt, 400)

        if len(human_rts) > 1:
            kde_h = gaussian_kde(human_rts, bw_method='scott')
            ax.plot(x_grid, kde_h(x_grid), color='royalblue', lw=2.5,
                    label=f'Human (N={n_obs})')
            ax.fill_between(x_grid, kde_h(x_grid), alpha=0.15, color='royalblue')

        if len(model_rts) > 1:
            kde_m = gaussian_kde(model_rts, bw_method='scott')
            ax.plot(x_grid, kde_m(x_grid), color='tomato', lw=2.5, ls='--',
                    label=f'Model (N={n_model})')
            ax.fill_between(x_grid, kde_m(x_grid), alpha=0.15, color='tomato')

        ax.axvline(beep, color='gray', ls=':', lw=1.8,
                   label=f'Time limit ({beep:.2f} s)')
        ks_txt = (f'KS = {ks_stat:.3f},  p = {ks_pval:.3f}'
                  if not (np.isnan(ks_stat) or np.isnan(ks_pval)) else 'KS: n/a')
        ax.set_xlabel('RT (s)')
        ax.set_ylabel('Density')
        ax.set_title(f'RT Distributions (KDE)\n{beep_title}\n{ks_txt}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ---- [0,1] Q-Q plot ----
        ax = axes[0, 1]
        if len(h_pct) > 0 and len(m_pct) > 0 and not np.any(np.isnan(m_pct)):
            ax.scatter(h_pct, m_pct, s=30, color='steelblue', alpha=0.8,
                       zorder=3, label='Percentile pairs')
            lo_q = min(h_pct.min(), m_pct.min())
            hi_q = max(h_pct.max(), m_pct.max())
            ax.plot([lo_q, hi_q], [lo_q, hi_q], 'r--', lw=1.8, label='y = x (equal)')
            ax.fill_between([lo_q, hi_q], [lo_q, hi_q], hi_q,
                             alpha=0.08, color='tomato',  label='Model > Human')
            ax.fill_between([lo_q, hi_q], lo_q, [lo_q, hi_q],
                             alpha=0.08, color='royalblue', label='Human > Model')
        ax.set_xlabel('Human RT Quantiles (s)')
        ax.set_ylabel('Model RT Quantiles (s)')
        ax.set_title(f'Q-Q Plot (Model vs Human)\n{beep_title}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- [1,0] Mean LL bar chart ----
        ax = axes[1, 0]
        lls    = [hh_mean, mod_mean]
        colors = ['royalblue', 'tomato']
        bars   = ax.bar(
            [f'Human-Human\n(LOSO pooled)\nN={n_obs}',
             f'Model\n(full samples)\nN={n_model}'],
            lls, color=colors, alpha=0.75)
        ax.axhline(0, color='black', lw=0.7)
        ax.set_ylabel('Mean Log-Likelihood')
        ax.set_title(f'Mean RT Log-Likelihood\n{beep_title}')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, ll in zip(bars, lls):
            if np.isnan(ll):
                continue
            sign   = np.sign(ll) if ll != 0 else 1
            offset = 0.02 * sign
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    ll + offset, f'{ll:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        if not np.isnan(adv):
            txt = (f'Model better\n(Δ = {adv:+.3f})'
                   if adv > 0 else f'HH better\n(Δ = {adv:+.3f})')
            fc  = 'lightgreen' if adv > 0 else 'lightblue'
            ax.text(0.5, 0.88, txt, ha='center', va='top',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor=fc, alpha=0.5),
                    fontsize=10)

        # ---- [1,1] RT statistics bar chart ----
        ax  = axes[1, 1]
        ax2 = ax.twinx()
        x   = np.array([0.0, 1.0])
        w   = 0.32
        b1  = ax.bar( x - w / 2, [h_mean_rt, m_mean_rt], w,
                      color=['royalblue', 'tomato'], alpha=0.75, label='Mean RT')
        b2  = ax2.bar(x + w / 2, [h_std_rt, m_std_rt], w,
                      color=['cornflowerblue', 'lightcoral'], alpha=0.75, label='Std RT')
        ax.set_xticks(x)
        ax.set_xticklabels(['Human', 'Model'])
        ax.set_ylabel('Mean RT (s)')
        ax2.set_ylabel('Std RT (s)')
        ax.set_title(f'RT Statistics (Mean & Std)\n{beep_title}')
        for bar, v in zip(b1, [h_mean_rt, m_mean_rt]):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        v * 1.01, f'{v:.3f}s', ha='center', fontsize=9)
        for bar, v in zip(b2, [h_std_rt, m_std_rt]):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width() / 2.,
                         v * 1.01, f'{v:.3f}s', ha='center', fontsize=9)
        lines1, lbl1 = ax.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axvline(0.5, color='gray', ls=':', lw=0.8, alpha=0.4)  # separator

        fig.suptitle(
            f'RT Distribution Comparison — Beep Pooled\n{beep_title}',
            fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        fname = save_dir_path / f'{filename_prefix}_{bi+1:02d}_{beep:.3f}s.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(str(fname))

    return saved_paths


def create_ll_comparison_plots_beep_cross_image(
        cross_results,
        save_dir='.',
        filename_prefix='rt_likelihood_comparison_beep_cross'):
    """
    SAT-only: 5 PNG files (one per beep) with a 2x2 layout.

    Uses the results of calculate_hh_model_ll_beep_cross_image.

    Layout for every beep:
    [0,0]  Bar chart: mean LL (HH leave-one-img-out vs Model per-img)
    [0,1]  Scatter:   HH LL vs Model LL — one point = mean per image
                      → a non-degenerate 2D cloud (varies across images)
    [1,0]  Histogram: distribution of the advantage per image
    [1,1]  Pie:       win/loss per image

    Returns
    -------
    list[str]: paths of the saved files.
    """
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    if not cross_results:
        return []

    save_dir_path = _Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    saved_paths   = []

    for bi, beep in enumerate(sorted(cross_results.keys())):
        res      = cross_results[beep]
        n_imgs   = res['n_images']
        n_obs    = res['n_obs']
        method   = res.get('method', '')
        hh_mean  = res['hh_mean']
        mod_mean = res['model_mean']
        adv      = res['model_advantage']

        hh_img_all  = np.array(res['hh_img_means'],    dtype=float)
        mod_img_all = np.array(res['model_img_means'], dtype=float)

        # ── Outlier detection (Tukey IQR k=3) on per-image LL distribution ──
        valid_mask, ll_lower, ll_upper, n_outliers = _ll_outlier_mask(
            hh_img_all, mod_img_all, k=3.0)
        hh_img  = hh_img_all[valid_mask]
        mod_img = mod_img_all[valid_mask]
        n_plot  = int(valid_mask.sum())
        adv_img = mod_img - hh_img
        mw      = int(np.sum(adv_img > 0))
        hw      = int(np.sum(adv_img < 0))
        ti      = int(np.sum(adv_img == 0))
        wr      = mw / n_plot if n_plot > 0 else 0.0

        # Recompute summary stats on inliers
        hh_mean_plot  = float(np.nanmean(hh_img))  if n_plot > 0 else float('nan')
        mod_mean_plot = float(np.nanmean(mod_img)) if n_plot > 0 else float('nan')
        adv_plot      = mod_mean_plot - hh_mean_plot if n_plot > 0 else float('nan')

        outlier_note = (
            f' [{n_outliers} outliers excl., Tukey IQR k=3: ({ll_lower:.1f}, {ll_upper:.1f})]'
            if n_outliers > 0 else ''
        )
        meth_label = f', {method}' if method else ''
        beep_title = (
            f'Beep = {beep:.3f} s  |  N_images = {n_imgs} (plot: {n_plot})  |  '
            f'N_obs = {n_obs}{meth_label}\n'
            f'HH: leave-one-image-out  |  Model: per-image conditioned'
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # ---- [0,0] Bar chart mean LL (inliers) ----
        ax   = axes[0, 0]
        lls  = [hh_mean_plot, mod_mean_plot]
        cols = ['royalblue', 'forestgreen']
        bars = ax.bar(
            ['Human-Human\n(leave-one-img-out)', 'Model\n(per-image conditioned)'],
            lls, color=cols, alpha=0.75)
        ax.axhline(0, color='black', lw=0.7)
        ax.set_ylabel('Mean Log-Likelihood')
        ax.set_title(f'RT Predictive Performance (inliers n={n_plot})\n{beep_title}')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, ll in zip(bars, lls):
            if not np.isfinite(ll):
                continue
            sign   = np.sign(ll) if ll != 0 else 1
            offset = 0.02 * sign
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    ll + offset, f'{ll:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        if np.isfinite(adv_plot):
            txt = (f'Model better\n(Δ = {adv_plot:+.3f})'
                   if adv_plot > 0 else f'HH better\n(Δ = {adv_plot:+.3f})')
            fc  = 'lightgreen' if adv_plot > 0 else 'lightblue'
            ax.text(0.5, 0.88, txt, ha='center', va='top',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor=fc, alpha=0.5),
                    fontsize=10)
        if n_outliers > 0:
            ax.text(0.5, 0.01, outlier_note.strip('[] '),
                    ha='center', va='bottom', transform=ax.transAxes,
                    fontsize=7, color='dimgray', style='italic')

        # ---- [0,1] Per-image scatter (inliers only) ----
        ax = axes[0, 1]
        if n_plot > 0:
            ax.scatter(hh_img, mod_img, alpha=0.55, s=35,
                       color='steelblue', zorder=3,
                       label=f'{n_plot} images')
            lo = float(np.nanmin(np.concatenate([hh_img, mod_img])))
            hi = float(np.nanmax(np.concatenate([hh_img, mod_img])))
            ax.plot([lo, hi], [lo, hi], 'r--', lw=1.8, label='Equal performance')
            ax.fill_between([lo, hi], [lo, hi], hi,
                             alpha=0.07, color='forestgreen', label='Model better')
            ax.fill_between([lo, hi], lo, [lo, hi],
                             alpha=0.07, color='royalblue',   label='HH better')
        ax.set_xlabel('HH LL (leave-one-image-out, mean per image)')
        ax.set_ylabel('Model LL (per-image conditioned, mean per image)')
        ax.set_title(
            f'Per-Image LL Comparison (inliers n={n_plot})\n'
            f'{beep_title}{outlier_note}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- [1,0] Per-image advantage histogram (inliers only) ----
        ax = axes[1, 0]
        if len(adv_img) > 0:
            n_bins    = min(30, max(5, n_plot // 3))
            mean_adv  = float(np.nanmean(adv_img))
            ax.hist(adv_img, bins=n_bins, alpha=0.72,
                    color='mediumpurple', edgecolor='black', lw=0.5)
            ax.axvline(0, color='red', ls='--', lw=2, label='No difference')
            ax.axvline(mean_adv, color='orange', ls='-', lw=2,
                       label=f'Mean: {mean_adv:.3f}')
        ax.set_xlabel('Model Advantage (Model LL − HH LL, per image)')
        ax.set_ylabel('Count')
        ax.set_title(
            f'Distribution of Advantages per Image (inliers n={n_plot})\n'
            f'{beep_title}{outlier_note}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # ---- [1,1] Per-image win/loss pie (inliers only) ----
        ax     = axes[1, 1]
        counts = [mw, hw, ti]
        labels = ['Model\nBetter', 'HH\nBetter', 'Ties']
        cpie   = ['forestgreen', 'royalblue', 'silver']
        trip   = [(c, l, cl) for c, l, cl in zip(counts, labels, cpie) if c > 0]
        if trip:
            c_nz, l_nz, cl_nz = zip(*trip)
            _, _, ats = ax.pie(c_nz, labels=l_nz, colors=cl_nz,
                               autopct='%1.1f%%', startangle=90)
            for at in ats:
                at.set_color('white')
                at.set_fontweight('bold')
                at.set_fontsize(10)
        ax.set_title(
            f'Win/Loss (inliers n={n_plot}, win rate: {wr*100:.1f}%)\n'
            f'{beep_title}{outlier_note}')

        fig.suptitle(
            f'RT Likelihood: Cross-Image Reference — Beep {beep:.3f} s',
            fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = save_dir_path / f'{filename_prefix}_{bi+1:02d}_{beep:.3f}s.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(str(fname))


def create_distribution_comparison_plots(human_human_metrics, model_human_metrics, 
                                         save_path='rt_distribution_comparison.png'):
    """
    Figure of the distribution-comparison metrics.
    
    Args:
        human_human_metrics: Results from compare_human_human_distributions
        model_human_metrics: Results from compare_model_human_distributions
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    mh = model_human_metrics['overall']
    per_stim = model_human_metrics['per_stimulus']
    
    # 1. KS Distance Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ks_values = [m['ks']['ks_statistic'] for m in per_stim.values() if 'ks' in m]
    if len(ks_values) > 0:
        ax1.hist(ks_values, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(mh['ks_distance']['mean'], color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {mh["ks_distance"]["mean"]:.3f}')
        ax1.axvline(mh['ks_distance']['median'], color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {mh["ks_distance"]["median"]:.3f}')
        ax1.set_xlabel('KS Distance')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of KS Distances\n(Model vs Human)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. KS p-values
    ax2 = fig.add_subplot(gs[0, 1])
    ks_pvalues = [m['ks']['ks_pvalue'] for m in per_stim.values() if 'ks' in m]
    if len(ks_pvalues) > 0:
        ax2.hist(ks_pvalues, bins=20, alpha=0.7, color='coral', edgecolor='black')
        ax2.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
        n_sig = sum(1 for p in ks_pvalues if p < 0.05)
        ax2.set_xlabel('KS Test p-value')
        ax2.set_ylabel('Count')
        ax2.set_title(f"KS Test p-values\n"
    f"({n_sig}/{len(ks_pvalues)} significant at $\\alpha=0.05$)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Quantile MAE
    ax3 = fig.add_subplot(gs[0, 2])
    quantile_maes = [m['quantiles']['mean_absolute_error'] for m in per_stim.values() 
                    if 'quantiles' in m]
    if len(quantile_maes) > 0:
        ax3.hist(quantile_maes, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(mh['quantile_mae']['mean'], color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {mh["quantile_mae"]["mean"]:.3f}')
        ax3.set_xlabel('Quantile MAE (s)')
        ax3.set_ylabel('Count')
        ax3.set_title('Quantile Matching Error')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Mean differences
    ax4 = fig.add_subplot(gs[1, 0])
    mean_diffs = [m['moments']['differences']['mean_diff'] for m in per_stim.values() 
                 if 'moments' in m]
    if len(mean_diffs) > 0:
        ax4.hist(mean_diffs, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect match')
        ax4.axvline(mh['moment_differences']['mean']['mean'], color='orange', 
                   linestyle='--', linewidth=2, 
                   label=f'Mean: {mh["moment_differences"]["mean"]["mean"]:.3f}')
        ax4.set_xlabel('Mean Difference (s)\n(Model - Human)')
        ax4.set_ylabel('Count')
        ax4.set_title('Mean RT Differences')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Std differences
    ax5 = fig.add_subplot(gs[1, 1])
    std_diffs = [m['moments']['differences']['std_diff'] for m in per_stim.values() 
                if 'moments' in m]
    if len(std_diffs) > 0:
        ax5.hist(std_diffs, bins=20, alpha=0.7, color='teal', edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect match')
        ax5.axvline(mh['moment_differences']['std']['mean'], color='orange', 
                   linestyle='--', linewidth=2,
                   label=f'Mean: {mh["moment_differences"]["std"]["mean"]:.3f}')
        ax5.set_xlabel('Std Difference (s)\n(Model - Human)')
        ax5.set_ylabel('Count')
        ax5.set_title('Standard Deviation Differences')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Skewness differences
    ax6 = fig.add_subplot(gs[1, 2])
    skew_diffs = [m['moments']['differences']['skewness_diff'] for m in per_stim.values() 
                 if 'moments' in m]
    if len(skew_diffs) > 0:
        ax6.hist(skew_diffs, bins=20, alpha=0.7, color='brown', edgecolor='black')
        ax6.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect match')
        ax6.axvline(mh['moment_differences']['skewness']['mean'], color='orange', 
                   linestyle='--', linewidth=2,
                   label=f'Mean: {mh["moment_differences"]["skewness"]["mean"]:.3f}')
        ax6.set_xlabel('Skewness Difference\n(Model - Human)')
        ax6.set_ylabel('Count')
        ax6.set_title('Skewness Differences')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. Kurtosis differences
    ax7 = fig.add_subplot(gs[2, 0])
    kurt_diffs = [m['moments']['differences']['kurtosis_diff'] for m in per_stim.values() 
                 if 'moments' in m]
    if len(kurt_diffs) > 0:
        ax7.hist(kurt_diffs, bins=20, alpha=0.7, color='olive', edgecolor='black')
        ax7.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect match')
        ax7.axvline(mh['moment_differences']['kurtosis']['mean'], color='orange', 
                   linestyle='--', linewidth=2,
                   label=f'Mean: {mh["moment_differences"]["kurtosis"]["mean"]:.3f}')
        ax7.set_xlabel('Kurtosis Difference\n(Model - Human)')
        ax7.set_ylabel('Count')
        ax7.set_title('Kurtosis Differences\n(Excess Kurtosis)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. Summary table
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    summary_text = f"""
    DISTRIBUTION COMPARISON SUMMARY
    {'='*60}
    
    Kolmogorov-Smirnov Distance:
      Mean:   {mh['ks_distance']['mean']:.4f} (lower is better, 0 = perfect)
      Median: {mh['ks_distance']['median']:.4f}
      Std:    {mh['ks_distance']['std']:.4f}
    
    Quantile Matching (MAE):
      Mean:   {mh['quantile_mae']['mean']:.4f}s
      Median: {mh['quantile_mae']['median']:.4f}s
    
    Moment Differences (Model - Human):
      Mean:     {mh['moment_differences']['mean']['mean']:.4f}s ± {mh['moment_differences']['mean']['std']:.4f}
      Std:      {mh['moment_differences']['std']['mean']:.4f}s ± {mh['moment_differences']['std']['std']:.4f}
      Skewness: {mh['moment_differences']['skewness']['mean']:.4f} ± {mh['moment_differences']['skewness']['std']:.4f}
      Kurtosis: {mh['moment_differences']['kurtosis']['mean']:.4f} ± {mh['moment_differences']['kurtosis']['std']:.4f}
    
    Number of stimuli evaluated: {mh['n_stimuli_evaluated']}
    """
    
    ax8.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('RT Distribution Comparison: Model vs Human', fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

