"""The three benchmark figures of the paper, built from computed results.

===========================  ==========================  =================
Figure                       function                    notebook
===========================  ==========================  =================
``RDM_bench.pdf``            :func:`plot_rdm_bench`      ``rdm.ipynb``
``SC_bench.pdf``             :func:`plot_sc_bench`       ``sc.ipynb``
``SAT_bench.pdf``            :func:`plot_sat_bench`      ``sat.ipynb``
``LDA.pdf``                  ``scripts.difficulty_parameter_correlation.plot_lda2d``
===========================  ==========================  =================

``RDM_bench`` and ``SC_bench`` are the two halves of the combined ``RDM_SC``
figure; :func:`stack_bench_figures` writes that combined PDF from the two
panels once both notebooks have run.

Every figure puts the human noise ceiling in front, NRM next, and the
comparators behind, so the claim the figure supports is readable at a glance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from scripts.results_io import display_path

__all__ = [
    "PDF_METADATA", "HUMAN_COLOR", "MODEL_STYLE", "RDM_MODEL_ORDER",
    "SC_MODEL_ORDER", "SAT_MODEL_ORDER", "SAT_TASK_LABELS",
    "plot_rdm_bench", "plot_sc_bench", "plot_sat_bench", "stack_bench_figures",
]

#: matplotlib stamps a ``/CreationDate`` into every PDF it writes, so the same
#: figure saved twice differs in bytes even when it differs in nothing else.
#: Omitting that one key makes the saved PDFs deterministic.
PDF_METADATA = {"CreationDate": None}

HUMAN_COLOR = "#4472C4"
HUMAN_MARKER = "o"

#: One palette across all three benchmarks, so a model keeps its colour.
MODEL_STYLE = {
    "NRM":        {"color": "#E74C3C", "marker": "D", "label": "NRM"},
    "RTify":      {"color": "#3498DB", "marker": "o", "label": "RTify"},
    "cRNN":       {"color": "#16A085", "marker": "*", "label": "cRNN"},
    "LNR":        {"color": "#2980B9", "marker": "s", "label": "LNR"},
    "LBA":        {"color": "#9B59B6", "marker": "^", "label": "LBA"},
    "DDM":        {"color": "#E67E22", "marker": "v", "label": "DDM"},
    "CNet(par.)": {"color": "#3498DB", "marker": "o", "label": "CNet (par.)"},
    "CNet(ser.)": {"color": "#2980B9", "marker": "s", "label": "CNet (ser.)"},
    "ConvRNN":    {"color": "#9B59B6", "marker": "^", "label": "ConvRNN"},
    "MSDNet":     {"color": "#E67E22", "marker": "v", "label": "MSDNet"},
    "SCAN":       {"color": "#1ABC9C", "marker": "P", "label": "SCAN"},
}

RDM_MODEL_ORDER = ["NRM", "RTify", "LNR", "LBA", "DDM"]
SC_MODEL_ORDER = ["NRM", "RTify", "cRNN", "LNR", "LBA", "DDM"]
SAT_MODEL_ORDER = ["NRM", "CNet(par.)", "CNet(ser.)", "ConvRNN", "MSDNet", "SCAN"]

SAT_TASK_LABELS = {
    "color-blur": "Color-Blur",
    "color-gray": "Color-Gray",
    "gray-noise": "Gray-Noise",
}


def _style(name):
    return MODEL_STYLE.get(
        name, {"color": "#7F8C8D", "marker": "x", "label": name})


def _emphasis(name):
    """NRM is the claim; everything else is context."""
    if name == "NRM":
        return dict(ls="-", lw=2.5, ms=7, alpha=1.0, zorder=6)
    return dict(ls="--", lw=1.5, ms=6, alpha=0.9, zorder=3)


# ─────────────────────────────────────────────────────────────────────────────
# RDM and SC: accuracy / median RT / predictive NLL against stimulus strength
# ─────────────────────────────────────────────────────────────────────────────

def _bench_three_panel(levels, per_level, human, model_order, *, xlabel,
                       log_x, titles, human_bands=None, suptitle=None):
    """Psychometric | chronometric | predictive-NLL panels on a shared x axis.

    ``per_level[model][level]`` and ``human[level]`` carry ``Acc``, ``RT``
    (median, ms) and ``NLL``; ``human_bands[metric]`` optionally gives a
    half-width per level for the shaded human spread.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = [("Acc", "Accuracy"), ("RT", "Median RT (ms)"), ("NLL", "NLL (nats)")]

    for ax, (key, ylabel), title in zip(axes, specs, titles):
        y_h = np.array([human.get(lv, {}).get(key, np.nan) for lv in levels],
                       dtype=float)
        ax.plot(levels, y_h, "-", color=HUMAN_COLOR, marker=HUMAN_MARKER,
                lw=2.5, ms=8, zorder=10, label="Human (NC)",
                markeredgewidth=0.5, markeredgecolor="white")
        band = (human_bands or {}).get(key)
        if band is not None:
            band = np.asarray(band, dtype=float)
            ax.fill_between(levels, y_h - band, y_h + band,
                            color=HUMAN_COLOR, alpha=0.15, zorder=1)

        for name in model_order:
            if name not in per_level:
                continue
            st, em = _style(name), _emphasis(name)
            y = [per_level[name].get(lv, {}).get(key, np.nan) for lv in levels]
            ax.plot(levels, y, color=st["color"], marker=st["marker"],
                    label=st["label"], markeredgewidth=0.5,
                    markeredgecolor="white", **em)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        if log_x:
            ax.set_xscale("log")
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{lv:g}" for lv in levels], fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=len(labels), fontsize=10, frameon=False)
    fig.subplots_adjust(bottom=0.22, top=0.86 if suptitle else 0.92,
                        left=0.06, right=0.98, wspace=0.30)
    return fig


def plot_rdm_bench(per_coh: dict, human: dict, levels, save_path,
                   human_bands: dict | None = None):
    """``RDM_bench.pdf`` — accuracy, median RT and NLL against coherence."""
    fig = _bench_three_panel(
        list(levels), per_coh, human, RDM_MODEL_ORDER,
        xlabel="Coherence (%)", log_x=True, human_bands=human_bands,
        titles=("Psychometric Function", "Chronometric Function",
                "Predictive Likelihood (NLL)"),
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight",
                metadata=PDF_METADATA)
    print(f"Saved -> {display_path(save_path)}")
    return fig


def plot_sc_bench(per_level: dict, human: dict, levels, save_path,
                  human_bands: dict | None = None):
    """``SC_bench.pdf`` — the same three panels against discriminability."""
    fig = _bench_three_panel(
        list(levels), per_level, human, SC_MODEL_ORDER,
        xlabel="Discriminability", log_x=False, human_bands=human_bands,
        titles=("Psychometric Function", "Chronometric Function",
                "RT Predictive Likelihood"),
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight",
                metadata=PDF_METADATA)
    print(f"Saved -> {display_path(save_path)}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SAT: accuracy against response deadline, averaged over perturbation levels
# ─────────────────────────────────────────────────────────────────────────────

def sat_mean_curve(curves: dict, source: str = "human"):
    """Mean accuracy per beep time, averaged over perturbation levels.

    ``curves`` is one ``sat_level_curves.json`` payload. Returns
    ``(times_s, accuracy_%, sem_%)``; the band is the standard error across
    perturbation levels at that deadline.
    """
    src = curves.get(source, {})
    by_beep: dict[float, list] = {}
    for level in src.get("ordered_levels", []):
        entry = (src.get("overall") or {}).get(level)
        if entry is None:
            continue
        for bp in entry["per_beep"].values():
            by_beep.setdefault(float(bp["beep_s"]), []).append(bp["acc_mean"])
    if not by_beep:
        return None, None, None
    times = np.array(sorted(by_beep))
    accs = np.array([np.nanmean(by_beep[t]) for t in times]) * 100.0
    sems = np.array([np.nanstd(by_beep[t]) / np.sqrt(max(len(by_beep[t]), 1))
                     for t in times]) * 100.0
    return times, accs, sems


#: Bottom-left corner of the *lower* inset, in axes coordinates. The curves
#: sweep up to the right everywhere except gray-noise, where accuracy stays low
#: and the free space is top-left instead — the placement the paper uses.
#: The upper inset sits SAT_INSET_H + SAT_INSET_GAP above it, so both stay
#: inside the axes.
SAT_INSET_W, SAT_INSET_H, SAT_INSET_GAP = 0.44, 0.20, 0.055
SAT_INSET_ANCHOR = {
    "color-blur": (0.52, 0.06),
    "color-gray": (0.52, 0.06),
    "gray-noise": (0.04, 0.52),
}


def _sat_metric_inset(ax, full_by_model, task, key, title, rect,
                      models, decimals=0):
    """A per-observer strip plot of one SAT metric, as an inset of `ax`.

    One translucent dot per observer plus a filled marker at the median, for
    the human ceiling and then each model. Reads the per-observer arrays the
    tables are built from (``ermse.model_per_obs`` / ``spearman.model_per_obs``
    in ``satbench_metrics_full.json``), so the inset and the table cannot
    disagree.
    """
    ins = ax.inset_axes(list(rect))
    rng = np.random.default_rng(0)      # jitter only, never the data

    columns = []
    human_payload = next((full_by_model[m][task] for m in models
                          if task in full_by_model.get(m, {})), None)
    if human_payload is not None:
        vals = human_payload.get(key, {}).get("human_per_obs", [])
        columns.append(("Human (NC)", HUMAN_COLOR, HUMAN_MARKER, vals))
    for name in models:
        payload = full_by_model.get(name, {}).get(task)
        if payload is None:
            continue
        st = _style(name)
        columns.append((st["label"], st["color"], st["marker"],
                        payload.get(key, {}).get("model_per_obs", [])))

    for i, (_, colour, marker, vals) in enumerate(columns):
        vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)],
                          dtype=float)
        if vals.size == 0:
            continue
        ins.scatter(i + rng.uniform(-0.20, 0.20, vals.size), vals, s=2,
                    color=colour, alpha=0.28, linewidths=0, zorder=2)
        ins.scatter([i], [np.median(vals)], s=28, color=colour, marker=marker,
                    edgecolors="white", linewidths=0.5, zorder=4)

    ins.set_title(title, fontsize=6.5, fontweight="bold", loc="left", pad=1.5)
    ins.set_xticks([])
    ins.set_xlim(-0.6, len(columns) - 0.4)
    ins.tick_params(axis="y", labelsize=5.5, length=1.8, pad=0.8)
    ins.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.{decimals}f}"))
    ins.grid(True, axis="y", alpha=0.15, lw=0.4)
    for side in ("top", "right"):
        ins.spines[side].set_visible(False)
    # Opaque, so the curves underneath do not read as part of the strip plot.
    ins.patch.set_facecolor("white")
    ins.patch.set_alpha(0.92)
    for side in ("left", "bottom"):
        ins.spines[side].set_linewidth(0.6)
    return ins


def plot_sat_bench(curves_by_model: dict, save_path, tasks=None,
                   n_choices: int = 16, xlim=(-0.05, 1.7),
                   full_by_model: dict | None = None):
    """``SAT_bench.pdf`` — one panel per perturbation condition.

    ``curves_by_model[model][task]`` is the ``sat_level_curves.json`` payload
    for that pair. Each curve is the accuracy averaged over the condition's
    perturbation levels, with a standard-error band across those levels; the
    human curve is read from whichever payload is available (it is the same
    behavioural data for every model).

    ``full_by_model[model][task]`` is the matching ``satbench_metrics_full``
    payload. When given, each panel carries two insets — the per-observer
    RMSE and Spearman ρ strips.
    """
    tasks = list(tasks if tasks is not None else
                 ["color-blur", "color-gray", "gray-noise"])
    chance = 100.0 / n_choices

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

    for ax, task in zip(axes, tasks):
        # Comparators first, NRM and the human ceiling on top of them.
        ordered = ([m for m in SAT_MODEL_ORDER if m != "NRM"]
                   + [m for m in SAT_MODEL_ORDER if m == "NRM"])
        human_done = False
        for name in ordered:
            curves = curves_by_model.get(name, {}).get(task)
            if curves is None:
                continue
            is_nrm = name == "NRM"

            if not human_done:
                ht, ha, hs = sat_mean_curve(curves, "human")
                if ht is not None:
                    t = np.concatenate([[0.0], ht])
                    a = np.concatenate([[chance], ha])
                    s = np.concatenate([[0.0], hs])
                    ax.plot(t, a, color=HUMAN_COLOR, lw=3.0,
                            marker=HUMAN_MARKER, ms=8, label="Human (NC)",
                            zorder=7)
                    ax.fill_between(t, a - s, a + s, color=HUMAN_COLOR,
                                    alpha=0.15, zorder=1)
                    human_done = True

            mt, ma, ms_ = sat_mean_curve(curves, "model")
            if mt is None:
                continue
            t = np.concatenate([[0.0], mt])
            a = np.concatenate([[chance], ma])
            s = np.concatenate([[0.0], ms_])
            st = _style(name)
            lw, alpha, ls, z, msz = ((2.8, 1.0, "-", 6, 7) if is_nrm
                                     else (1.2, 0.38, "--", 2, 4))
            ax.plot(t, a, color=st["color"], lw=lw, ls=ls, marker=st["marker"],
                    ms=msz, label=st["label"], alpha=alpha, zorder=z)
            ax.fill_between(t, a - s, a + s, color=st["color"],
                            alpha=0.10 if is_nrm else 0.06, zorder=1)

        if not human_done:
            ax.text(0.5, 0.5, f"no curves for {task}", ha="center",
                    transform=ax.transAxes)

        ax.axhline(chance, color="gray", ls=":", lw=1.0, alpha=0.5,
                   label=f"Chance ({chance:.2f}%)")
        ax.set_title(SAT_TASK_LABELS.get(task, task), fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("Beep time (s)", fontsize=10)
        ax.set_ylabel("Accuracy (%)" if task == tasks[0] else "", fontsize=10)
        ax.set_ylim([0, 100])
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.2, lw=0.5)

        if full_by_model:
            present = [m for m in SAT_MODEL_ORDER
                       if task in full_by_model.get(m, {})]
            x0, y0 = SAT_INSET_ANCHOR.get(task, (0.52, 0.06))
            upper = (x0, y0 + SAT_INSET_H + SAT_INSET_GAP,
                     SAT_INSET_W, SAT_INSET_H)
            lower = (x0, y0, SAT_INSET_W, SAT_INSET_H)
            _sat_metric_inset(ax, full_by_model, task, "ermse", "RMSE",
                              upper, present, decimals=0)
            _sat_metric_inset(ax, full_by_model, task, "spearman",
                              "Spearman $\\rho$", lower, present, decimals=1)

    # One legend for the whole figure, below the panels: a per-axes legend
    # would cover the curves and the insets.
    handles, labels = axes[0].get_legend_handles_labels()
    seen, uniq = set(), []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq.append((h, l))
    order = {st["label"]: i for i, st in
             enumerate(_style(m) for m in SAT_MODEL_ORDER if m != "NRM")}
    # comparators in SAT_MODEL_ORDER, then Human, NRM, Chance
    tail = {"Human (NC)": 90, "NRM": 91}
    uniq.sort(key=lambda hl: order.get(
        hl[1], tail.get(hl[1], 99 if hl[1].startswith("Chance") else 95)))
    fig.legend([h for h, _ in uniq], [l for _, l in uniq],
               loc="lower center", ncol=len(uniq), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.4,
               columnspacing=1.4)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight",
                metadata=PDF_METADATA)
    print(f"Saved -> {display_path(save_path)}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────

def stack_bench_figures(rdm_pdf, sc_pdf, out_pdf):
    """Stack ``RDM_bench.pdf`` over ``SC_bench.pdf`` into the combined
    ``RDM_SC.pdf``. Needs ``pypdf``; returns None (and says so) without it."""
    rdm_pdf, sc_pdf, out_pdf = Path(rdm_pdf), Path(sc_pdf), Path(out_pdf)
    missing = [p for p in (rdm_pdf, sc_pdf) if not p.exists()]
    if missing:
        print("Cannot stack — not produced yet: "
              + ", ".join(display_path(p) for p in missing))
        return None
    try:
        from pypdf import PdfWriter, PdfReader, Transformation
        from pypdf.generic import RectangleObject
    except ImportError:
        print("pypdf is not installed; keeping RDM_bench.pdf and SC_bench.pdf "
              "separate (RDM_SC.pdf is the two stacked).")
        return None

    top = PdfReader(str(rdm_pdf)).pages[0]
    bot = PdfReader(str(sc_pdf)).pages[0]
    w = max(float(top.mediabox.width), float(bot.mediabox.width))
    h = float(top.mediabox.height) + float(bot.mediabox.height)

    writer = PdfWriter()
    page = writer.add_blank_page(width=w, height=h)
    page.merge_transformed_page(
        top, Transformation().translate(0, float(bot.mediabox.height)))
    page.merge_page(bot)
    page.mediabox = RectangleObject((0, 0, w, h))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdf, "wb") as f:
        writer.write(f)
    print(f"Saved -> {display_path(out_pdf)}")
    return out_pdf
