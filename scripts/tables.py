"""The LaTeX tables that appear in the paper, built from computed results.

One module so the three benchmark notebooks share a single set of formatting
rules, and so every published table has exactly one generator:

================================  =========================  ==================
LaTeX label                       function                   notebook
================================  =========================  ==================
``tab:rdm_percoh``                :func:`rdm_per_coherence_table`  ``rdm.ipynb``
``tab:rdm_global``                :func:`rdm_global_table`         ``rdm.ipynb``
``tab:rdm_mechanism``             :func:`rdm_mechanism_table`      ``rdm.ipynb``
``tab:sc_perlevel``               :func:`sc_per_level_table`       ``sc.ipynb``
``tab:sc_global``                 :func:`sc_global_table`          ``sc.ipynb``
``tab:sat_perobs``                :func:`sat_per_level_table`      ``sat.ipynb``
``tab:sat_global``                :func:`sat_global_table`         ``sat.ipynb``
================================  =========================  ==================

Highlighting is **computed**, never transcribed: bold marks the best model in a
column and underline the second best, under the direction each caption states —
smallest :math:`|\\Delta|` from the human noise ceiling for the human-relative
metrics (Acc, AUC, :math:`\\kappa`, RMSE, :math:`\\mathcal{K}`), largest value
for the intrinsically human-relative ones (:math:`r`, :math:`r_{\\rm RT}`), and
lowest value for NLL. The noise-ceiling row itself never competes.

The ``Gray`` row colour and the ``\\rowcolor`` command need
``\\usepackage[table]{xcolor}`` and ``\\definecolor{Gray}{gray}{0.92}`` in the
preamble, and the multi-row model labels need ``\\usepackage{multirow}``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "RDM_COHERENCES", "SC_LEVELS", "SAT_TABLE_TASKS", "SAT_TASK_LEVELS",
    "bootstrap_ci", "bootstrap_choice_rt_cis", "rank_best_two",
    "rdm_per_coherence_table", "rdm_global_table", "rdm_mechanism_table",
    "sc_per_level_table", "sc_global_table",
    "sat_per_level_table", "sat_global_table",
]

# ── benchmark axes ───────────────────────────────────────────────────────────

#: Motion coherences of the RDM benchmark, in %.
RDM_COHERENCES = [0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2]

#: Discriminability levels of the scene-categorisation benchmark.
SC_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]

#: Column order of the SAT tables (differs from the evaluation order).
SAT_TABLE_TASKS = ["color-gray", "gray-noise", "color-blur"]

#: Perturbation levels per SAT condition, in table order.
SAT_TASK_LEVELS = {
    "color-gray": ["c", "g"],
    "gray-noise": ["0", "0.04", "0.16"],
    "color-blur": ["0", "3", "9"],
}

_SAT_TASK_HEADERS = {
    "color-gray": r"\textbf{Color \& Gray}",
    "gray-noise": r"\textbf{Gray \& Noise} ($\sigma$)",
    "color-blur": r"\textbf{Color \& Blur} ($\sigma$)",
}
_SAT_TASK_PLAIN = {
    "color-gray": r"\textbf{Color \& Gray}",
    "gray-noise": r"\textbf{Gray \& Noise}",
    "color-blur": r"\textbf{Color \& Blur}",
}
_SAT_LEVEL_HEADERS = {
    "color-gray": {"c": "Color", "g": "Gray"},
    "gray-noise": {"0": "$0$", "0.04": "$0.04$", "0.16": "$0.16$"},
    "color-blur": {"0": "$0$", "3": "$3.0$", "9": "$9.0$"},
}

#: ``(section title, [(key, multi-line LaTeX label)])`` per benchmark. The
#: section title becomes a ``\rowcolor{Gray}`` banner; an empty title emits no
#: banner. The model marked "ours" is appended last, after a ``\midrule``.
RDM_GROUPS = [
    ("Classical Sequential Sampling Approaches", [
        ("LNR", r"LNR \\ \citep{rouder2015lognormal}"),
        ("LBA", r"LBA \\ \citep{brown2008lba}"),
        ("DDM", r"DDM \\ \citep{ratcliff2008diffusion}"),
    ]),
    ("Video-Computable Approaches", [
        ("RTify", r"RTify \\ \citep{cheng2024rtify}"),
    ]),
]
SC_GROUPS = [
    ("Classical Sequential Sampling Approaches", [
        ("LNR", r"LNR \\ \citep{rouder2015lognormal}"),
        ("LBA", r"LBA \\ \citep{brown2008lba}"),
        ("DDM", r"DDM \\ \citep{ratcliff2008diffusion}"),
    ]),
    ("Image-Computable Approaches", [
        ("RTify", r"RTify \\ \citep{cheng2024rtify}"),
        ("cRNN", r"cRNN \\ \citep{goetschalckx2023computing}"),
    ]),
]
SAT_MODELS = [
    ("CNet(par.)", r"CNet (par.) \\ \citep{iuzzolino2021cascaded}"),
    ("CNet(ser.)", r"CNet (ser.) \\ \citep{iuzzolino2021cascaded}"),
    ("ConvRNN", r"ConvRNN \\ \citep{spoerer2020recurrent}"),
    ("MSDNet", r"MSDNet \\ \citep{huang2017multiscale}"),
    ("SCAN", r"SCAN \\ \citep{zhang2019scan}"),
]
OURS = ("NRM", r"NRM \\ (ours)")

_EM = r"\textemdash{}"


# ── number formatting ────────────────────────────────────────────────────────

def _finite(v) -> bool:
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _num(v, dec: int = 3) -> str:
    """A value as LaTeX, with ``$-$`` for the minus sign so it sets properly."""
    if not _finite(v):
        return _EM
    fv = float(v)
    body = format(abs(fv), f".{dec}f")
    # -0.0004 at dec=3 would print "$-$0.000"; treat a rounded-away sign as +.
    if fv < 0 and float(body) != 0.0:
        return "$-$" + body
    return body


def _strip0(s: str) -> str:
    """``0.030`` -> ``.030``; CI deviations below 1 drop the leading zero."""
    return s[1:] if s.startswith("0.") else s


def _delta_sup(value, nc, dec: int = 3) -> str:
    r"""``\textsuperscript{\textcolor{gray}{\tiny 0.008}}`` — |value - NC|."""
    if not (_finite(value) and _finite(nc)):
        return ""
    d = abs(float(value) - float(nc))
    return r"\textsuperscript{\textcolor{gray}{\tiny " + format(d, f".{dec}f") + "}}"


def _ci_sup(point, ci, dec: int = 3, strip_zero: bool = True) -> str:
    r"""``$^{+.011}_{-.012}$`` — asymmetric deviations from the point estimate."""
    if ci is None or not _finite(point):
        return ""
    lo, hi = ci
    if not (_finite(lo) and _finite(hi)):
        return ""
    up = format(abs(float(hi) - float(point)), f".{dec}f")
    dn = format(abs(float(point) - float(lo)), f".{dec}f")
    if strip_zero:
        up, dn = _strip0(up), _strip0(dn)
    return "$^{+" + up + "}_{-" + dn + "}$"


def _sem_sup(sem, dec: int = 3) -> str:
    r"""``$^{\pm.010}$`` — the standard error carried by the NLL cells."""
    if not _finite(sem):
        return ""
    return r"$^{\pm" + _strip0(format(abs(float(sem)), f".{dec}f")) + "}$"


def rank_best_two(values: dict, nc=None, mode: str = "delta"):
    """Return ``(best_key, second_key)`` under one of three orderings.

    ``mode`` is ``"delta"`` (smallest ``|value - nc|``), ``"max"`` (largest
    value) or ``"min"`` (smallest value). Non-finite entries never rank, and
    ``"delta"`` falls back to ``"max"`` when the ceiling itself is missing.
    """
    items = [(k, float(v)) for k, v in values.items() if _finite(v)]
    if not items:
        return None, None
    if mode == "delta" and _finite(nc):
        items.sort(key=lambda kv: abs(kv[1] - float(nc)))
    elif mode == "min":
        items.sort(key=lambda kv: kv[1])
    else:
        items.sort(key=lambda kv: -kv[1])
    best = items[0][0]
    second = items[1][0] if len(items) > 1 else None
    return best, second


def _emph(cell: str) -> str:
    """Bold a reference-row cell, leaving an absent value as a plain dash."""
    return cell if cell == _EM else r"\textbf{" + cell + "}"


def _mark(cell: str, key, best, second) -> str:
    if cell == _EM:
        return cell
    if key == best:
        return r"\textbf{" + cell + "}"
    if key == second:
        return r"\underline{" + cell + "}"
    return cell


def bootstrap_ci(values, B: int = 1000, seed: int = 42, stat=np.nanmedian):
    """Percentile 95% CI of ``stat`` over ``B`` resamples of ``values``.

    Used for the SAT tables, where the resampling unit is the observer.
    """
    arr = np.asarray([v for v in values if _finite(v)], dtype=float)
    if arr.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = [float(stat(rng.choice(arr, size=arr.size, replace=True)))
             for _ in range(B)]
    draws = [d for d in draws if np.isfinite(d)]
    if len(draws) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def _safe_mean(vals):
    clean = [float(v) for v in vals if _finite(v)]
    return float(np.mean(clean)) if clean else float("nan")


def _pct_ci(draws):
    arr = np.asarray([d for d in draws if _finite(d)], dtype=float)
    if arr.size < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def bootstrap_choice_rt_cis(level_values, true_labels, pred_choices, pred_probs,
                            human_rts, pred_rts, levels, auc_fn, kappa_fn,
                            B: int = 1000, seed: int = 42) -> dict:
    """95% CIs for the five global choice/RT metrics, resampling **trials**.

    All arrays are per-trial and aligned. Each of ``B`` resamples is
    re-aggregated exactly as the point estimates are: accuracy, AUC and kappa
    are computed within each stimulus level and averaged over levels, and
    ``r_RT`` is the Pearson correlation between the model's and the humans'
    per-level median RT. ``auc_fn``/``kappa_fn`` are the benchmark's own
    NaN-tolerant wrappers, passed in so the CI and the point estimate cannot
    use different definitions.

    Returns ``{"Acc": (lo, hi), "AUC": ..., "kappa": ..., "r_RT": ...}``.
    """
    from scipy.stats import pearsonr

    level_values = np.asarray(level_values, dtype=float)
    true_labels = np.asarray(true_labels)
    pred_choices = np.asarray(pred_choices)
    pred_probs = (np.asarray(pred_choices, dtype=float) if pred_probs is None
                  else np.asarray(pred_probs, dtype=float))
    human_rts = np.asarray(human_rts, dtype=float)
    pred_rts = np.asarray(pred_rts, dtype=float)
    levels = list(levels)

    rng = np.random.default_rng(seed)
    n = len(true_labels)
    draws = {k: [] for k in ("Acc", "AUC", "kappa", "r_RT")}

    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        lv, tl, pc = level_values[idx], true_labels[idx], pred_choices[idx]
        pp, hrt, prt = pred_probs[idx], human_rts[idx], pred_rts[idx]

        acc, auc, kap, m_rt, h_rt = [], [], [], [], []
        for level in levels:
            m = np.isclose(lv, level)
            if m.sum() < 2:
                continue
            acc.append(float((pc[m] == tl[m]).mean()))
            a, k = auc_fn(tl[m], pp[m]), kappa_fn(tl[m], pc[m])
            if _finite(a):
                auc.append(float(a))
            if _finite(k):
                kap.append(float(k))
            ok = hrt[m] > 0
            if ok.sum() > 0:
                h_rt.append(float(np.median(hrt[m][ok])))
                m_rt.append(float(np.median(prt[m][ok])))

        draws["Acc"].append(_safe_mean(acc))
        draws["AUC"].append(_safe_mean(auc))
        draws["kappa"].append(_safe_mean(kap))
        if len(m_rt) >= 3 and np.std(m_rt) > 0 and np.std(h_rt) > 0:
            try:
                draws["r_RT"].append(float(pearsonr(m_rt, h_rt)[0]))
            except (ValueError, FloatingPointError):
                pass

    return {k: _pct_ci(v) for k, v in draws.items()}


def _present(groups, ours, available):
    """Flatten the group spec to ``[(title_or_None, key, label)]``, dropping
    models that were not evaluated."""
    rows = []
    for title, members in groups:
        members = [(k, lbl) for k, lbl in members if k in available]
        if not members:
            continue
        rows.append((title, members))
    tail = [(None, [ours])] if ours[0] in available else []
    return rows + tail


# ─────────────────────────────────────────────────────────────────────────────
# Per-level tables (RDM tab:rdm_percoh, SC tab:sc_perlevel)
# ─────────────────────────────────────────────────────────────────────────────

# (metric key, row label, decimals, ranking mode)
_RDM_METRICS = [
    ("Acc",   "Acc",        3, "delta"),
    ("RT",    "RT",         0, "delta"),
    ("NLL",   "NLL",        3, "min"),
    ("AUC",   "AUC",        3, "delta"),
    ("kappa", r"$\kappa$",  3, "delta"),
]
_SC_METRICS = [
    ("Acc",   "Acc",        3, "delta"),
    ("AUC",   "AUC",        3, "delta"),
    ("kappa", r"$\kappa$",  3, "delta"),
    ("NLL",   "NLL",        3, "min"),
]


def _per_level_table(per_level, human, levels, metrics, groups, caption, label,
                     level_header, col_header, tabcolsep="3pt",
                     avg=None, human_avg=None):
    """Shared body of the two per-level tables.

    ``per_level[model][level][metric]`` and ``human[level][metric]`` carry the
    raw numbers; every cell is the model value plus a grey superscript holding
    its absolute deviation from the ceiling.

    ``avg[model][metric]`` and ``human_avg[metric]`` override the *Avg* column.
    Pass the values the companion global table reports, so the two tables are
    one computation rather than two; a metric left out of them falls back to
    the plain mean over levels.
    """
    sections = _present(groups, OURS, set(per_level))
    n_lv = len(levels)
    avg = avg or {}
    human_avg = human_avg or {}

    def val(model, lv, key):
        return per_level.get(model, {}).get(lv, {}).get(key, float("nan"))

    def hval(lv, key):
        return human.get(lv, {}).get(key, float("nan"))

    # Column keys: each level, then the Avg column.
    columns = list(levels) + ["avg"]

    def column_values(key, col):
        if col == "avg":
            return {m: (avg.get(m, {}).get(key)
                        if _finite(avg.get(m, {}).get(key))
                        else _safe_mean([val(m, lv, key) for lv in levels]))
                    for m in per_level}
        return {m: val(m, col, key) for m in per_level}

    def column_nc(key, col):
        if col == "avg":
            if _finite(human_avg.get(key)):
                return float(human_avg[key])
            return _safe_mean([hval(lv, key) for lv in levels])
        return hval(col, key)

    out = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{" + tabcolsep + "}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{@{} ll " + "c" * n_lv + r" c @{}}",
        r"    \toprule",
        r"    & & \multicolumn{" + str(n_lv) + r"}{c}{\textbf{" + col_header + r"}} & \\",
        r"    \cmidrule(lr){3-" + str(2 + n_lv) + "}",
        r"    \textbf{Model} & \textbf{Metric} & "
        + " & ".join(level_header(lv) for lv in levels) + r" & \textit{Avg} \\",
        r"    \midrule",
    ]

    # Human (NC) block — plain values, no deltas, excluded from the ranking.
    nc_label = (r"\multirow{" + str(len(metrics))
                + r"}{*}{\begin{tabular}[c]{@{}l@{}}Human\\(NC)\end{tabular}}")
    for i, (key, mlabel, dec, _mode) in enumerate(metrics):
        head = f"    {nc_label}" if i == 0 else "    "
        cells = [_num(column_nc(key, c), dec) for c in columns]
        out.append(f"{head} & {mlabel} & " + " & ".join(cells) + r" \\")

    for title, members in sections:
        out.append(r"    \midrule")
        if title:
            out.append(r"    \rowcolor{Gray}")
            out.append(r"    \multicolumn{" + str(n_lv + 3)
                       + r"}{l}{\textit{" + title + r"}} \\")
        for j, (mkey, mlabel_tex) in enumerate(members):
            if j > 0:
                out.append(r"    \addlinespace")
            row_label = (r"\multirow{" + str(len(metrics))
                         + r"}{*}{\begin{tabular}[c]{@{}l@{}}"
                         + mlabel_tex + r"\end{tabular}}")
            for i, (key, metric_label, dec, mode) in enumerate(metrics):
                head = f"    {row_label}" if i == 0 else "    "
                cells = []
                for col in columns:
                    vals = column_values(key, col)
                    nc = column_nc(key, col)
                    best, second = rank_best_two(vals, nc, mode)
                    v = vals.get(mkey, float("nan"))
                    cell = _num(v, dec)
                    if cell != _EM:
                        cell = _mark(cell, mkey, best, second) + _delta_sup(v, nc, dec)
                    cells.append(cell)
                out.append(f"{head} & {metric_label} & " + " & ".join(cells) + r" \\")

    out += [
        r"    \bottomrule",
        r"  \end{tabular}}",
        r"\end{table}",
    ]
    return "\n".join(out)


def rdm_per_coherence_table(per_coh: dict, human: dict, levels=None,
                            avg=None, human_avg=None) -> str:
    """``tab:rdm_percoh`` — Acc / RT / NLL / AUC / kappa per coherence.

    Parameters
    ----------
    per_coh
        ``per_coh[model][coherence]`` with keys ``Acc``, ``RT`` (median, ms),
        ``NLL``, ``AUC``, ``kappa``.
    human
        ``human[coherence]`` with the same keys — the noise ceiling.
    avg, human_avg
        Optional overrides for the *Avg* column; pass the values
        :func:`rdm_global_table` receives so the two tables agree.
    """
    levels = list(RDM_COHERENCES if levels is None else levels)
    caption = (
        r"RDM task: per-coherence model comparison. "
        r"RT: median (ms). NLL: negative log-likelihood (lower is better). "
        r"AUC: area under ROC curve grouped by coherence. "
        r"$\kappa$: Cohen\textquotesingle s kappa. "
        r"\textit{Avg}: mean over the " + str(len(levels))
        + r" coherence levels, for every row (identical to the global table). "
        r"Human~(NC): noise ceiling (AUC/$\kappa$ vs.~ground-truth labels; "
        r"NLL = LOTO). Superscript grey values give $|\Delta|$ from Human~(NC). "
        r"\textbf{Bold} = best, \underline{underline} = second best per column "
        r"(smallest $|\Delta|$, lowest value for NLL; Human~(NC) excluded)."
    )
    return _per_level_table(
        per_coh, human, levels, _RDM_METRICS, RDM_GROUPS, caption,
        "tab:rdm_percoh",
        level_header=lambda lv: f"{lv:g}",
        col_header=r"Coherence Levels (\%)",
        avg=avg, human_avg=human_avg,
    )


def sc_per_level_table(per_level: dict, human: dict, levels=None,
                       avg=None, human_avg=None) -> str:
    """``tab:sc_perlevel`` — Acc / AUC / kappa / NLL per discriminability."""
    levels = list(SC_LEVELS if levels is None else levels)
    caption = (
        r"SC task: each block shows one metric across discriminability levels "
        r"$\{$" + ", ".join(f"{lv:.1f}" for lv in levels) + r"$\}$. "
        r"NRM has DINOv2 as visual backbone. \textit{Avg}: mean over the levels, "
        r"for every row (identical to the global table). "
        r"Superscript grey values give $|\Delta|$ from Human~(NC). "
        r"\textbf{Bold} = best model per column, \underline{underline} = second "
        r"best (smallest $|\Delta|$, lowest value for NLL)."
    )
    return _per_level_table(
        per_level, human, levels, _SC_METRICS, SC_GROUPS, caption,
        "tab:sc_perlevel",
        level_header=lambda lv: f"{lv:.1f}",
        col_header="Discriminability Levels",
        avg=avg, human_avg=human_avg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Global choice/RT tables (RDM tab:rdm_global, SC tab:sc_global)
# ─────────────────────────────────────────────────────────────────────────────

# (metric key, header, decimals, ranking mode, carries a CI)
_GLOBAL_METRICS = [
    ("Acc",   r"Acc ($|\Delta|\!\downarrow$)",                    3, "delta", True),
    ("AUC",   r"AUC ($|\Delta|\!\downarrow$)",                    3, "delta", True),
    ("kappa", r"$\kappa_{\mathrm{acc}}$ ($|\Delta|\!\downarrow$)", 3, "delta", True),
    ("r_RT",  r"$r_{\rm RT}$ ($\uparrow$)",                       3, "max",   True),
    ("NLL",   r"NLL ($\downarrow$)",                              3, "min",   False),
]


def _global_table(globals_, human, groups, caption, label, cis=None,
                  resize=".85\\linewidth"):
    """Shared body of the two global tables (3 choice columns + 2 RT columns).

    ``globals_[model][metric]`` holds the point estimates, ``human[metric]`` the
    noise ceiling, and the optional ``cis[model][metric] = (lo, hi)`` plus
    ``cis[model]["NLL_sem"]`` the uncertainty annotations.
    """
    cis = cis or {}
    sections = _present(groups, OURS, set(globals_))

    out = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \renewcommand{\arraystretch}{1.4}",
        r"  \resizebox{" + resize + r"}{!}{%",
        r"  \begin{tabular}{@{} l ccc cc @{}}",
        r"    \toprule",
        r"    & \multicolumn{3}{c}{\textbf{Choice}}",
        r"    & \multicolumn{2}{c}{\textbf{RT}} \\",
        r"    \cmidrule(lr){2-4} \cmidrule(lr){5-6}",
        r"    \textbf{Model}",
    ]
    for _k, header, _d, _m, _c in _GLOBAL_METRICS:
        out.append("      & " + header)
    out[-1] += r" \\"
    out.append(r"    \midrule")

    nc_cells = [_emph(_num(human.get(k), dec))
                for k, _h, dec, _m, _c in _GLOBAL_METRICS]
    out.append(r"    \textbf{Human~(NC)}" + " & " + " & ".join(nc_cells) + r" \\")

    ranks = {}
    for key, _h, _dec, mode, _c in _GLOBAL_METRICS:
        vals = {m: globals_[m].get(key, float("nan")) for m in globals_}
        ranks[key] = rank_best_two(vals, human.get(key), mode)

    for title, members in sections:
        out.append(r"    \midrule")
        if title:
            out.append(r"    \rowcolor{Gray}")
            out.append(r"    \multicolumn{6}{l}{\textit{" + title + r"}} \\")
        for mkey, mlabel_tex in members:
            label_inline = mlabel_tex.replace(r" \\ ", "~")
            cells = []
            for key, _h, dec, _mode, has_ci in _GLOBAL_METRICS:
                v = globals_[mkey].get(key, float("nan"))
                cell = _num(v, dec)
                best, second = ranks[key]
                cell = _mark(cell, mkey, best, second)
                if cell == _EM:
                    cells.append(cell)
                    continue
                stats = cis.get(mkey, {})
                if key == "NLL":
                    cell += _sem_sup(stats.get("NLL_sem", float("nan")), dec)
                elif has_ci:
                    cell += _ci_sup(v, stats.get(key), dec)
                cells.append(cell)
            out.append(f"    {label_inline} & " + " & ".join(cells) + r" \\")

    out += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table}",
    ]
    return "\n".join(out)


def rdm_global_table(globals_: dict, human: dict, cis: dict | None = None) -> str:
    """``tab:rdm_global`` — choice and RT alignment, averaged over coherences."""
    caption = (
        r"%"
        "\n    RDM task. Choice performance is measured by Acc, AUC, and Cohen's~$\\kappa$"
        "\n    ($\\kappa_{\\mathrm{acc}}$); RT alignment by $r_{\\rm RT}$; and RT distribution"
        "\n    fit by NLL."
        "\n    \\textbf{Bold} = best; \\underline{underline} = second best (relative to NC,"
        "\n    or absolute for NLL and $r_{\\rm RT}$)."
        "\n    Asymmetric 95\\% CIs (bootstrap, $B{=}1000$) shown as superscripts/subscripts;"
        "\n    NLL reports $\\pm$ Std. Error of the Mean.\n  "
    )
    return _global_table(globals_, human, RDM_GROUPS, caption, "tab:rdm_global", cis)


def sc_global_table(globals_: dict, human: dict, cis: dict | None = None) -> str:
    """``tab:sc_global`` — choice and RT alignment, averaged over levels."""
    caption = (
        r"Scene Categorisation. Choice performance is measured by Acc, AUC, and "
        r"Cohen's~$\kappa$ ($\kappa_{\mathrm{acc}}$); RT alignment by "
        r"$r_{\rm RT}$ (correlation of per-discriminability median RT); and RT "
        r"distribution fit by NLL. \textbf{Bold} = best; \underline{underline} = "
        r"second best (relative to NC, or absolute for $r_{\rm RT}$ and NLL). "
        r"Asymmetric 95\% CIs (bootstrap, $B{=}1000$) shown as "
        r"superscripts/subscripts; NLL reports $\pm$ Std. Error of the Mean."
    )
    return _global_table(globals_, human, SC_GROUPS, caption, "tab:sc_global", cis)


# ─────────────────────────────────────────────────────────────────────────────
# SAT tables
# ─────────────────────────────────────────────────────────────────────────────

# (index into the (rmse, r, kappa) triple, row label, decimals, ranking mode,
#  whether the cell carries a grey |Delta| superscript)
_SAT_METRICS = [
    (0, r"RMSE $\downarrow$",  2, "delta", True),
    (1, r"$r$ $\uparrow$",     3, "max",   False),
    (2, r"$\mathcal{K}$",      3, "delta", True),
]


def _sat_columns(tasks, task_levels):
    """``[('lv', task, level) | ('avg', task)]`` in table order."""
    cols = []
    for t in tasks:
        cols += [("lv", t, lv) for lv in task_levels[t]]
        cols.append(("avg", t, None))
    return cols


def sat_per_level_table(model_vals: dict, human_vals: dict,
                        tasks=None, task_levels=None) -> str:
    """``tab:sat_perobs`` — RMSE / r / kappa per perturbation level.

    Parameters
    ----------
    model_vals
        ``model_vals[(model, task, level)] = (rmse, spearman_r, steepness)``.
    human_vals
        ``human_vals[(task, level)] = (rmse, spearman_r, steepness)`` — the
        per-observer noise ceiling.
    """
    tasks = list(SAT_TABLE_TASKS if tasks is None else tasks)
    task_levels = SAT_TASK_LEVELS if task_levels is None else task_levels
    columns = _sat_columns(tasks, task_levels)
    models = [(k, lbl) for k, lbl in SAT_MODELS + [OURS]
              if any(k == m for m, _t, _l in model_vals)]

    def mv(model, col, i):
        kind, t, lv = col
        if kind == "lv":
            return model_vals.get((model, t, lv), (np.nan,) * 3)[i]
        return _safe_mean([model_vals.get((model, t, x), (np.nan,) * 3)[i]
                           for x in task_levels[t]])

    def hv(col, i):
        kind, t, lv = col
        if kind == "lv":
            return human_vals.get((t, lv), (np.nan,) * 3)[i]
        return _safe_mean([human_vals.get((t, x), (np.nan,) * 3)[i]
                           for x in task_levels[t]])

    spans = {t: len(task_levels[t]) + 1 for t in tasks}
    col_spec = "@{} ll " + " ".join("l" * spans[t] for t in tasks) + " @{}"

    caption = (
        r"Speed--accuracy tradeoff (SAT) on the SATBench benchmark "
        r"\citep{subramanian2022satbench}, evaluated across input types and "
        r"perturbation levels. Human~(NC): median metric per-observer per-level "
        r"(noise ceiling). Model rows report the raw metric value; superscript "
        r"grey values show $|\Delta|$ from Human~(NC) for RMSE and "
        r"$\mathcal{K}$. Column $r$ is intrinsically human-relative and carries "
        r"no superscript delta. \textit{Avg} = mean over perturbation levels. "
        r"\textbf{Bold} = best, \underline{underline} = second best per column "
        r"(ranked by smallest $|\Delta|$ for RMSE and $\mathcal{K}$; largest "
        r"value for $r$)."
    )

    out = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{tab:sat_perobs}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2.2pt}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{" + col_spec + "}",
        r"    \toprule",
    ]

    header = "    & &"
    cmid, first = [], 3
    for t in tasks:
        header += f" \\multicolumn{{{spans[t]}}}{{c}}{{{_SAT_TASK_HEADERS[t]}}} &"
        cmid.append(f"\\cmidrule(lr){{{first}-{first + spans[t] - 1}}}")
        first += spans[t]
    out.append(header.rstrip(" &") + r" \\")
    out.append("    " + " ".join(cmid))

    sub = r"    \textbf{Model} & \textbf{Metric}"
    for t in tasks:
        sub += "".join(" & " + _SAT_LEVEL_HEADERS[t][lv] for lv in task_levels[t])
        sub += r" & \textit{Avg}"
    out.append(sub + r" \\")
    out.append(r"    \midrule")

    nc_label = (r"\multirow{3}{*}{\begin{tabular}[c]{@{}l@{}}Human \\ (NC)"
                r"\end{tabular}}")
    for i, (idx, mlabel, dec, _mode, _delta) in enumerate(_SAT_METRICS):
        head = f"    {nc_label}" if i == 0 else "    "
        out.append(f"{head} & {mlabel} & "
                   + " & ".join(_num(hv(c, idx), dec) for c in columns) + r" \\")

    ranks = {}
    for idx, _lbl, _dec, mode, _delta in _SAT_METRICS:
        for col in columns:
            vals = {m: mv(m, col, idx) for m, _l in models}
            ranks[(idx, col)] = rank_best_two(vals, hv(col, idx), mode)

    for mkey, mlabel_tex in models:
        out.append(r"    \midrule")
        row_label = (r"\multirow{3}{*}{\begin{tabular}[c]{@{}l@{}}"
                     + mlabel_tex + r"\end{tabular}}")
        for i, (idx, metric_label, dec, _mode, with_delta) in enumerate(_SAT_METRICS):
            head = f"    {row_label}" if i == 0 else "    "
            cells = []
            for col in columns:
                v = mv(mkey, col, idx)
                cell = _num(v, dec)
                if cell != _EM:
                    best, second = ranks[(idx, col)]
                    cell = _mark(cell, mkey, best, second)
                    if with_delta:
                        cell += _delta_sup(v, hv(col, idx), dec)
                cells.append(cell)
            out.append(f"{head} & {metric_label} & " + " & ".join(cells) + r" \\")

    out += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table}",
    ]
    return "\n".join(out)


def sat_global_table(model_vals: dict, human_vals: dict, cis: dict | None = None,
                     tasks=None, task_levels=None) -> str:
    """``tab:sat_global`` — RMSE / r / kappa averaged over perturbation levels.

    Point estimates are the per-level means (the ``Avg`` columns of
    ``tab:sat_perobs``); the CIs come from resampling observers, so they are
    asymmetric about the point estimate by construction.

    ``cis[(model, task)]`` may hold ``"RMSE"`` and ``"r"`` ``(lo, hi)`` pairs.
    """
    tasks = list(tasks if tasks is not None else ["color-blur", "color-gray", "gray-noise"])
    task_levels = SAT_TASK_LEVELS if task_levels is None else task_levels
    cis = cis or {}
    models = [(k, lbl) for k, lbl in SAT_MODELS + [OURS]
              if any(k == m for m, _t, _l in model_vals)]

    def avg(model, task, i):
        return _safe_mean([model_vals.get((model, task, lv), (np.nan,) * 3)[i]
                           for lv in task_levels[task]])

    def havg(task, i):
        return _safe_mean([human_vals.get((task, lv), (np.nan,) * 3)[i]
                           for lv in task_levels[task]])

    # The caption as printed in the paper. The highlighting is computed:
    # RMSE and $\mathcal{K}$ are ranked by smallest $|\Delta|$ from Human (NC),
    # $r$ by largest value, with the ceiling row excluded from the ranking.
    caption = (
        r"%"
        "\n        SATBench: Global comparison averaged across all perturbation levels. "
        "\n        RMSE, Spearman's correlation ($r$) and curve steepness "
        r"($\mathcal{K}$) assess SAT alignment. \textbf{Bold} and "
        r"\underline{underline} indicate the best and second-best performing "
        r"models, respectively (for $\mathcal{K}$, values are relative to NC). "
        "\n        Asymmetric 95\\% confidence intervals, derived via bootstrapping "
        r"($B=1000$), are reported as superscripts/subscripts."
        "\n    "
    )

    out = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{tab:sat_global}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \renewcommand{\arraystretch}{1.4}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{@{} l " + " ".join(["ccc"] * len(tasks)) + r" @{}}",
        r"    \toprule",
    ]
    out.append("    & " + "\n    & ".join(
        r"\multicolumn{3}{c}{" + _SAT_TASK_PLAIN[t] + "}" for t in tasks) + r" \\")
    out.append("    " + " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                                 for i in range(len(tasks))))
    group = r"RMSE ($\downarrow$) & $r$ ($\uparrow$) & $\mathcal{K}$ ($|\Delta|\!\downarrow$)"
    out.append(r"    \textbf{Model}" + "".join("\n      & " + group for _ in tasks) + r" \\")
    out.append(r"    \midrule")

    nc_cells = []
    for t in tasks:
        nc_cells += [_emph(_num(havg(t, i), dec))
                     for i, dec in ((0, 2), (1, 3), (2, 3))]
    out.append(r"    \textbf{Human~(NC)}" + "\n      & " + "\n      & ".join(nc_cells) + r" \\")
    out.append(r"    \midrule")

    ranks = {}
    for t in tasks:
        for idx, _lbl, _dec, mode, _d in _SAT_METRICS:
            vals = {m: avg(m, t, idx) for m, _l in models}
            ranks[(t, idx)] = rank_best_two(vals, havg(t, idx), mode)

    for i, (mkey, mlabel_tex) in enumerate(models):
        # Rule off "ours" from the comparators — unless it is the only row,
        # which would double the rule under the header.
        if mkey == OURS[0] and i > 0:
            out.append(r"    \midrule")
        cells = []
        for t in tasks:
            stats = cis.get((mkey, t), {})
            for idx, dec, ci_key, strip in ((0, 2, "RMSE", False),
                                            (1, 3, "r", True),
                                            (2, 3, None, True)):
                v = avg(mkey, t, idx)
                cell = _num(v, dec)
                best, second = ranks[(t, idx)]
                cell = _mark(cell, mkey, best, second)
                if cell != _EM and ci_key:
                    cell += _ci_sup(v, stats.get(ci_key), dec, strip_zero=strip)
                cells.append(cell)
        label_inline = mlabel_tex.replace(r" \\ ", "~")
        out.append(f"    {label_inline}" + "\n      & " + "\n      & ".join(cells) + r" \\")

    out += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table}",
    ]
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# RDM mechanism table
# ─────────────────────────────────────────────────────────────────────────────

#: Columns of the mechanism table, in order: dataframe column -> LaTeX header.
MECHANISM_COLUMNS = [
    ("drift_mag",      r"drift $\lVert f\rVert$"),
    ("drift_contrast", r"contrast $f^{+}\!-\!f^{-}$"),
    ("diffusion_mag",  r"diffusion $\lVert g\rVert$"),
    ("threshold",      r"threshold $\theta$"),
    ("snr_contrast",   r"SNR contrast"),
    ("snr_mag",        r"SNR mag."),
]


def rdm_mechanism_table(per_trial, coherence_col: str = "coherence_pct") -> str:
    """Per-coherence means of the learned drift/diffusion/threshold fields.

    ``per_trial`` is the per-trial frame returned by
    ``scripts.difficulty_parameter_correlation.extract_parameters``. The model never sees the
    coherence label, so grouping by it is what makes the table informative.
    """
    cols = [c for c, _h in MECHANISM_COLUMNS if c in per_trial.columns]
    means = per_trial.groupby(coherence_col)[cols].mean()

    out = [
        r"\begin{center}\small",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        r"Coh.\ (\%) & "
        + " & ".join(h for c, h in MECHANISM_COLUMNS if c in cols) + r" \\",
        r"\midrule",
    ]
    # drift/diffusion/threshold sit near 1, 0.4 and 0.5: 3 decimals everywhere
    # except diffusion, whose spread across coherences is in the 4th.
    decs = {"diffusion_mag": 4}
    for coh, row in means.iterrows():
        cells = [format(float(row[c]), f".{decs.get(c, 3)}f") for c in cols]
        out.append(f"{float(coh):>5g}  & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(out)
