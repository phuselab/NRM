"""Canonical result layout shared by the pipeline notebooks.

Every notebook writes its metrics through this module and ``ranking.ipynb``
reads them back, so the aggregate ranking is always computed from results that
were actually produced on this machine.

Layout under ``outputs/``::

    rdm/rdm_global_results.json             written by pipelines/rdm.ipynb
    sc/sc_global_results.json               written by pipelines/sc.ipynb
    sat/<task>/nrm_metrics.json             written by pipelines/sat.ipynb
    sat/<task>/<model>_metrics.json         written by competitors/*/train_sat.py
    sat/<task>/<model>/                     per-model SAT detail, written by the
        sat_level_curves.json               same producer: the human and model
        satbench_metrics_full.json          SAT curves and the per-observer
                                            eRMSE / Spearman / steepness arrays
    ranking/                                written by pipelines/ranking.ipynb

The RDM/SC files map ``model -> {"global": {acc, auc, kappa, r_RT, nll_mean}}``.
The SAT files map ``{"satbench_metrics": {ermse, spearman_r, steepness}}`` —
``run_sat_level_analysis`` already returns that shape, so a full result dict can
be handed over unchanged.

Refreshed runs
--------------
A run that recomputes predictions (``REFRESH``) or uses retrained weights
(``REFRESH_WEIGHTS``) never overwrites the reference outputs: every file it
writes carries the ``_REFRESH`` suffix (``rdm_global_results_REFRESH.json``,
``nrm_metrics_REFRESH.json``, ``sat/<task>/<model>_REFRESH/`` ...). Every
writer and reader below takes ``refresh=``; ``ranking.ipynb`` reads the
reference outputs only. Retrained weights follow the same rule
(:func:`refresh_path`, :func:`refresh_weights_available`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = REPO_ROOT / "outputs"

RANKING_DIR = OUTPUTS / "ranking"

METRIC_COLS = ["Acc", "AUC", "kappa", "r_RT", "NLL"]
SAT_TASKS = ["color-blur", "color-gray", "gray-noise"]
SAT_PREFIXES = {"color-blur": "CB", "color-gray": "CG", "gray-noise": "GN"}
SAT_COLS = [f"{p}_{m}" for p in ("CB", "CG", "GN") for m in ("RMSE", "r", "K")]

#: ``outputs/sat/<task>/<key>_metrics.json`` -> table label
SAT_MODEL_KEYS = {
    "nrm": "NRM",
    "cnet_par": "CNet(par.)",
    "cnet_ser": "CNet(ser.)",
    "convrnn": "ConvRNN",
    "msdnet": "MSDNet",
    "scan": "SCAN",
}

#: Table label -> ``outputs/sat/<task>/<key>/`` directory key.
SAT_LABEL_TO_KEY = {label: key for key, label in SAT_MODEL_KEYS.items()}

#: Spearman is undefined when a model's accuracy is constant within a category
#: (MSDNet, systematically). The tables report 0.001 rather than a hole;
#: :func:`satbench_scalars` and :func:`fill_undefined_spearman` both substitute
#: it, so the canonical scalars and the tables cannot disagree.
SPEARMAN_UNDEFINED_FILL = 0.001

#: Reserved row name for the human noise ceiling in the RDM/SC result files.
#: It is the reference the models are scored against, so ``load_global_results``
#: leaves it out of the model table.
NC_KEY = "Human (NC)"

#: Which notebook/script produces each output, for error messages.
PRODUCER = {
    "rdm": "pipelines/rdm.ipynb",
    "sc": "pipelines/sc.ipynb",
    "sat": "pipelines/sat.ipynb (NRM) and competitors/<model>/train_sat.py",
}


class MissingResultsError(FileNotFoundError):
    """Raised when a ranking input has not been produced yet."""


# ── writing ──────────────────────────────────────────────────────────────────

def _serialisable(obj):
    """JSON-safe copy: numpy/torch scalars out, non-finite floats -> None."""
    if isinstance(obj, dict):
        return {str(k): _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _serialisable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if hasattr(obj, "detach"):  # torch.Tensor without importing torch
        return _serialisable(obj.detach().cpu().numpy())
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    return str(obj)


#: Public alias — notebooks dump their own side outputs with this.
serialisable = _serialisable


def display_path(path) -> str:
    """``path`` relative to the repository root when it lies inside it."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return str(resolved.relative_to(REPO_ROOT))
    return str(path)


#: Suffix carried by every output of a refreshed run and by retrained weights.
REFRESH_SUFFIX = "_REFRESH"


def refresh_tag(refresh: bool) -> str:
    """``"_REFRESH"`` for a refreshed run, ``""`` for the reference one."""
    return REFRESH_SUFFIX if refresh else ""


def refresh_path(path, refresh: bool) -> Path:
    """``path`` with ``_REFRESH`` inserted before its extension when ``refresh``.

    ``nrm/weights/NRM_rdm.pth`` -> ``nrm/weights/NRM_rdm_REFRESH.pth``;
    a directory gains the suffix at its end.
    """
    path = Path(path)
    if not refresh:
        return path
    return path.with_name(f"{path.stem}{REFRESH_SUFFIX}{path.suffix}")


def require_refresh_weights(paths) -> list[Path]:
    """The ``_REFRESH`` counterparts of ``paths``; all of them must exist.

    Raises ``FileNotFoundError`` naming every missing file, so a run asked to
    use retrained weights never mixes them with the released ones.
    """
    resolved = [refresh_path(p, True) for p in paths]
    missing = [p for p in resolved if not p.exists()]
    if missing:
        listing = "\n  ".join(str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT)
                              else str(p) for p in missing)
        raise FileNotFoundError(
            f"REFRESH_WEIGHTS is set but {len(missing)} retrained weight file(s) "
            f"are missing:\n  {listing}\nTrain them first (scripts/train_nrm_*.py, "
            f"competitors/*/train_*.py), or set REFRESH_WEIGHTS = False."
        )
    return resolved


def refresh_weights_available(models: dict) -> dict[str, bool]:
    """Which models have all of their retrained (``_REFRESH``) weight files.

    ``models`` maps a model name to its ``_REFRESH`` paths. A model missing any
    of them is reported and left out of the run; the released weights never
    stand in for it. Raises ``FileNotFoundError`` when no model has its
    retrained weights, as there is then nothing to evaluate.
    """
    missing = {name: [p for p in map(Path, paths) if not p.exists()]
               for name, paths in models.items()}
    if all(missing.values()):
        listing = "\n  ".join(display_path(p) for ps in missing.values() for p in ps)
        raise FileNotFoundError(
            f"REFRESH_WEIGHTS is set but no model has its retrained weights:\n  "
            f"{listing}\nTrain them first (scripts/train_nrm_*.py, "
            f"competitors/*/train_*.py), or set REFRESH_WEIGHTS = False."
        )
    for name, ps in missing.items():
        if ps:
            print(f"REFRESH_WEIGHTS: skipping {name} — missing "
                  + ", ".join(display_path(p) for p in ps))
    return {name: not ps for name, ps in missing.items()}


def benchmark_dir(benchmark: str) -> Path:
    return OUTPUTS / benchmark


def _level_mean(by_level, fill: float | None = None) -> float:
    """Mean over perturbation levels.

    ``fill`` substitutes a value for levels where the statistic is undefined
    instead of dropping them, so the Spearman r is always averaged over every
    level: ConvRNN on color-gray is undefined at *Color* and 0.166 at *Gray*,
    and its scalar is (0.001 + 0.166) / 2 = 0.084.
    """
    vals = []
    for v in (by_level or {}).values():
        if v is not None and np.isfinite(float(v)):
            vals.append(float(v))
        elif fill is not None:
            vals.append(float(fill))
    return float(np.mean(vals)) if vals else float("nan")


def satbench_scalars(result: dict) -> dict:
    """Reduce a SAT analysis result to the three scalars the tables report.

    Accepts any of:

    * the full ``run_sat_level_analysis`` return (has a ``"metrics"`` key),
    * the ``compute_satbench_metrics`` return,
    * an already-reduced mapping.

    **eRMSE and the Spearman r are per-observer statistics.** The tables
    report the mean over levels of the *median across observers*
    (``ermse.model_by_level_median`` / ``spearman.model_by_level_median``),
    which is what ``tab:sat_perobs`` and ``tab:sat_global`` show. That is a
    different number from ``mean_over_levels``, which compares the model curve
    to the *mean* human curve and the *mean* (not median) per-observer
    Spearman, so the per-observer medians are used and ``mean_over_levels`` is
    only the fallback for a payload without them.

    Steepness has no per-observer form; it is the Weibull fit to the aggregate
    curve, identical in both. ``steepness_human`` is carried through because it
    is the human noise ceiling for kappa.
    """
    metrics = result.get("metrics", result)

    if {"ermse", "spearman_r", "steepness"} & set(metrics) and \
            not isinstance(metrics.get("ermse"), dict):
        return {k: metrics.get(k) for k in
                ("ermse", "spearman_r", "steepness", "steepness_human")}

    mol = metrics.get("mean_over_levels")
    ermse_block = metrics.get("ermse")
    spearman_block = metrics.get("spearman")
    if mol is None and not isinstance(ermse_block, dict):
        raise KeyError(
            "cannot find 'mean_over_levels' or 'ermse' — pass the result of "
            "compute_satbench_metrics() or run_sat_level_analysis()"
        )
    mol = mol or {}

    ermse = float("nan")
    if isinstance(ermse_block, dict):
        ermse = _level_mean(ermse_block.get("model_by_level_median"))
    if not np.isfinite(ermse):
        ermse = mol.get("rmse")

    # Same substitution the tables make via `fill_undefined_spearman`, applied
    # **per level** before averaging: when a model's accuracy is constant
    # within every category the rank correlation is undefined, and the tables
    # report 0.001 rather than a hole. MSDNet does this at every level of
    # color-blur and gray-noise, ConvRNN at color-gray's *Color* only.
    spearman = float("nan")
    if isinstance(spearman_block, dict):
        spearman = _level_mean(spearman_block.get("model_by_level_median"),
                               fill=SPEARMAN_UNDEFINED_FILL)
    if not np.isfinite(spearman):
        spearman = mol.get("spearman_r")
    if spearman is None or not np.isfinite(float(spearman)):
        spearman = SPEARMAN_UNDEFINED_FILL

    return {
        "ermse": ermse,
        "spearman_r": spearman,
        "steepness": mol.get("steepness_model"),
        "steepness_human": mol.get("steepness_human"),
    }


def global_results_path(benchmark: str, refresh: bool = False) -> Path:
    return benchmark_dir(benchmark) / f"{benchmark}_global_results{refresh_tag(refresh)}.json"


def save_global_results(benchmark: str, results: dict, refresh: bool = False) -> Path:
    """Write the RDM or SC per-model result dict to its canonical path."""
    path = global_results_path(benchmark, refresh)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_serialisable(results), f, indent=2)
    return path


def save_sat_metrics(task: str, model_key: str, metrics: dict,
                     refresh: bool = False) -> Path:
    """Write one model's SAT metrics for one perturbation.

    ``metrics`` may be the full ``run_sat_level_analysis`` result, the
    ``compute_satbench_metrics`` result, or an already-reduced mapping — see
    :func:`satbench_scalars`.
    """
    if task not in SAT_TASKS:
        raise ValueError(f"unknown SAT task {task!r}, expected one of {SAT_TASKS}")
    if model_key not in SAT_MODEL_KEYS:
        raise ValueError(
            f"unknown model key {model_key!r}, expected one of {sorted(SAT_MODEL_KEYS)}"
        )
    out = benchmark_dir("sat") / task
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{model_key}_metrics{refresh_tag(refresh)}.json"
    payload = {"satbench_metrics": satbench_scalars(metrics)}
    with open(path, "w") as f:
        json.dump(_serialisable(payload), f, indent=2)
    return path


# ── reading ──────────────────────────────────────────────────────────────────

def _read_global_rows(benchmark: str, refresh: bool = False) -> dict:
    path = global_results_path(benchmark, refresh)
    if not path.exists():
        raise MissingResultsError(
            f"{path.relative_to(REPO_ROOT)} not found.\n"
            f"Run {PRODUCER[benchmark]} first — it writes this file."
        )
    with open(path) as f:
        raw = json.load(f)

    rows = {}
    for model, data in raw.items():
        g = data.get("global", data)
        rows[model] = {
            "Acc": g.get("acc"),
            "AUC": g.get("auc"),
            "kappa": g.get("kappa"),
            "r_RT": g.get("r_RT"),
            "NLL": g.get("nll_mean"),
        }
    return rows


def load_global_results(benchmark: str, refresh: bool = False) -> pd.DataFrame:
    """Load RDM or SC results as a **model** x metric table.

    The human noise ceiling travels in the same file under :data:`NC_KEY` but
    is excluded here: it is the reference the models are scored against, not a
    competitor. Read it with :func:`load_computed_nc`.

    Raises
    ------
    MissingResultsError
        If the upstream notebook has not been run.
    """
    rows = {k: v for k, v in _read_global_rows(benchmark, refresh).items() if k != NC_KEY}
    if not rows:
        path = global_results_path(benchmark, refresh)
        raise MissingResultsError(
            f"{path.relative_to(REPO_ROOT)} contains no models. "
            f"Re-run {PRODUCER[benchmark]}."
        )
    return pd.DataFrame(rows).T[METRIC_COLS].astype(float)


def load_computed_nc(benchmark: str, refresh: bool = False) -> dict | None:
    """The human noise ceiling as computed by the benchmark notebook, or None.

    For RDM and SC it is the :data:`NC_KEY` entry of the global results file;
    for SAT it is ``outputs/sat/human_nc.json`` (see
    :func:`save_sat_noise_ceiling`).
    """
    if benchmark == "sat":
        path = benchmark_dir("sat") / f"human_nc{refresh_tag(refresh)}.json"
        if not path.exists():
            return None
        with open(path) as f:
            nc = json.load(f)
        return nc if any(v is not None for v in nc.values()) else None

    try:
        rows = _read_global_rows(benchmark, refresh)
    except MissingResultsError:
        return None
    nc = rows.get(NC_KEY)
    if nc is None or all(v is None for v in nc.values()):
        return None
    return nc


def sat_noise_ceiling_scalars(results_by_task: dict) -> dict:
    """The SAT human noise ceiling, from the per-task analysis results.

    The mirror of :func:`satbench_scalars` for the human side, and reduced the
    same way: the mean over levels of the median across observers for eRMSE and
    the Spearman r, and the Weibull steepness of the aggregate human curve.

    ``results_by_task`` maps each SAT task to either the
    ``run_sat_level_analysis`` return or a parsed
    ``satbench_metrics_full.json``; both carry the blocks used here.

    """
    nc: dict = {}
    for task, prefix in SAT_PREFIXES.items():
        result = results_by_task.get(task)
        if result is None:
            continue
        metrics = result.get("metrics", result)
        ermse = (metrics.get("ermse") or {}).get("human_by_level_median")
        spearman = (metrics.get("spearman") or {}).get("human_by_level")
        mol = metrics.get("mean_over_levels") or {}
        nc[f"{prefix}_RMSE"] = _level_mean(ermse) if ermse else None
        nc[f"{prefix}_r"] = _level_mean(spearman) if spearman else None
        nc[f"{prefix}_K"] = mol.get("steepness_human")
    return nc


def save_sat_noise_ceiling(nc: dict, refresh: bool = False) -> Path:
    """Write the SAT human noise ceiling, keyed like :data:`SAT_COLS`."""
    out = benchmark_dir("sat")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"human_nc{refresh_tag(refresh)}.json"
    with open(path, "w") as f:
        json.dump(_serialisable({c: nc.get(c) for c in SAT_COLS}), f, indent=2)
    return path


def load_sat_results(require_all: bool = False, refresh: bool = False) -> pd.DataFrame:
    """Load the SAT table (models x 9 columns) from per-task JSONs.

    Parameters
    ----------
    require_all
        When True, every model in :data:`SAT_MODEL_KEYS` must be present for
        every task. When False (default) whatever has been produced is
        returned, and partially-evaluated models carry NaN columns.
    """
    sat_dir = benchmark_dir("sat")
    rows: dict[str, dict[str, float]] = {}
    missing: list[str] = []

    for task in SAT_TASKS:
        prefix = SAT_PREFIXES[task]
        for key, label in SAT_MODEL_KEYS.items():
            path = sat_dir / task / f"{key}_metrics{refresh_tag(refresh)}.json"
            if not path.exists():
                missing.append(str(path.relative_to(REPO_ROOT)))
                continue
            with open(path) as f:
                payload = json.load(f)
            sm = payload.get("satbench_metrics", payload)
            row = rows.setdefault(label, {})
            row[f"{prefix}_RMSE"] = sm.get("ermse")
            row[f"{prefix}_r"] = sm.get("spearman_r")
            row[f"{prefix}_K"] = sm.get("steepness")

    if not rows:
        raise MissingResultsError(
            "No SAT metrics found under "
            f"{sat_dir.relative_to(REPO_ROOT)}.\nRun {PRODUCER['sat']} first."
        )
    if missing and require_all:
        listing = "\n  ".join(missing)
        raise MissingResultsError(
            f"{len(missing)} SAT metric file(s) missing:\n  {listing}\n"
            f"Run {PRODUCER['sat']}, or pass require_all=False to rank what exists."
        )
    if missing:
        print(f"[results_io] {len(missing)} SAT metric file(s) not yet produced; "
              f"those cells will be NaN. First missing: {missing[0]}")

    df = pd.DataFrame(rows).T
    for c in SAT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[SAT_COLS].astype(float)


def sat_model_dir(task: str, model_key: str, refresh: bool = False) -> Path:
    """``outputs/sat/<task>/<model_key>[_REFRESH]/`` — one model's SAT detail."""
    if task not in SAT_TASKS:
        raise ValueError(f"unknown SAT task {task!r}, expected one of {SAT_TASKS}")
    if model_key not in SAT_MODEL_KEYS:
        raise ValueError(
            f"unknown model key {model_key!r}, expected one of {sorted(SAT_MODEL_KEYS)}"
        )
    return benchmark_dir("sat") / task / f"{model_key}{refresh_tag(refresh)}"


def load_sat_detail(refresh: bool = False) -> tuple[dict, dict]:
    """Collect every produced SAT per-model detail file.

    Returns ``(full, curves)``, both keyed by ``(table_label, task)``:
    ``full`` holds the ``satbench_metrics_full.json`` payloads (per-observer
    eRMSE / Spearman / steepness) and ``curves`` the ``sat_level_curves.json``
    payloads (the human and model SAT curves). Pairs that have not been
    evaluated are simply absent, so a partial run still yields a figure and a
    table over whatever exists.
    """
    full: dict[tuple[str, str], dict] = {}
    curves: dict[tuple[str, str], dict] = {}
    for task in SAT_TASKS:
        for key, label in SAT_MODEL_KEYS.items():
            d = sat_model_dir(task, key, refresh)
            for name, store in (("satbench_metrics_full.json", full),
                                ("sat_level_curves.json", curves)):
                path = d / name
                if path.exists():
                    with open(path) as f:
                        store[(label, task)] = json.load(f)
    if not full and not curves:
        raise MissingResultsError(
            "No per-model SAT detail found under "
            f"{benchmark_dir('sat').relative_to(REPO_ROOT)}/<task>/<model>/.\n"
            f"Run {PRODUCER['sat']} first."
        )
    return full, curves


def fill_undefined_spearman(full: dict, fill: float = SPEARMAN_UNDEFINED_FILL):
    """In-place: replace missing Spearman values in ``load_sat_detail`` output."""
    def fix(v):
        try:
            return fill if v is None or not np.isfinite(float(v)) else float(v)
        except (TypeError, ValueError):
            return fill

    for payload in full.values():
        sp = payload.get("spearman", {})
        # The per-observer arrays drive the bootstrap CI, and the producer
        # drops observers whose correlation is undefined rather than storing a
        # hole — so a model undefined for *every* observer arrives as an empty
        # list, not a list of Nones. `compute_per_obs_bootstrap_ci` needs at
        # least three values, so the fill is sized from the human array, which
        # always carries one entry per observer: "undefined for all N
        # observers" resamples as N and the cell keeps its (degenerate)
        # interval.
        n_obs = len(sp["human_per_obs"]) if isinstance(sp.get("human_per_obs"), list) else 0
        for field in ("model_per_obs", "human_per_obs"):
            if isinstance(sp.get(field), list):
                fixed = [fix(x) for x in sp[field]]
                sp[field] = fixed or [fill] * max(n_obs, 1)
        for field in ("model_by_level_median", "human_by_level", "by_level"):
            if isinstance(sp.get(field), dict):
                sp[field] = {k: fix(v) for k, v in sp[field].items()}
        for field in ("spearman_cat_median", "spearman_cat_median_human"):
            if isinstance(payload.get(field), dict):
                payload[field] = {k: fix(v) for k, v in payload[field].items()}
    return full
