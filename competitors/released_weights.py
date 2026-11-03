"""Resolve a SAT competitor checkpoint: the released one or its retrained twin.

The release ships the reference checkpoints under ``competitors/weights/sat/``
as ``<model>_<perturbation>.pt``. Retraining a competitor never overwrites
them: ``competitors/<model>/train_sat.py`` saves its final checkpoint next to
the released one as ``<model>_<perturbation>_REFRESH.pt``.

``resolve_sat_checkpoint`` picks, in order:

1. an explicit ``--weights`` path,
2. the retrained ``_REFRESH`` checkpoint, when asked for (``refresh_weights``),
3. the released checkpoint,

and fails with the path it looked for rather than falling through to another.

The two on-disk layouts are kept as they are. CNet checkpoints are
``{"model": state_dict, "optimizer": ...}``, read through its own
``models/model_utils.load_model``; the other three are
``{"epoch", "model_state", "optim_state"}``, read with
``load_state_dict(ckpt["model_state"])``.
"""

from __future__ import annotations

import os

from scripts.results_io import display_path

#: ``competitors/weights/sat/`` relative to this file.
SAT_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "weights", "sat")

#: Suffix of a retrained checkpoint (same as ``scripts.results_io.REFRESH_SUFFIX``).
REFRESH_SUFFIX = "_REFRESH"

#: Perturbation name in the behavioural files -> the spelling used in weight
#: file names (``color-blur`` -> ``color_blur``).
PERTURB_SLUG = {
    "color-blur": "color_blur",
    "color-gray": "color_gray",
    "gray-noise": "gray_noise",
}


def released_sat_path(model: str, perturb: str, refresh: bool = False) -> str:
    """``competitors/weights/sat/<model>_<perturbation>[_REFRESH].pt``.

    ``model`` is the weight-file stem: ``cnet_parallel``, ``cnet_serial``,
    ``convrnn``, ``msdnet`` or ``scan``.
    """
    if perturb not in PERTURB_SLUG:
        raise ValueError(f"unknown perturbation {perturb!r}, "
                         f"expected one of {sorted(PERTURB_SLUG)}")
    suffix = REFRESH_SUFFIX if refresh else ""
    return os.path.join(SAT_WEIGHTS_DIR, f"{model}_{PERTURB_SLUG[perturb]}{suffix}.pt")


def resolve_sat_checkpoint(model: str, perturb: str,
                           explicit: str | None = None,
                           refresh_weights: bool = False) -> str:
    """Pick the checkpoint to evaluate.

    Parameters
    ----------
    model
        Weight-file stem, e.g. ``"cnet_parallel"`` or ``"scan"``.
    perturb
        ``"color-blur"``, ``"color-gray"`` or ``"gray-noise"``.
    explicit
        A path given on the command line; used as-is, and missing is an error.
    refresh_weights
        Use the retrained ``_REFRESH`` checkpoint instead of the released one.

    Returns
    -------
    str
        An existing path.
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"--weights {explicit} does not exist.")
        print(f"Checkpoint (--weights): {explicit}")
        return explicit

    path = released_sat_path(model, perturb, refresh=refresh_weights)
    if os.path.isfile(path):
        kind = "retrained" if refresh_weights else "released"
        print(f"Checkpoint ({kind} weights): {display_path(path)}")
        return path

    if refresh_weights:
        raise FileNotFoundError(
            f"No retrained checkpoint for {model} / {perturb} at {path}.\n"
            f"Train it first (run this script without --eval_only), or drop "
            f"--refresh_weights to evaluate the released weights."
        )
    raise FileNotFoundError(
        f"No checkpoint for {model} / {perturb} at {path}.\n"
        f"The competitor weights are not distributed: run without --eval_only "
        f"to train (and evaluate) the model, or pass --weights <path>."
    )


def keep_final_checkpoint(ckpt_dir: str, model: str, perturb: str,
                          pattern: str = "ckpt__*.pt") -> str:
    """Copy the newest checkpoint of a training run to its ``_REFRESH`` path.

    The training loops save periodic checkpoints into a scratch ``ckpt_dir``;
    only the final one is kept, as
    ``competitors/weights/sat/<model>_<perturbation>_REFRESH.pt``. The
    released checkpoint is never touched.
    """
    import glob
    import shutil

    local = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    if not local:
        raise FileNotFoundError(f"training wrote no checkpoint matching "
                                f"{os.path.join(ckpt_dir, pattern)}")
    dest = released_sat_path(model, perturb, refresh=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(local[-1], dest)
    print(f"Final checkpoint saved: {display_path(dest)}")
    return dest
