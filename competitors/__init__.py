"""Baseline models compared against NRM.

Classical accumulator models are importable directly from this package::

    from competitors import prepare_data, fit_lnr, simulate_lnr

Three are fitted by maximum likelihood at evaluation time, so they ship as code
rather than weights:

====  ==========================  ==========================
Name  Model                       Module
====  ==========================  ==========================
LNR   Log-Normal Race             :mod:`competitors.lnr`
LBA   Linear Ballistic Accumulator :mod:`competitors.lba`
DDM   Racing Diffusion Model      :mod:`competitors.ddm`
====  ==========================  ==========================

DDM is the *racing* diffusion model, so its functions are named ``fit_rdm`` /
``simulate_rdm`` after the model class — not after the RDM benchmark. Aliases
``fit_ddm`` / ``simulate_ddm`` are provided to match the paper's table labels.

The neural baselines are heavier and keep their own subpackages, each with the
upstream code, a README citing the original work, and its LICENSE:
:mod:`competitors.rtify`, :mod:`competitors.crnn`, :mod:`competitors.cnet`,
:mod:`competitors.convrnn`, :mod:`competitors.msdnet`, :mod:`competitors.scan`.
They are not imported here — several pull in large or optional dependencies.
Train or evaluate them through the per-benchmark entry points, e.g.
``competitors/cnet/train_sat.py`` or ``competitors/rtify/train_rdm.py``.
"""

from competitors.classical_utils import aic, bic, prepare_data, print_fit
from competitors.ddm import fit_rdm, simulate_rdm
from competitors.lba import fit_lba, simulate_lba
from competitors.lnr import fit_lnr, simulate_lnr, simulate_lnr_with_samples

#: The paper's tables label the racing diffusion model "DDM".
fit_ddm = fit_rdm
simulate_ddm = simulate_rdm

__all__ = [
    "prepare_data", "print_fit", "aic", "bic",
    "fit_lnr", "simulate_lnr", "simulate_lnr_with_samples",
    "fit_lba", "simulate_lba",
    "fit_rdm", "simulate_rdm", "fit_ddm", "simulate_ddm",
]
