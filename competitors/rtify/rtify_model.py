"""RTify models for the RDM benchmark, importable without running training.

`train_rdm.py` is a training script with heavy module-level setup; the pipeline
notebooks only need the architectures and an inference path, so the model
definitions live here and `train_rdm.py` imports them.

RTify decides in two stages:

1. An AlexNet+BatchNorm+LSTM **backbone** consumes a 150-frame random-dot movie
   and emits a 4096-d hidden state per frame.
2. An evidence-accumulation **head** projects those states to a scalar,
   accumulates it, and stops when |accumulated evidence| crosses a learned
   threshold. The crossing index is the reaction time; a soft temporal mask
   around it selects the frames that determine the choice.

Only the head is trained against human RTs, so the backbone states are
identical across head-training runs and the original authors pre-compute them
once. That cache is ~33 GB and is therefore not shipped — see
`competitors/weights/rdm/README.md`. :class:`RTifyModel` reports what is
present via :meth:`RTifyModel.availability` rather than failing late.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent


def _vendored(module_name: str):
    """Load a vendored top-level RTify module by path.

    The upstream files sit next to this one and assume their directory is on
    ``sys.path``. Loading them by location instead keeps the generic names they
    use (``utils``, ``RNN_decision``) out of the import namespace, where they
    would shadow the other vendored competitors.
    """
    spec = importlib.util.spec_from_file_location(
        f"competitors.rtify._vendored.{module_name}",
        str(_HERE / f"{module_name}.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Resolved the same way as train_rdm.py so both see one set of weights.
CKPT_DIR = Path(os.environ.get(
    "RDM_CKPT_DIR", str(REPO_ROOT / "competitors" / "weights" / "rdm")))
BACKBONE_NAME = "AlexNet_BN_LSTM_norm"

# Coherence levels in *descending* order — RTify's own convention.
COHERENCE_LIST = [51.2, 25.6, 12.8, 6.4, 3.2, 1.6, 0.8]
N_COH = len(COHERENCE_LIST)

# Synthetic video parameters (match the original backbone training exactly).
STIM_SIZE = 42        # pixels — AlexNet reduces a 42px frame to a 1x1 map
TIME_STEPS = 150      # frames per video
SPEED = 5.0 / 3       # dots/frame
DOT_SIZE = 1
GROUPS = 3
N_DOT = 6
FS = 75               # frames per second
DT = 1.0 / FS

# Synthetic movies drawn per coherence for the RT pool the NLL's KDE is fitted
# to. Part of the evaluation protocol, not a tuning knob: it is both the size of
# that pool and — because the draws come off one generator — part of the draw
# order. See :meth:`RTifyModel.sample_onto_trials`.
N_RT_POOL = 4096

__all__ = [
    "AlexNet_BN_LSTM", "AlexNet_BN_LSTM_wrapper", "RTifyModel",
    "COHERENCE_LIST", "N_COH", "TIME_STEPS", "DT", "STIM_SIZE", "FS",
    "N_RT_POOL", "CKPT_DIR", "BACKBONE_NAME",
]


class AlexNet_BN_LSTM(nn.Module):
    """
    AlexNet + BatchNorm + LSTM backbone (replicates RTify/RDM/AlexNet_BN_LSTM_backbone.py).

    Input  : (B, T, 3, H, W)   H=W=42
    Output : (B, 1) logit           when self.output = False  (training mode)
             (B, T, hidden_size)    when self.output = True   (precompute mode)
    """

    def __init__(self, time_steps: int = TIME_STEPS, hidden_size: int = 4096):
        super().__init__()
        self.time_steps  = time_steps
        self.hidden_size = hidden_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 64,  kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),   nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64,  192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),  nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),  nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  nn.ReLU(inplace=True),
        )
        # 42px input → 1×1 spatial feature map → 256-dim LSTM input
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=hidden_size,
            num_layers=1, batch_first=True,
        )
        self.fc1    = nn.Linear(hidden_size, 1)
        self.fc2    = nn.Linear(time_steps, 1)
        self.output = False   # toggled to True for LSTM-state precomputation

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W)
        feat_seq = []
        for t in range(self.time_steps):
            feat = self.features(x[:, t])          # (B, 256, 1, 1)
            feat = feat.squeeze(3).permute(0, 2, 1)  # (B, 1, 256)
            feat_seq.append(feat)
        feat_seq = torch.cat(feat_seq, dim=1)      # (B, T, 256)

        lstm_out, _ = self.lstm(feat_seq)          # (B, T, hidden_size)

        if self.output:
            return lstm_out                        # (B, T, hidden_size)

        logits = self.fc1(lstm_out).squeeze(-1)    # (B, T)
        return self.fc2(logits)                    # (B, 1)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class AlexNet_BN_LSTM_wrapper(nn.Module):
    """
    RTify evidence-accumulation wrapper around the frozen backbone.

    Takes pre-computed LSTM hidden states (B, T, 4096) — never raw video.

    Trainable parameters
    --------------------
    linear    : evidence projection  4096 → 1 (per time step)
    w, b      : temporal re-weighting of the evidence trajectory
    threshold : accumulation threshold

    Frozen parameters (loaded from backbone checkpoint via strict=False)
    -------------------------------------------------------------------
    fc1, fc2  : used to compute the *decision logit* (what the model chose)
    """

    def __init__(self, time_steps: int = TIME_STEPS, dt: float = DT, sigma: float = 2.0):
        super().__init__()
        self.time_steps = time_steps
        self.dt         = dt
        self.sigma      = sigma

        # Loaded from backbone checkpoint (frozen after loading)
        self.fc1 = nn.Linear(4096, 1)
        self.fc2 = nn.Linear(time_steps, 1)

        # RTify-specific trainable parameters
        self.linear    = nn.Linear(4096, 1)
        self.w         = nn.Parameter(torch.ones(1, time_steps))
        self.b         = nn.Parameter(torch.zeros(1, time_steps))
        self.threshold = nn.Parameter(torch.tensor(0.2))

    def forward(self, lstm_out: torch.Tensor):
        # Loaded lazily so importing this module stays cheap; `_vendored`
        # resolves RNN_decision.py by path, so no sys.path setup is needed.
        DiffDecision = _vendored("RNN_decision").DiffDecision

        # lstm_out: (B, T, 4096)
        # ── Evidence trajectory ────────────────────────────────────────────
        s_traj       = self.linear(lstm_out).squeeze(-1) * self.w + self.b  # (B, T)
        s_accumulated = torch.cumsum(s_traj, dim=1)                          # (B, T)
        s_abs        = torch.abs(s_accumulated)                              # (B, T)

        # Derivative for the backward pass of DiffDecision
        dsdt = torch.diff(s_abs, dim=1)
        dsdt = torch.cat([dsdt[:, :1], dsdt], dim=1)                        # (B, T)

        # ── Differentiable decision time ───────────────────────────────────
        decision_time = DiffDecision.apply(s_abs - self.threshold, dsdt)    # (B,)

        # ── Soft temporal attention: down-weight frames after decision ─────
        t_range    = torch.arange(self.time_steps, device=lstm_out.device)
        soft_index = torch.sigmoid(
            (decision_time.unsqueeze(1) - t_range + 0.5) * self.sigma
        )                                                                     # (B, T)

        # ── Decision logit (weighted sum of per-frame logits) ──────────────
        logit_traj     = self.fc1(lstm_out).squeeze(-1) * self.fc2.weight   # (B, T)
        decision_logit = (logit_traj * soft_index).sum(dim=1, keepdim=True) \
                         + self.fc2.bias                                     # (B, 1)

        return decision_logit, decision_time   # (B, 1), (B,)


class RTifyModel:
    """Inference-only RTify: load the trained head, score backbone states.

    The head alone is enough to evaluate RTify on RDM, because the backbone
    contribution is entirely captured by its cached hidden states.

    Examples
    --------
    >>> rtify = RTifyModel(device='cuda')           # doctest: +SKIP
    >>> rtify.availability()                        # doctest: +SKIP
    {'head': True, 'backbone': True, 'backbone_outputs': False}
    >>> choices, rts = rtify.predict(lstm_states)   # doctest: +SKIP
    """

    #: Not distributed (77 KB); trained by competitors/rtify/train_rdm.py.
    HEAD_PATH = CKPT_DIR / "rtify_head.pth"
    #: Not distributed (~886 MB); trained by competitors/rtify/train_rdm.py.
    BACKBONE_PATH = CKPT_DIR / "rtify_backbone.pth"
    #: NOT shipped (~33 GB) — see competitors/weights/rdm/README.md.
    BACKBONE_OUTPUTS_PATH = CKPT_DIR / "rtify_backbone_states.pth"

    def __init__(self, device="cpu", time_steps: int = TIME_STEPS, dt: float = DT,
                 head_path=None, load: bool = True):
        self.device = torch.device(device)
        self.dt = dt
        self.head_path = Path(head_path) if head_path else self.HEAD_PATH
        self.model = AlexNet_BN_LSTM_wrapper(time_steps=time_steps, dt=dt)
        self.model.to(self.device).eval()
        self.loaded = False
        if load:
            self.load_head()

    # ── availability ────────────────────────────────────────────────────────
    @classmethod
    def availability(cls) -> dict:
        """Which RTify files are on disk."""
        return {
            "head": cls.HEAD_PATH.exists(),
            "backbone": cls.BACKBONE_PATH.exists(),
            "backbone_outputs": cls.BACKBONE_OUTPUTS_PATH.exists(),
        }

    @classmethod
    def missing_report(cls) -> str:
        """Human-readable note on what is missing and how to obtain it."""
        have = cls.availability()
        lines = []
        if not have["head"]:
            lines.append(f"  head weights absent: {cls.HEAD_PATH}")
        if not have["backbone"]:
            lines.append(f"  backbone absent: {cls.BACKBONE_PATH} "
                         "(train it with competitors/rtify/train_rdm.py)")
        if not have["backbone_outputs"]:
            lines.append(
                f"  cached backbone states absent: {cls.BACKBONE_OUTPUTS_PATH}\n"
                "    ~33 GB, not shipped.\n"
                "    It is not needed for the RDM tables: pipelines/rdm.ipynb\n"
                "    reads the per-trial predictions from\n"
                "    outputs/rdm/rtify_predictions.npz.\n"
                "    To recompute from the model instead, put the cache at the\n"
                "    path above (or point RDM_CKPT_DIR at a directory holding\n"
                "    rtify_backbone_states.pth), or regenerate it from the\n"
                "    backbone with competitors/rtify/train_rdm.py.\n"
                "    See competitors/weights/rdm/README.md.")
        return "\n".join(lines) if lines else "  all RTify files present"

    def load_head(self):
        """Load the trained accumulator head; backbone fc1/fc2 ride along."""
        if not self.head_path.exists():
            raise FileNotFoundError(
                f"RTify head weights not found at {self.head_path}\n"
                + self.missing_report())
        ckpt = torch.load(self.head_path, map_location=self.device,
                          weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.loaded = True
        return self

    # ── inference ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, lstm_states: torch.Tensor, batch_size: int = 256):
        """Choices and reaction times for pre-computed backbone states.

        Parameters
        ----------
        lstm_states
            (N, T, 4096) cached LSTM hidden states.
        batch_size
            Trials per forward pass; the states are large, so this is chunked.

        Returns
        -------
        choices : (N,) int64 tensor
            1 where the decision logit is positive, else 0 — the same
            thresholding the training loop uses for accuracy.
        rts : (N,) float32 tensor
            Reaction times in **seconds** (crossing index x dt).
        """
        if not self.loaded:
            self.load_head()
        if lstm_states.dim() != 3:
            raise ValueError(
                f"expected (N, T, 4096) backbone states, got {tuple(lstm_states.shape)}")

        choices, rts = [], []
        for i in range(0, lstm_states.shape[0], batch_size):
            chunk = lstm_states[i:i + batch_size].to(self.device).float()
            logit, decision_idx = self.model(chunk)
            choices.append((logit.squeeze(-1) > 0).long().cpu())
            rts.append((decision_idx.float() * self.dt).cpu())

        return torch.cat(choices), torch.cat(rts)

    @torch.no_grad()
    def sample_onto_trials(self, backbone_outputs, directions, true_labels,
                           rng, n_pool: int = N_RT_POOL, batch_size: int = 256):
        """RTify's predictions for a block of real trials at one coherence.

        RTify never saw the human stimuli: it was trained on synthetic movies,
        and the cache holds one backbone trajectory per synthetic movie. So for
        each real trial it draws a movie **with replacement** from the cache,
        scores it against the direction it was generated with, and transplants
        that correctness onto the real trial's label — RTify contributes its
        accuracy and its RT distribution, not a per-stimulus prediction it
        could not have.

        Two draws happen here, in this order, off the same generator:

        1. ``len(true_labels)`` movies — the per-trial choices and RTs;
        2. ``n_pool`` further movies — the RT pool the NLL's KDE is fitted to,
           which is a property of the model at that coherence rather than of
           the particular trials, and so is sampled independently.

        The order is the contract. Dropping the second draw does not merely
        lose the pool: it leaves the generator ``n_pool`` draws behind, so
        every coherence after the first gets different movies. Keep both
        draws even if the caller discards the pool.

        Parameters
        ----------
        backbone_outputs
            (N_cached, T, 4096) states for this coherence.
        directions
            (N_cached, 1) the direction each cached movie was generated with.
        true_labels
            (n_trials,) the real trials' direction labels, 0/1.
        rng
            ``numpy.random.Generator`` — seed it to make the draw repeatable.
        n_pool
            Movies drawn for the RT pool. Protocol, not a knob.

        Returns
        -------
        choices  : (n_trials,) int64 ndarray
        rts      : (n_trials,) float64 ndarray, seconds
        pool_rts : (n_pool,) float64 ndarray, seconds
        """
        if not self.loaded:
            self.load_head()
        true_labels = np.asarray(true_labels)
        n = len(true_labels)
        if n == 0:
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=float),
                    np.empty(0, dtype=float))

        n_avail = backbone_outputs.shape[0]

        sel = rng.integers(0, n_avail, size=n)
        correct, rts = [], []
        for i in range(0, n, batch_size):
            idx = sel[i:i + batch_size]
            chunk = backbone_outputs[idx].to(self.device).float()
            logit, decision_idx = self.model(chunk)
            dirs = directions[idx].to(self.device)
            correct.append(((logit > 0) == (dirs > 0)).reshape(-1).cpu().numpy())
            rts.append((decision_idx.float() * self.dt).reshape(-1).cpu().numpy())

        correct = np.concatenate(correct).astype(bool)
        choices = np.where(correct, true_labels, 1 - true_labels).astype(np.int64)

        pool_idx = rng.integers(0, n_avail, size=n_pool)
        pool_rts = []
        for i in range(0, n_pool, batch_size):
            idx = pool_idx[i:i + batch_size]
            chunk = backbone_outputs[idx].to(self.device).float()
            _, decision_idx = self.model(chunk)
            pool_rts.append((decision_idx.float() * self.dt).reshape(-1).cpu().numpy())

        return (choices, np.concatenate(rts).astype(float),
                np.concatenate(pool_rts).astype(float))

    @torch.no_grad()
    def predict_from_video(self, video: torch.Tensor, backbone=None,
                           batch_size: int = 16):
        """Convenience path: run the backbone then the head on raw movies.

        Much slower than :meth:`predict` on cached states; useful for a handful
        of trials or for regenerating the cache.
        """
        if backbone is None:
            backbone = self.load_backbone()
        backbone.output = True
        states = []
        for i in range(0, video.shape[0], batch_size):
            states.append(backbone(video[i:i + batch_size].to(self.device)).cpu())
        return self.predict(torch.cat(states), batch_size=batch_size)

    def load_backbone(self, time_steps: int = TIME_STEPS):
        """Instantiate and load the AlexNet+LSTM backbone."""
        if not self.BACKBONE_PATH.exists():
            raise FileNotFoundError(
                f"RTify backbone not found at {self.BACKBONE_PATH}\n"
                + self.missing_report())
        backbone = AlexNet_BN_LSTM(time_steps=time_steps)
        ckpt = torch.load(self.BACKBONE_PATH, map_location=self.device,
                          weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        backbone.load_state_dict(state, strict=False)
        return backbone.to(self.device).eval()

    @classmethod
    def load_backbone_outputs(cls, path=None, map_location="cpu"):
        """Load the ~33 GB cache of backbone states, if present."""
        p = Path(path) if path else cls.BACKBONE_OUTPUTS_PATH
        if not p.exists():
            raise FileNotFoundError(
                f"cached backbone states not found at {p}\n" + cls.missing_report())
        return torch.load(p, map_location=map_location, weights_only=False)
