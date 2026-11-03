"""Learnable threshold Θ(c) for the racing accumulators (§2.3).

Softplus ensures positive thresholds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionedConstantThreshold(nn.Module):
    """Scalar threshold Θ(c) for static stimuli (§2.3).

    MLP -> Softplus outputs a single positive threshold per trial.
    Used by NRM for SAT and SC.
    """

    def __init__(self, cond_dim, t_max, dt, device="cuda"):
        super().__init__()
        self.t_max = t_max
        self.dt = dt
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 16, bias=False),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 1, bias=False),
            nn.LayerNorm(1),
            nn.Softplus()
        )

    def forward(self, cond):
        params = self.net(cond)
        return params[:, 0]


class TimeEmbeddedThreshold(nn.Module):
    """Per-timestep threshold Θ(c_t) from frame-level conditioning (§2.3).

    Produces one positive threshold value per SDE step.
    Used by TimeEmbeddedNRM for RDM.
    """

    def __init__(self, cond_dim, t_max, dt, device="cuda"):
        super().__init__()
        self.t_max = t_max
        self.dt = dt
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 16, bias=False),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 1, bias=False),
            nn.Softplus()
        )

    def forward(self, cond):
        BM, T, D = cond.shape
        h = cond.flatten(0, 1)
        out = self.net(h)
        return out.view(BM, T)
