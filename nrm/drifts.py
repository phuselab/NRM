"""Drift parameterisations f_θ for the evidence accumulation SDE (§2.1, Eq. 1)."""

import torch
import torch.nn as nn


class LearnableConstantDrift(nn.Module):
    """Unconditioned constant drift (baseline, not used in final experiments)."""

    def __init__(self, n_choices):
        super().__init__()
        self.drift = nn.Parameter(torch.ones(n_choices) * 0.5)

    def forward(self, t, x, cond):
        return self.drift.expand_as(x)


class ConditionedConstantDriftNet(nn.Module):
    """Drift f_θ(c) -> R^N for static stimuli (§2.1, Eq. 1).

    Maps the conditioning vector to per-accumulator drift rates via
    Linear -> LayerNorm -> Softmax. Used by NRM for SAT and SC.
    """

    def __init__(self, cond_dim, n_choices, hidden_size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, n_choices),
            nn.LayerNorm(n_choices),
            nn.Softmax(dim=1),
        )

    def forward(self, t, x, cond):
        return self.net(cond)


class TimeEmbeddedDriftNet(nn.Module):
    """Time-varying drift f_θ(t, c_t) -> R^N for video stimuli (§2.1, Eq. 1).

    Concatenates frame-level conditioning c_t with a learned time embedding
    before mapping to per-accumulator drift rates. Used by TimeEmbeddedNRM for RDM.
    """

    def __init__(self, cond_dim, n_choices, hidden_size=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(cond_dim + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_choices)
        )

    def forward(self, t, x, cond):
        t_val = t.item() if hasattr(t, 'item') else float(t)
        t_input = torch.full((cond.size(0), 1), t_val, device=cond.device)
        t_feat = self.time_mlp(t_input)
        cond_input = torch.cat([cond, t_feat], dim=1)
        return self.net(cond_input)
