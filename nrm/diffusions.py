"""Diffusion parameterisations g_φ for the evidence accumulation SDE (§2.1, Eq. 1)."""

import torch
import torch.nn as nn


class LearnableConstantDiffusion(nn.Module):
    """Unconditioned constant diffusion (baseline, not used in final experiments)."""

    def __init__(self, N_CHOICES):
        super().__init__()
        self.log_std = nn.Parameter(torch.ones(N_CHOICES) * (-1.5))

    def forward(self, t, x, cond):
        std = self.log_std.exp()
        return std.expand_as(x)


class ConditionedConstantDiffusion(nn.Module):
    """Diffusion g_φ(c) -> R^N for static stimuli (§2.1, Eq. 1).

    Linear -> LayerNorm -> Softplus ensures positive diffusion coefficients.
    Used by NRM for SAT and SC.
    """

    def __init__(self, n_choices, hidden_size=8, cond_dim=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, n_choices),
            nn.LayerNorm(n_choices),
            nn.Softplus()
        )

    def forward(self, t, x, cond):
        return self.net(cond)


class TimeEmbeddedDiffusion(nn.Module):
    """Time-varying diffusion g_φ(t, c_t) -> R^N for video stimuli (§2.1, Eq. 1).

    Used by TimeEmbeddedNRM for RDM.
    """

    def __init__(self, cond_dim, n_choices, hidden_size=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(cond_dim + hidden_size, n_choices),
            nn.LayerNorm(n_choices),
            nn.Softplus()
        )

    def forward(self, t, x, cond):
        t_val = t.item() if hasattr(t, 'item') else float(t)
        t_input = torch.full((cond.size(0), 1), t_val, device=cond.device)
        t_feat = self.time_mlp(t_input)
        cond_input = torch.cat([cond, t_feat], dim=1)
        return self.net(cond_input)


class FullDiffusion(nn.Module):
    """Correlated N*N diffusion matrix (not used in final experiments)."""

    def __init__(self, N_CHOICES, hidden_size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_CHOICES + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, N_CHOICES * N_CHOICES)
        )
        self.N_CHOICES = N_CHOICES

    def forward(self, t, x):
        batch_size = x.size(0)
        t_input = torch.full((batch_size, 1), t, device=x.device)
        x_input = torch.cat([x, t_input], dim=1)
        flat = self.net(x_input)
        mat = flat.view(batch_size, self.N_CHOICES, self.N_CHOICES)
        # Symmetrise for valid covariance structure
        mat = 0.5 * (mat + mat.transpose(1, 2))
        diag_noise = torch.eye(self.N_CHOICES, device=x.device).unsqueeze(0) * 1e-3
        return mat + diag_noise
