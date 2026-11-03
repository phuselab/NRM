"""Neural Race Model: stimulus-conditional neural SDE with racing accumulators.

Implements the core NRM (static conditioning, §2.2) and its temporal variant
TimeEmbeddedNRM (frame-level conditioning for video stimuli, used for RDM).
"""

import numpy as np
import torch
import torchsde
import torch.nn.functional as F
import torch.nn as nn
import random


def detect_noise_type(diff_net):
    from nrm.diffusions import FullDiffusion
    if isinstance(diff_net, FullDiffusion):
        return "general"
    else:
        return "diagonal"


def shifted_sigmoid(x, steepness=100., shift=0.05):
    """Smooth approximation to the threshold-crossing indicator (§2.4, Eq. 3)."""
    return torch.sigmoid(steepness * (x + shift))


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NRM(torchsde.SDEIto):
    """Neural Race Model for static stimuli (§2).

    N accumulators integrate evidence via the SDE dX = f_θ(t,X,c)dt + g_φ(t,X,c)dW
    (Eq. 1) and race towards a stimulus-dependent threshold Θ(c) (§2.3).
    Used for the SAT and SC benchmarks.
    """

    sde_type = "ito"

    def __init__(self, drift_net, diff_net, threshold_net, cond_encoder,
                 n_choices, t_max, dt, device, noise_type=None,
                 kappa=1., ndt=0.01, solver='euler', activation='sigmoid',
                 use_adjoint=False):
        if noise_type is None:
            noise_type = detect_noise_type(diff_net)
        super().__init__(noise_type=noise_type)
        self.f_net = drift_net.to(device)
        self.g_net = diff_net.to(device)
        self.cond_encoder = cond_encoder.to(device) if cond_encoder is not None else None
        self.n_choices = n_choices
        self.t_max = t_max
        self.dt = dt
        self.device = device
        self.threshold_net = threshold_net.to(device)
        self.kappa = kappa
        self.solver = solver
        self.activation = activation
        self.use_adjoint = use_adjoint
        self.log_ndt = nn.Parameter(
            torch.log(torch.ones(n_choices, device=device) * ndt),
            requires_grad=False
        )
        self.x0 = nn.Parameter(
            torch.full((n_choices,), 0., device=device),
            requires_grad=False
        )

    @property
    def ndt(self):
        return torch.exp(self.log_ndt)

    # -- torchsde interface: drift and diffusion (Eq. 1) --

    def f(self, t, y):
        return self.f_net(t, y, self.cond)

    def g(self, t, y):
        return self.g_net(t, y, self.cond)

    def _sdeint(self, x0, ts):
        if self.use_adjoint:
            return torchsde.sdeint_adjoint(self, x0, ts, method=self.solver)
        return torchsde.sdeint(self, x0, ts, method=self.solver)

    def forward_pass(self, covariates=None, batch_size=512, n_reps=1):
        """Differentiable forward pass for training (§2.4–§2.6).

        Returns (choice_probs, fpt_density, ts, ndt). The soft FPT density
        and choice probabilities are the training-time surrogates; at test
        time use simulate_nDDM instead (§2.7).
        """
        B, N, M = batch_size, self.n_choices, n_reps
        ts = torch.linspace(0, self.t_max, steps=int(self.t_max / self.dt),
                            device=self.device)
        T = ts.size(0)

        if self.cond_encoder is not None:
            if covariates is None:
                raise ValueError("cond_encoder present: pass covariates dict")
            cond = self.cond_encoder(covariates, n_reps)
        else:
            cond = None
        self.cond = cond

        # M independent SDE realisations per data point (§2.4)
        x0 = self.x0.unsqueeze(0).expand(B, -1).to(self.device)
        x0 = x0.repeat_interleave(M, dim=0)

        xs = self._sdeint(x0, ts)           # (T, B*M, N)
        xs = xs.permute(1, 0, 2)
        xs = xs.view(B, M, T, N)

        self.theta_t = self.threshold_net(self.cond)  # Θ(c) (§2.3)

        # Soft crossing probability σ(κ·(X - Θ)) (Eq. 3)
        crossing_probs = shifted_sigmoid(
            xs - self.theta_t.view(B, M, 1, 1), self.kappa, 0.05
        )

        # Overall crossing: max over accumulators (Eq. 4)
        prob_cross = crossing_probs.max(dim=3).values
        # FPT increments ΔP (Eq. 5) and normalised FPT density (Eq. 6)
        fcd = torch.clamp(
            F.pad(prob_cross[:, :, 1:] - prob_cross[:, :, :-1], (1, 0), value=0),
            min=0.0
        )
        first_cross_density = fcd.mean(dim=1)
        first_cross_density = first_cross_density / (
            first_cross_density.sum(dim=1, keepdim=True) + 1e-8
        )

        # Temporally-weighted evidence X̄ and choice probabilities (Eqs. 7–8)
        temporal_weights = first_cross_density.unsqueeze(2)
        weighted_evidence = (xs.mean(dim=1) * temporal_weights).sum(dim=1)
        choice_probs = torch.softmax(10. * weighted_evidence, dim=1)

        return choice_probs, first_cross_density, ts, self.ndt

    @torch.no_grad()
    def simulate_nDDM(self, batch_size, covariates=None, soft=True,
                      n_reps=32, t=1.0 - 1e-8):
        """Hard simulation at test time (§2.7).

        Returns (trajectories, ts, choices, rts). Choices and RTs are
        aggregated over n_reps realisations (mode for choice, median for RT).
        """
        ts = torch.linspace(0, self.t_max, int(self.t_max / self.dt)).to(self.device)
        x0 = self.x0.unsqueeze(0).expand(batch_size, -1).to(self.device)
        x0 = x0.repeat_interleave(n_reps, dim=0)

        B, N, M = batch_size, self.n_choices, n_reps

        if self.cond_encoder is not None:
            if covariates is None:
                raise ValueError("cond_encoder present: pass covariates dict")
            cond = self.cond_encoder(covariates, n_reps=n_reps)
        else:
            cond = None
        self.cond = cond

        xs = self._sdeint(x0, ts)
        xs = xs.permute(1, 0, 2)

        self.theta_t = self.threshold_net(self.cond)

        if soft:
            margin = xs - self.theta_t.unsqueeze(1).unsqueeze(2)
            crossing_probs = shifted_sigmoid(margin, self.kappa, 0.05)
            crossed = (crossing_probs >= t).any(dim=2)
        else:
            above_thresh = (xs >= self.threshold)
            crossed = above_thresh.any(dim=2)

        first_cross = crossed.float().cumsum(dim=1).eq(1)
        rts_idx = first_cross.float().argmax(dim=1)

        has_crossed = crossed.any(dim=1)

        rts = ts[rts_idx]
        rts[~has_crossed] = self.t_max

        choices = torch.full((batch_size * n_reps,), -1, dtype=torch.long).to(self.device)
        for i in range(batch_size * n_reps):
            if has_crossed[i]:
                choices[i] = xs[i, rts_idx[i]].argmax()

        choices_reshaped = choices.view(batch_size, n_reps)
        rts_reshaped = rts.view(batch_size, n_reps)

        aggregated_choices = torch.mode(choices_reshaped, dim=1).values
        aggregated_rts = rts_reshaped.median(dim=1).values

        return xs, ts, aggregated_choices, aggregated_rts


class TimeEmbeddedNRM(torchsde.SDEIto):
    """Temporal NRM: frame-level conditioning c_t for video stimuli (§2).

    Extends NRM with time-varying conditioning: the drift, diffusion and
    threshold receive a different c_t at each SDE step, interpolated from
    the stimulus frame rate to the integration grid. Used for the RDM benchmark.
    """

    sde_type = "ito"

    def __init__(self, drift_net, diff_net, threshold_net, cond_encoder,
                 n_choices, t_max, fs, dt, device, noise_type=None,
                 kappa=1., ndt=0.01, solver='euler', activation='sigmoid',
                 use_adjoint=False):
        if noise_type is None:
            noise_type = detect_noise_type(diff_net)
        super().__init__(noise_type=noise_type)
        self.f_net = drift_net.to(device)
        self.g_net = diff_net.to(device)
        self.cond_encoder = cond_encoder.to(device) if cond_encoder is not None else None
        self.n_choices = n_choices
        self.t_max = t_max
        self.fs = fs
        self.dt = dt
        self.device = device
        self.threshold_net = threshold_net.to(device)
        self.kappa = kappa
        self.solver = solver
        self.activation = activation
        self.use_adjoint = use_adjoint
        self.log_ndt = nn.Parameter(
            torch.log(torch.ones(n_choices, device=device) * ndt),
            requires_grad=False
        )
        self.x0 = nn.Parameter(
            torch.full((n_choices,), 0., device=device),
            requires_grad=False
        )

    @property
    def ndt(self):
        return torch.exp(self.log_ndt)

    def _t_to_idx(self, t):
        """Map continuous SDE time to the nearest conditioning frame index."""
        t_val = t.item() if hasattr(t, 'item') else float(t)
        return min(int(round(t_val / self.dt)), self.cond.shape[1] - 1)

    # -- torchsde interface: drift and diffusion with time-varying c_t (Eq. 1) --

    def f(self, t, y):
        cond_t = self.cond[:, self._t_to_idx(t), :]
        return self.f_net(t, y, cond_t)

    def g(self, t, y):
        cond_t = self.cond[:, self._t_to_idx(t), :]
        return self.g_net(t, y, cond_t)

    def _sdeint(self, x0, ts):
        if self.use_adjoint:
            return torchsde.sdeint_adjoint(self, x0, ts, method=self.solver)
        return torchsde.sdeint(self, x0, ts, method=self.solver)

    def interpolate_covariates(self, covariates, ts, video_duration=None,
                               mode='nearest'):
        """Resample conditioning from stimulus frame rate to the SDE time grid.

        Frames beyond the video duration are held at the last frame value.
        """
        B, T_orig, D = covariates.shape
        T_new = ts.shape[0]

        if video_duration is not None:
            mask = ts <= video_duration
            T_video = mask.sum().item()
        else:
            T_video = T_new

        interp_kwargs = dict(size=T_video, mode=mode)
        if mode == 'linear':
            interp_kwargs['align_corners'] = True

        x = covariates.permute(0, 2, 1)
        x = F.interpolate(x, **interp_kwargs)
        x = x.permute(0, 2, 1)

        # Hold last frame for any SDE steps past the video duration
        T_pad = T_new - T_video
        if T_pad > 0:
            last_frame = x[:, -1:, :]
            pad = last_frame.expand(B, T_pad, D)
            x = torch.cat([x, pad], dim=1)

        return x

    def forward_pass(self, covariates=None, batch_size=512, n_reps=1):
        """Differentiable forward pass with temporal conditioning (§2.4–§2.6)."""
        B, N, M = batch_size, self.n_choices, n_reps

        duration = covariates['img_embedding'].size(1) / self.fs
        ts = torch.linspace(0, self.t_max, steps=int(self.t_max / self.dt),
                            device=self.device)
        T = ts.size(0)

        if self.cond_encoder is not None:
            if covariates is None:
                raise ValueError("cond_encoder present: pass covariates dict")
            cond = self.cond_encoder(covariates, n_reps)
            cond = self.interpolate_covariates(cond, ts, duration, 'nearest')
        else:
            cond = None
        self.cond = cond

        x0 = self.x0.unsqueeze(0).expand(B, -1).to(self.device)
        x0 = x0.repeat_interleave(M, dim=0)

        xs = self._sdeint(x0, ts)
        xs = xs.permute(1, 0, 2)
        xs = xs.view(B, M, T, N)

        # Per-step threshold Θ(c_t) (§2.3)
        theta_seq = self.threshold_net(self.cond)
        # Soft crossing, FPT density, choice probs (Eqs. 3–8)
        crossing_probs = shifted_sigmoid(
            xs - theta_seq.view(B, M, T, 1), self.kappa, 0.05
        )

        prob_cross = crossing_probs.max(dim=3).values
        fcd = torch.clamp(
            F.pad(prob_cross[:, :, 1:] - prob_cross[:, :, :-1], (1, 0), value=0),
            min=0.0
        )
        first_cross_density = fcd.mean(dim=1)
        first_cross_density = first_cross_density / (
            first_cross_density.sum(dim=1, keepdim=True) + 1e-8
        )

        temporal_weights = first_cross_density.unsqueeze(2)
        weighted_evidence = (xs.mean(dim=1) * temporal_weights).sum(dim=1)
        choice_probs = torch.softmax(10. * weighted_evidence, dim=1)

        return choice_probs, first_cross_density, ts, self.ndt

    @torch.no_grad()
    def simulate_nDDM(self, batch_size, covariates=None, soft=True,
                      n_reps=32, t=1.0 - 1e-8):
        """Hard simulation at test time with temporal conditioning (§2.7)."""
        ts = torch.linspace(0, self.t_max, int(self.t_max / self.dt)).to(self.device)
        x0 = self.x0.unsqueeze(0).expand(batch_size, -1).to(self.device)
        x0 = x0.repeat_interleave(n_reps, dim=0)

        B, N, M = batch_size, self.n_choices, n_reps

        if self.cond_encoder is not None:
            if covariates is None:
                raise ValueError("cond_encoder present: pass covariates dict")
            cond = self.cond_encoder(covariates, n_reps=n_reps)
            duration = covariates['img_embedding'].size(1) / self.fs
            cond = self.interpolate_covariates(cond, ts, duration, 'nearest')
        else:
            cond = None
        self.cond = cond

        xs = self._sdeint(x0, ts)
        xs = xs.permute(1, 0, 2)

        theta_seq = self.threshold_net(self.cond)

        if soft:
            margin = xs - theta_seq.unsqueeze(2)
            crossing_probs = shifted_sigmoid(margin, self.kappa, 0.05)
            crossed = (crossing_probs >= t).any(dim=2)
        else:
            above_thresh = (xs >= theta_seq.unsqueeze(2))
            crossed = above_thresh.any(dim=2)

        first_cross = crossed.float().cumsum(dim=1).eq(1)
        rts_idx = first_cross.float().argmax(dim=1)

        has_crossed = crossed.any(dim=1)

        rts = ts[rts_idx]
        rts[~has_crossed] = self.t_max

        choices = torch.full((batch_size * n_reps,), -1, dtype=torch.long).to(self.device)
        for i in range(batch_size * n_reps):
            if has_crossed[i]:
                choices[i] = xs[i, rts_idx[i]].argmax()

        choices_reshaped = choices.view(batch_size, n_reps)
        rts_reshaped = rts.view(batch_size, n_reps)

        aggregated_choices = torch.mode(choices_reshaped, dim=1).values
        aggregated_rts = rts_reshaped.median(dim=1).values

        return xs, ts, aggregated_choices, aggregated_rts
