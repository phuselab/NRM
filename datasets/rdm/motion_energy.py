"""Spatiotemporal Gabor motion-energy embedder for RDM videos (Adelson & Bergen 1985, §2.2)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Module-level constants (importable for model configuration) ───────────────

N_DIRECTIONS: int       = 8
SPEEDS:       list[float] = [0.5, 1.0, 2.0]
EMBED_DIM:    int       = N_DIRECTIONS * len(SPEEDS)   # 24

_DEFAULT_FILTER_PARAMS: dict = dict(
    lam=8.0, sigma_xy=4.0, sigma_t=3.0,
    kx=13,   ky=13,        kt=7,
)


# ── Filter bank construction ──────────────────────────────────────────────────

def build_filter_bank(
    n_directions: int   = N_DIRECTIONS,
    speeds:       list  = SPEEDS,
    lam:          float = 8.0,
    sigma_xy:     float = 4.0,
    sigma_t:      float = 3.0,
    kx:           int   = 13,
    ky:           int   = 13,
    kt:           int   = 7,
) -> torch.Tensor:
    """
    Build a 3D Gabor motion-energy filter bank.

    Each (direction, speed) pair yields two filters in phase quadrature:
        g_even = envelope × cos(ω(x cosθ + y sinθ) − ω_t t)
        g_odd  = envelope × sin(ω(x cosθ + y sinθ) − ω_t t)

    Energy from one pair = g_even² + g_odd², which is invariant to the
    spatial phase of the stimulus (Adelson & Bergen 1985).

    Returns
    -------
    fbank : (2 * n_directions * len(speeds), 1, kt, ky, kx) float32 tensor
            Ordered as [even₀, odd₀, even₁, odd₁, …] — one pair per
            (direction, speed) combination.
    """
    directions_deg = np.linspace(0, 360, n_directions, endpoint=False)

    x = np.arange(-(kx // 2), kx // 2 + 1, dtype=np.float32)
    y = np.arange(-(ky // 2), ky // 2 + 1, dtype=np.float32)
    t = np.arange(-(kt // 2), kt // 2 + 1, dtype=np.float32)
    T, Y, X = np.meshgrid(t, y, x, indexing="ij")   # (kt, ky, kx)

    omega   = 2.0 * np.pi / lam
    gauss   = (np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma_xy ** 2))
               * np.exp(-T ** 2 / (2.0 * sigma_t ** 2)))

    filters = []
    for theta_deg in directions_deg:
        theta = np.radians(theta_deg)
        cx, cy = np.cos(theta), np.sin(theta)
        for speed in speeds:
            phase  = omega * (X * cx + Y * cy) - omega * speed * T
            g_even = gauss * np.cos(phase)
            g_odd  = gauss * np.sin(phase)
            g_even -= g_even.mean()   # remove DC component
            g_odd  -= g_odd.mean()
            filters.append(g_even[np.newaxis])   # (1, kt, ky, kx)
            filters.append(g_odd[np.newaxis])

    fbank = np.stack(filters, axis=0)            # (2*N_dir*N_sp, 1, kt, ky, kx)
    return torch.tensor(fbank, dtype=torch.float32)


# ── Embedder class ────────────────────────────────────────────────────────────

class MotionEnergyEmbedder(nn.Module):
    """
    Motion energy embedder for RDM videos.

    Public methods: embed_video, embed_batch; output (T, EMBED_DIM) per video.

    Parameters
    ----------
    n_directions : int
        Number of motion directions (equally spaced over 360°). Default 8.
    speeds : list[float]
        Dot speeds in px/frame for which filters are built. Default [0.5, 1, 2].
    device : str | torch.device
        Inference device.
    **filter_kw
        Override any of: lam, sigma_xy, sigma_t, kx, ky, kt.
    """

    def __init__(
        self,
        n_directions: int               = N_DIRECTIONS,
        speeds:       list | None       = None,
        device:       str | torch.device = "cpu",
        **filter_kw,
    ):
        super().__init__()
        self.n_directions = n_directions
        self.speeds       = list(speeds) if speeds is not None else list(SPEEDS)
        self.n_filters    = n_directions * len(self.speeds)   # = EMBED_DIM
        self.device       = torch.device(device)

        kw = {**_DEFAULT_FILTER_PARAMS, **filter_kw}
        self._kt = kw["kt"]
        self._ky = kw["ky"]
        self._kx = kw["kx"]

        fbank = build_filter_bank(
            n_directions=n_directions,
            speeds=self.speeds,
            **kw,
        )   # (2*N, 1, kt, ky, kx)
        self.register_buffer("fbank", fbank)
        self.to(self.device)

    # ── Public API ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed_video(self, video: np.ndarray) -> torch.Tensor:
        """
        Compute the motion energy embedding for a single RDM video.

        Parameters
        ----------
        video : np.ndarray  (T, H, W)  uint8

        Returns
        -------
        torch.Tensor  (T, n_filters)  float32  on CPU
            Spatial-average motion energy per frame per direction/speed.
            For the default configuration: (150, 24).
        """
        T, H, W = video.shape
        N  = self.n_filters
        kt, ky, kx = self._kt, self._ky, self._kx

        vid = (
            torch.from_numpy(video.astype(np.float32) / 255.0)
            .unsqueeze(0).unsqueeze(0)       # (1, 1, T, H, W)
            .to(self.device)
        )

        # Same-size output via symmetric zero-padding in all three dimensions
        pad        = (kx // 2, kx // 2, ky // 2, ky // 2, kt // 2, kt // 2)
        vid_padded = F.pad(vid, pad, mode="constant", value=0.0)

        # Apply filter bank: (1, 2*N, T, H, W)
        resp   = F.conv3d(vid_padded, self.fbank)
        resp   = resp.squeeze(0)                # (2*N, T, H, W)

        # Energy = cos² + sin² per quadrature pair → (N, T, H, W)
        resp   = resp.view(N, 2, T, H, W)
        energy = (resp ** 2).sum(dim=1)         # (N, T, H, W)

        # Spatial average pool, then put time first
        energy = energy.mean(dim=(-2, -1))      # (N, T)
        energy = energy.permute(1, 0)           # (T, N)

        return energy.cpu()

    @torch.no_grad()
    def embed_batch(self, videos: list[np.ndarray]) -> torch.Tensor:
        """
        Embed a list of videos sequentially.

        Returns
        -------
        torch.Tensor  (len(videos), T, n_filters)
        """
        return torch.stack([self.embed_video(v) for v in videos])
