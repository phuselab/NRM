"""Stimulus conditioning: covariates -> conditioning vector c (§2.2, Eq. 2)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditioningEncoder(nn.Module):
    """c = [e^(1) || ... || e^(K)]: per-covariate projection + BatchNorm + concat (Eq. 2).

    Each covariate is independently projected to a fixed-size embedding,
    normalised, and concatenated into the conditioning vector c.
    Used by NRM for SAT and SC (static stimuli).
    """

    def __init__(self, config):
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        self.activations = nn.ModuleDict()

        for name, spec in config.items():
            dim = spec["dim"]

            if spec["type"] == "categorical":
                self.encoders[name] = nn.Embedding(
                    spec["num_embeddings"], dim
                )
                self.activations[name] = None

            elif spec["type"] == "continuous":
                self.encoders[name] = nn.Linear(1, dim)
                self.activations[name] = None

            elif spec["type"] == "raw":
                self.encoders[name] = nn.Linear(
                    spec["input_dim"], dim
                )
                self.activations[name] = nn.ReLU()

            else:
                raise ValueError(f"Unknown covariate type {spec['type']}")

            self.norms[name] = nn.BatchNorm1d(dim)

    def forward(self, covariates, n_reps):
        if covariates is None:
            return None

        embeds = []
        for name in self.encoders:
            covariates[name] = (covariates[name].unsqueeze(-1)
                                if covariates[name].ndim == 1
                                else covariates[name])
            h = self.encoders[name](covariates[name])
            h = self.norms[name](h)
            if self.activations[name] is not None:
                h = self.activations[name](h)
            h = h.repeat_interleave(n_reps, dim=0)
            embeds.append(h)

        return torch.cat(embeds, dim=-1)


class TemporalFlatEncoder(nn.Module):
    """Flattens temporal embeddings (B, T, D) -> (B, out_dim) before encoding (Eq. 2)."""

    def __init__(self, config):
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        self.activations = nn.ModuleDict()

        for name, spec in config.items():
            dim = spec["dim"]

            if spec["type"] == "categorical":
                self.encoders[name] = nn.Embedding(spec["num_embeddings"], dim)
                self.activations[name] = None
            elif spec["type"] == "continuous":
                self.encoders[name] = nn.Linear(1, dim)
                self.activations[name] = None
            elif spec["type"] == "raw":
                self.encoders[name] = nn.Linear(spec["input_dim"], dim)
                self.activations[name] = nn.ReLU()
            else:
                raise ValueError(f"Unknown covariate type {spec['type']}")

            self.norms[name] = nn.BatchNorm1d(dim)

    def forward(self, covariates, n_reps):
        if covariates is None:
            return None

        embeds = []
        for name in self.encoders:
            x = covariates[name]
            if x.ndim == 3:
                x = x.flatten(1)
            elif x.ndim == 1:
                x = x.unsqueeze(-1)

            h = self.encoders[name](x)
            h = self.norms[name](h)
            if self.activations[name] is not None:
                h = self.activations[name](h)
            h = h.repeat_interleave(n_reps, dim=0)
            embeds.append(h)

        return torch.cat(embeds, dim=-1)


class TimeEmbeddedEncoder(nn.Module):
    """Preserves temporal structure: (B, T, D) -> (B, T, out_dim) (Eq. 2).

    Used by TimeEmbeddedNRM (RDM) so that each SDE step receives a
    frame-specific conditioning vector c_t.
    """

    def __init__(self, config):
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        self.activations = nn.ModuleDict()

        for name, spec in config.items():
            dim = spec["dim"]

            if spec["type"] == "categorical":
                self.encoders[name] = nn.Embedding(spec["num_embeddings"], dim)
                self.activations[name] = None
            elif spec["type"] == "continuous":
                self.encoders[name] = nn.Linear(1, dim)
                self.activations[name] = None
            elif spec["type"] == "raw":
                self.encoders[name] = nn.Linear(spec["input_dim"], dim)
                self.activations[name] = nn.ReLU()
            else:
                raise ValueError(f"Unknown covariate type {spec['type']}")

            self.norms[name] = nn.BatchNorm1d(dim)

    def forward(self, covariates, n_reps):
        if covariates is None:
            return None

        embeds = []
        for name in self.encoders:
            x = covariates[name]

            if x.ndim == 3:
                B, T, _ = x.shape
                x = x.flatten(0, 1)
            elif x.ndim == 1:
                B, T = x.shape[0], 1
                x = x.unsqueeze(-1)
            else:
                B, T = x.shape[0], 1

            h = self.encoders[name](x)
            h = self.norms[name](h)
            if self.activations[name] is not None:
                h = self.activations[name](h)

            h = h.view(B, T, -1)
            h = h.repeat_interleave(n_reps, dim=0)
            embeds.append(h)

        return torch.cat(embeds, dim=-1)
