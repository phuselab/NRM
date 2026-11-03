"""Benchmark configuration and model construction."""

import os
import yaml
import torch

from nrm.model import NRM, TimeEmbeddedNRM
from nrm.drifts import ConditionedConstantDriftNet, TimeEmbeddedDriftNet
from nrm.diffusions import ConditionedConstantDiffusion, TimeEmbeddedDiffusion
from nrm.thresholds import ConditionedConstantThreshold, TimeEmbeddedThreshold
from nrm.conditioning import ConditioningEncoder, TimeEmbeddedEncoder


# -- Per-benchmark defaults ---------------------------------------------------

BENCHMARK_DEFAULTS = {
    "rdm": {
        "model_class": "TimeEmbeddedNRM",
        "n_choices": 2,
        "cond_dim": 32,
        "dt": 0.01,
        "t_max": 3.0,
        "fs": 75,
        "kappa": 100,
        "solver": "euler",
        "use_adjoint": False,
        "encoder_config": {
            "img_embedding": {"type": "raw", "input_dim": 24, "dim": 32},
        },
        "lr": 1e-3,
        "num_epochs": 100,
        "batch_size": 256,
        "n_reps": 64,
        "patience": 10,
        "val_size": 0.1,
        "test_size": 0.1,
        "seed": 42,
        "weights_file": "NRM_rdm.pth",
        "embeddings_file": "rdm/rdm_embeddings.pt",
    },
    "sat": {
        "model_class": "NRM",
        "n_choices": 16,
        "cond_dim": 128,
        "dt": 0.01,
        "t_max": 5.0,
        "kappa": 100,
        "solver": "euler",
        "use_adjoint": False,
        "encoder_config": {
            "img_embedding": {"type": "raw", "input_dim": 384, "dim": 64},
            "time_limit": {"type": "raw", "input_dim": 1, "dim": 64},
        },
        "lr": 1e-1,
        "num_epochs": 100,
        "batch_size": 256,
        "n_reps": 64,
        "patience": 10,
        "val_size": 0.1,
        "test_size": 0.2,
        "seed": 42,
        "perturbations": ["color-blur", "color-gray", "gray-noise"],
        "sat_outlier_range_ms": [-100, 100],
        "weights_files": {
            "color-blur": "NRM_sat_blur.pth",
            "color-gray": "NRM_sat_color.pth",
            "gray-noise": "NRM_sat_noise.pth",
        },
    },
    "sc": {
        "model_class": "NRM",
        "n_choices": 2,
        "cond_dim": 32,
        "dt": 0.01,
        "t_max": 3.0,
        "kappa": 100,
        "solver": "euler",
        "use_adjoint": False,
        "encoder_config": {
            "img_embedding": {"type": "raw", "input_dim": 18, "dim": 32},
        },
        "lr": 1e-3,
        "num_epochs": 100,
        "batch_size": 256,
        "n_reps": 64,
        "patience": 10,
        "projection_dim": 18,
        "val_size": 0.1,
        "test_size": 0.2,
        "seed": 42,
        "weights_file": "NRM_sc.pt",
        "embeddings_file": "sc/sc_embeddings.pt",
    },
}


def load_config(config_path=None):
    """Load user config from YAML, falling back to defaults."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def get_training_config(benchmark, user_config=None):
    """Return merged training configuration for a benchmark."""
    defaults = BENCHMARK_DEFAULTS[benchmark].copy()
    if user_config:
        defaults.update(user_config)
    return defaults


def build_model(benchmark, device="cuda", user_config=None):
    """Construct the full NRM for a benchmark: encoder + drift + diffusion + threshold."""
    cfg = get_training_config(benchmark, user_config)

    if cfg["model_class"] == "TimeEmbeddedNRM":
        encoder = TimeEmbeddedEncoder(cfg["encoder_config"])

        drift = TimeEmbeddedDriftNet(
            n_choices=cfg["n_choices"],
            cond_dim=cfg["cond_dim"],
        )
        diffusion = TimeEmbeddedDiffusion(
            n_choices=cfg["n_choices"],
            cond_dim=cfg["cond_dim"],
        )
        threshold = TimeEmbeddedThreshold(
            cond_dim=cfg["cond_dim"],
            t_max=cfg["t_max"],
            dt=cfg["dt"],
            device=device,
        )
        model = TimeEmbeddedNRM(
            drift_net=drift,
            diff_net=diffusion,
            cond_encoder=encoder,
            n_choices=cfg["n_choices"],
            t_max=cfg["t_max"],
            fs=cfg["fs"],
            dt=cfg["dt"],
            threshold_net=threshold,
            kappa=cfg["kappa"],
            solver=cfg["solver"],
            device=device,
            use_adjoint=cfg["use_adjoint"],
        )
    else:
        encoder = ConditioningEncoder(cfg["encoder_config"])

        drift = ConditionedConstantDriftNet(
            n_choices=cfg["n_choices"],
            cond_dim=cfg["cond_dim"],
        )
        diffusion = ConditionedConstantDiffusion(
            n_choices=cfg["n_choices"],
            cond_dim=cfg["cond_dim"],
        )
        threshold = ConditionedConstantThreshold(
            cond_dim=cfg["cond_dim"],
            t_max=cfg["t_max"],
            dt=cfg["dt"],
            device=device,
        )
        model = NRM(
            drift_net=drift,
            diff_net=diffusion,
            cond_encoder=encoder,
            n_choices=cfg["n_choices"],
            t_max=cfg["t_max"],
            dt=cfg["dt"],
            threshold_net=threshold,
            kappa=cfg["kappa"],
            solver=cfg["solver"],
            device=device,
            use_adjoint=cfg["use_adjoint"],
        )

    return model
