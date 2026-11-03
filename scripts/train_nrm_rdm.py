"""
Train the Neural Race Model on the Random Dot Motion (RDM) task.

The model receives per-frame motion energy embeddings (Gabor filters,
Adelson & Bergen 1985) and learns to predict choice and RT distributions
across 7 coherence levels.
"""

import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nrm.drifts import ConditionedConstantDriftNet, TimeEmbeddedDriftNet
from nrm.diffusions import ConditionedConstantDiffusion, TimeEmbeddedDiffusion
from nrm.thresholds import ConditionedConstantThreshold, TimeEmbeddedThreshold
from nrm.conditioning import TemporalFlatEncoder, TimeEmbeddedEncoder
from nrm.model import NRM, TimeEmbeddedNRM, set_seed
from nrm.train import train_model
from scripts.results_io import display_path, refresh_path

from datasets.rdm.loader import (
    RDMDataset,
    build_rdm_nrm_dataset,
    load_rdm_nrm_dataset,
    rdm_collate_fn,
    _stratified_split,
    PKL_OUT,
    ME_EMBS_OUT,
)


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_and_prepare_data(config):
    embeddings_path = config.get("embeddings_path", None)

    if PKL_OUT.exists() and not config.get("force_rebuild", False):
        dataset = load_rdm_nrm_dataset(
            pkl_out=PKL_OUT,
            embeddings_path=embeddings_path,
        )
    else:
        _csv = config.get("csv_path") or None
        _build_kwargs = dict(seed=config["seed"])
        if _csv is not None:
            _build_kwargs["csv_path"] = _csv
        dataset = build_rdm_nrm_dataset(**_build_kwargs)
        if embeddings_path is not None:
            data = torch.load(embeddings_path, weights_only=False)
            embs = data["embeddings"]
            if "video_indices" in data:
                idx_map = {int(v): i for i, v in enumerate(data["video_indices"])}
                order = [idx_map[int(vi)] for vi in dataset.df["video_index"]]
                embs = embs[order]
            dataset.embeddings = embs

    masks = _stratified_split(dataset.df, val_size=config["val_size"],
                              test_size=config["test_size"], seed=config["seed"])
    dataset.df["split"] = np.select([masks["test"], masks["val"]],
                                    ["test", "val"], "train")

    dataset.remove_rt_outliers_mad(
        mad_thresh=config.get("mad_thresh", 3.0),
        min_rt=config.get("min_rt", 0.05),
        max_rt=config.get("max_rt", 3.0),
    )

    train_ds, val_ds, _ = dataset.train_val_test_split(
        val_size=config["val_size"],
        test_size=config["test_size"],
        seed=config["seed"],
    )

    print(f"Data split sizes — Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds


def create_dataloaders(train_ds, val_ds, batch_size, num_workers):
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rdm_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rdm_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


def ensure_me_embeddings(embeddings_path, me_config, device):
    from pathlib import Path

    emb_path = Path(embeddings_path)
    if emb_path.exists():
        print(f"Motion energy embeddings found: {display_path(emb_path)}")
        return

    print("\n" + "=" * 80)
    print("COMPUTING MOTION ENERGY EMBEDDINGS")
    print("=" * 80)

    full_ds = load_rdm_nrm_dataset(pkl_out=PKL_OUT, embeddings_path=None)
    full_ds.compute_motion_energy_embeddings(
        save_path=emb_path,
        device=str(device),
        n_directions=me_config["n_directions"],
        speeds=me_config["speeds"],
        lam=me_config["lam"],
        sigma_xy=me_config["sigma_xy"],
        sigma_t=me_config["sigma_t"],
        kx=me_config["kx"],
        ky=me_config["ky"],
        kt=me_config["kt"],
    )


def build_rdm_model(config, device):
    if config["type"] == "explicit_time":
        encoder = TimeEmbeddedEncoder(config["encoder_config"])
        drift = TimeEmbeddedDriftNet(
            n_choices=config["n_choices"],
            cond_dim=config["cond_dim"],
        )
        diffusion = TimeEmbeddedDiffusion(
            n_choices=config["n_choices"],
            cond_dim=config["cond_dim"],
        )
        threshold = TimeEmbeddedThreshold(
            cond_dim=config["cond_dim"],
            t_max=config["t_max"],
            dt=config["dt"],
            device=device,
        )
        model = TimeEmbeddedNRM(
            drift_net=drift,
            diff_net=diffusion,
            cond_encoder=encoder,
            n_choices=config["n_choices"],
            t_max=config["t_max"],
            fs=config["fs"],
            dt=config["dt"],
            threshold_net=threshold,
            kappa=config["kappa"],
            solver=config["solver"],
            device=device,
            use_adjoint=False,
        )
    elif config["type"] == "avg":
        encoder = TemporalFlatEncoder(config["encoder_config"])
        drift = ConditionedConstantDriftNet(
            n_choices=config["n_choices"],
            cond_dim=config["cond_dim"],
        )
        diffusion = ConditionedConstantDiffusion(
            n_choices=config["n_choices"],
            cond_dim=config["cond_dim"],
        )
        threshold = ConditionedConstantThreshold(
            t_max=config["t_max"],
            dt=config["dt"],
            cond_dim=config["cond_dim"],
            device=device,
        )
        model = NRM(
            drift_net=drift,
            diff_net=diffusion,
            cond_encoder=encoder,
            n_choices=config["n_choices"],
            t_max=config["t_max"],
            dt=config["dt"],
            threshold_net=threshold,
            kappa=config["kappa"],
            solver=config["solver"],
            device=device,
            use_adjoint=False,
        )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train NRM on the RDM benchmark from scratch. The weights "
                    "are saved as nrm/weights/NRM_rdm_REFRESH.pth; the "
                    "released checkpoint is never overwritten. "
                    "pipelines/rdm.ipynb evaluates them with "
                    "REFRESH_WEIGHTS = True.")
    parser.parse_args()

    set_seed(42)

    device = setup_device()

    TYPE = "explicit_time"

    # Motion energy filter parameters (Gabor bank)
    ME_N_DIR    = 8
    ME_SPEEDS   = [0.5, 1.0, 2.0]
    ME_LAM      = 8.0
    ME_SIGMA_XY = 4.0
    ME_SIGMA_T  = 3.0
    NOISE_SIGMA = 1.5
    ME_KX       = 13
    ME_KY       = 13
    ME_KT       = 7

    EMBED_DIM = ME_N_DIR * len(ME_SPEEDS)

    _is_default = (
        ME_N_DIR == 8 and ME_SPEEDS == [0.5, 1.0, 2.0]
        and ME_LAM == 8.0 and ME_SIGMA_XY == 4.0 and ME_SIGMA_T == 3.0
        and ME_KX == 13 and ME_KY == 13 and ME_KT == 7
    )
    if _is_default:
        EMBEDDINGS_PATH = str(ME_EMBS_OUT)
        ME_TAG = ""
    else:
        _sp = "-".join(f"{s:g}" for s in ME_SPEEDS)
        ME_TAG = (
            f"_nd{ME_N_DIR}_sp{_sp}"
            f"_lam{ME_LAM:g}_sxy{ME_SIGMA_XY:g}_st{ME_SIGMA_T:g}"
            f"_k{ME_KX}x{ME_KY}x{ME_KT}"
        )
        EMBEDDINGS_PATH = str(ME_EMBS_OUT.parent / f"rdm_me_embeddings{ME_TAG}.pt")

    if TYPE == "avg":
        N_T = 150
        encoder_config = {
            "img_embedding": {"type": "raw", "input_dim": N_T * EMBED_DIM, "dim": 64},
        }
        cond_dim = 64
    elif TYPE == "explicit_time":
        encoder_config = {
            "img_embedding": {"type": "raw", "input_dim": EMBED_DIM, "dim": 32},
        }
        cond_dim = 32

    model_config = {
        "n_choices":      2,
        "dt":             0.01,
        "t_max":          3.0,
        "fs":             75,
        "kappa":          100,
        "solver":         "euler",
        "type":           TYPE,
        "cond_dim":       cond_dim,
        "encoder_config": encoder_config,
    }

    training_config = {
        "lr":          1e-3,
        "num_epochs":  100,
        "batch_size":  256,
        "n_reps":      64,
        "patience":    10,
        "num_workers": 2,
        "noise_sigma": NOISE_SIGMA,
        "early_stop":  53,
    }

    data_config = {
        "val_size":        0.10,
        "test_size":       0.20,
        "seed":            42,
        "mad_thresh":      3.0,
        "min_rt":          0.05,
        "max_rt":          3.0,
        "force_rebuild":   False,
        "embeddings_path": EMBEDDINGS_PATH,
    }

    WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "nrm", "weights")
    WEIGHT_PATH = str(refresh_path(os.path.join(WEIGHTS_DIR, "NRM_rdm.pth"), True))

    me_filter_config = {
        "n_directions": ME_N_DIR,
        "speeds":       ME_SPEEDS,
        "lam":          ME_LAM,
        "sigma_xy":     ME_SIGMA_XY,
        "sigma_t":      ME_SIGMA_T,
        "kx":           ME_KX,
        "ky":           ME_KY,
        "kt":           ME_KT,
    }
    ensure_me_embeddings(EMBEDDINGS_PATH, me_filter_config, device)

    print("\n" + "=" * 80)
    print("LOADING RDM DATASET")
    print("=" * 80)

    train_ds, val_ds = load_and_prepare_data(data_config)
    train_loader, val_loader = create_dataloaders(
        train_ds, val_ds,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
    )

    model = build_rdm_model(model_config, device)
    optimizer = optim.AdamW(model.parameters(), lr=training_config["lr"])

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    np.random.seed(42)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=training_config["num_epochs"],
        save_path=WEIGHT_PATH,
        n_reps=training_config["n_reps"],
        patience=training_config["patience"],
        device=device,
        noise_sigma=training_config["noise_sigma"],
        early_stop=training_config["early_stop"],
    )


if __name__ == "__main__":
    main()
