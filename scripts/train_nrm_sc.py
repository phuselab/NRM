"""
Train the Neural Race Model on the Scene Categorisation (SC) task.

The model receives DINOv2 embeddings projected to 18 dimensions and learns
to predict binary (natural vs. manmade) choice and RT distributions across
5 discriminability levels.
"""

import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nrm.drifts import ConditionedConstantDriftNet
from nrm.diffusions import ConditionedConstantDiffusion
from nrm.thresholds import ConditionedConstantThreshold
from nrm.conditioning import ConditioningEncoder
from nrm.model import NRM, set_seed
from nrm.train import train_model
from scripts.results_io import refresh_path

from datasets.sc.loader import SCDataset, project_embeddings, sc_collate_fn


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_and_prepare_data(config, projection_dim):
    dataset = SCDataset(
        csv_path=config["csv_path"],
        image_dir=config["image_dir"],
    )

    # The projected splits are cached at embeddings_path; only without them are
    # the 384-d DINOv2 features computed, and those are kept in memory only.
    have_projected = projection_dim > 0 and os.path.exists(config["embeddings_path"])
    if not have_projected:
        dataset.compute_dinov2_embeddings()

    train_ds, val_ds, test_ds = dataset.train_val_test_split(
        val_size=config["val_size"],
        test_size=config["test_size"],
        seed=config["seed"],
    )

    if projection_dim > 0:
        print(f"\nApplying linear projection (n_components={projection_dim})...")
        project_embeddings(
            train_ds, val_ds, test_ds,
            n_components=projection_dim,
            cache_path=config["embeddings_path"],
        )

    print(
        f"Data split sizes — Train: {len(train_ds)}, "
        f"Val: {len(val_ds)}, Test: {len(test_ds)}"
    )
    return train_ds, val_ds, test_ds


def _make_dropout_collate_fn(base_collate_fn, dropout_p):
    if dropout_p <= 0.0:
        return base_collate_fn
    dropout = torch.nn.Dropout(p=dropout_p)
    dropout.train()

    def _collate(batch):
        collated = base_collate_fn(batch)
        if "embeddings" in collated:
            with torch.no_grad():
                collated["embeddings"] = dropout(collated["embeddings"])
        return collated

    return _collate


def create_dataloaders(train_ds, val_ds, batch_size, num_workers,
                       train_collate_fn=None):
    pin_memory = torch.cuda.is_available()
    if train_collate_fn is None:
        train_collate_fn = sc_collate_fn

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=train_collate_fn,
        pin_memory=pin_memory, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=sc_collate_fn,
        pin_memory=pin_memory, persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


def build_sc_model(config, device):
    encoder = ConditioningEncoder(config["encoder_config"])

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
        description="Train NRM on the SC benchmark from scratch. The weights "
                    "are saved as nrm/weights/NRM_sc_REFRESH.pt; the released "
                    "checkpoint is never overwritten. pipelines/sc.ipynb "
                    "evaluates them with REFRESH_WEIGHTS = True.")
    parser.parse_args()

    set_seed(42)

    device = setup_device()

    REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

    NOISE_SIGMA = 0.2
    PROJECTION_DIM = 18
    EMB_DROPOUT = 0.3
    EARLY_STOP = 53

    training_config = {
        "lr": 1e-3,
        "num_epochs": 100,
        "batch_size": 256,
        "n_reps": 64,
        "patience": 10,
        "num_workers": 2,
        "noise_sigma": NOISE_SIGMA,
        "early_stop": EARLY_STOP,
    }

    data_config = {
        "csv_path": os.path.join(REPO_ROOT, "datasets", "sc", "experiment_1.csv"),
        "image_dir": os.path.join(REPO_ROOT, "datasets", "sc", "stimuli"),
        "embeddings_path": os.path.join(REPO_ROOT, "embeddings", "sc",
                                         "sc_embeddings.pt"),
        "val_size": 0.10,
        "test_size": 0.20,
        "seed": 42,
    }

    # Trained weights are saved as NRM_sc_REFRESH.pt, next to the released
    # checkpoint, which is never overwritten.
    WEIGHTS_DIR = os.path.join(REPO_ROOT, "nrm", "weights")
    WEIGHT_PATH = str(refresh_path(os.path.join(WEIGHTS_DIR, "NRM_sc.pt"), True))

    print("\n" + "=" * 80)
    print("LOADING SC DATASET")
    print("=" * 80)

    train_ds, val_ds, test_ds = load_and_prepare_data(data_config, PROJECTION_DIM)

    embedding_dim = train_ds.embeddings.shape[1]
    print(f"Embedding dimension: {embedding_dim}")

    train_collate = _make_dropout_collate_fn(sc_collate_fn, EMB_DROPOUT)
    train_loader, val_loader = create_dataloaders(
        train_ds, val_ds,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        train_collate_fn=train_collate,
    )

    model_config = {
        "n_choices": 2,
        "dt": 0.01,
        "t_max": 3.0,
        "kappa": 100,
        "solver": "euler",
        "cond_dim": 32,
        "encoder_config": {
            "img_embedding": {"type": "raw", "input_dim": embedding_dim, "dim": 32},
        },
    }

    model = build_sc_model(model_config, device)
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
