"""
Train the Neural Race Model on the SAT (Speed-Accuracy Trade-off) benchmark.

The model receives DINOv2 embeddings (384-d) and a time-limit signal,
conditioned on 16 object categories with 5 response deadlines.
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nrm.drifts import ConditionedConstantDriftNet
from nrm.diffusions import ConditionedConstantDiffusion
from nrm.thresholds import ConditionedConstantThreshold
from nrm.conditioning import ConditioningEncoder
from nrm.model import NRM, set_seed
from nrm.train import train_model
from scripts.results_io import refresh_path

from datasets.sat.loader import SATDataset, sat_collate_fn


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_and_prepare_data(config):
    dataset = SATDataset(
        file_path=config["file_path"],
        image_dir=config["image_dir"],
        # NRM reads only the DINOv2 embeddings; the images are needed only to
        # compute them.
        load_images=False,
    )

    dataset.compute_dinov2_embeddings(save_path=config["embeddings_path"])

    if config.get("use_sat_outlier_removal", True):
        dataset.remove_rt_outliers(
            outlier_range_ms=config.get("sat_outlier_range_ms", [-100, 100])
        )
    else:
        dataset.remove_rt_outliers_mad()

    exp_mask = dataset.df["block"] != "cattut"
    n_cattut = int((~exp_mask).sum())
    if n_cattut:
        if hasattr(dataset, "embeddings") and dataset.embeddings is not None:
            mask_tensor = torch.tensor(exp_mask.values, dtype=torch.bool)
            dataset.embeddings = dataset.embeddings[mask_tensor]
        dataset.df = dataset.df[exp_mask].reset_index(drop=True)
        print(f"  Removed {n_cattut} 'cattut' trials "
              f"({len(dataset.df)} experimental trials remaining).")

    train_ds, val_ds, test_ds = dataset.train_val_test_split(
        val_size=config["val_size"],
        test_size=config["test_size"],
        seed=config["seed"],
        shuffle=True,
    )

    print(f"Data split sizes — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_ds, val_ds


def create_dataloaders(train_ds, val_ds, batch_size, num_workers):
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=sat_collate_fn,
        pin_memory=pin_memory, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=sat_collate_fn,
        pin_memory=pin_memory, persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


def build_sat_model(config, device):
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
        description="Train NRM on one SAT condition from scratch. The weights "
                    "are saved as nrm/weights/NRM_sat_<condition>_REFRESH.pth; "
                    "the released checkpoint is never overwritten. "
                    "pipelines/sat.ipynb evaluates them with "
                    "REFRESH_WEIGHTS = True.")
    parser.add_argument("--perturbation", type=str, default="color-blur",
                        choices=["color-blur", "color-gray", "gray-noise"])
    args = parser.parse_args()

    set_seed(42)

    device = setup_device()

    REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

    PERT_TO_WEIGHT = {
        "color-blur": "NRM_sat_blur.pth",
        "color-gray": "NRM_sat_color.pth",
        "gray-noise": "NRM_sat_noise.pth",
    }

    pname = args.perturbation.replace("-", "_")

    model_config = {
        "n_choices": 16,
        "dt": 0.01,
        "t_max": 5.0,
        "kappa": 100,
        "solver": "euler",
        "cond_dim": 128,
        "encoder_config": {
            "img_embedding": {"type": "raw", "input_dim": 384, "dim": 64},
            "time_limit": {"type": "raw", "input_dim": 1, "dim": 64},
        },
    }

    training_config = {
        "lr": 1e-1,
        "num_epochs": 100,
        "batch_size": 256,
        "n_reps": 64,
        "patience": 10,
        "num_workers": 2,
    }

    data_config = {
        "file_path": os.path.join(REPO_ROOT, "datasets", "sat", f"{args.perturbation}.txt"),
        "image_dir": os.path.join(REPO_ROOT, "datasets", "sat", "data"),
        "embeddings_path": os.path.join(REPO_ROOT, "embeddings", "sat",
                                         f"sat_{pname}_embeddings.pt"),
        "val_size": 0.1,
        "test_size": 0.2,
        "seed": 42,
        "use_sat_outlier_removal": True,
        "sat_outlier_range_ms": [-100, 100],
    }

    # Trained weights are saved as NRM_sat_<condition>_REFRESH.pth, next to the
    # released checkpoint, which is never overwritten.
    WEIGHTS_DIR = os.path.join(REPO_ROOT, "nrm", "weights")
    WEIGHT_PATH = str(refresh_path(os.path.join(WEIGHTS_DIR, PERT_TO_WEIGHT[args.perturbation]),
                                   True))

    print("\n" + "=" * 80)
    print(f"LOADING SAT DATASET ({args.perturbation})")
    print("=" * 80)

    train_ds, val_ds = load_and_prepare_data(data_config)
    train_loader, val_loader = create_dataloaders(
        train_ds, val_ds,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
    )

    model = build_sat_model(model_config, device)
    optimizer = optim.AdamW(model.parameters(), lr=training_config["lr"])

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
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
    )


if __name__ == "__main__":
    main()
