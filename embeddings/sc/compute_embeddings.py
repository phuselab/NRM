"""DINOv2 embeddings + linear dimensionality reduction for the SC benchmark (§2.2)."""

import argparse
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.normpath(_REPO_ROOT))

from datasets import display_path  # noqa: E402


def project_embeddings(embeddings, n_components=18, train_indices=None):
    """Linear projection to ``n_components``.

    Fits on ``train_indices`` only (all rows if None) and transforms every row.

    Returns
    -------
    projected : torch.Tensor  (N, n_components)
    mean      : np.ndarray    (D,)              the fitted centring
    components: np.ndarray    (n_components, D) the fitted basis

    """
    X = embeddings.numpy().astype(np.float64) if hasattr(embeddings, "numpy") \
        else np.asarray(embeddings, dtype=np.float64)
    X_train = X[train_indices] if train_indices is not None else X

    mu = X_train.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_train - mu, full_matrices=False)
    components = Vt[:n_components].astype(np.float32)
    mu_f32 = mu.astype(np.float32)

    X_proj = (X.astype(np.float32) - mu_f32) @ components.T
    return torch.from_numpy(X_proj), mu_f32, components


def main():
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "sc_embeddings.pt")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv_path",
                   default=os.path.normpath(os.path.join(
                       _REPO_ROOT, "datasets", "sc", "experiment_1.csv")))
    p.add_argument("--image_dir",
                   default=os.path.normpath(os.path.join(
                       _REPO_ROOT, "datasets", "sc", "stimuli")))
    p.add_argument("--output", default=default_out)
    p.add_argument("--raw_cache", default=None,
                   help="optionally cache the un-projected 384-d features here "
                        "(not saved by default)")
    p.add_argument("--n_components", type=int, default=18)
    p.add_argument("--val_size", type=float, default=0.10)
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print(f"{os.path.basename(args.output)} exists — "
              "nothing to do (use --overwrite)")
        return

    from datasets.sc.loader import SCDataset

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    ds = SCDataset(csv_path=args.csv_path, image_dir=args.image_dir,
                   load_images=True)

    # Stage 1 — 384-d features for every trial, via the canonical loader path.
    ds.compute_dinov2_embeddings(save_path=args.raw_cache,
                                 batch_size=args.batch_size)

    # Split first: the projection must not see val or test.
    train_ds, val_ds, test_ds = ds.train_val_test_split(
        val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    n_train, n_val = len(train_ds), len(val_ds)

    all_emb = torch.cat([train_ds.embeddings, val_ds.embeddings,
                         test_ds.embeddings])
    print(f"Fitting projection {all_emb.shape[1]} -> {args.n_components} "
          f"on {n_train} training rows")

    # Stage 2 — fit on the leading n_train rows, which are the train split.
    projected, mean, components = project_embeddings(
        all_emb, n_components=args.n_components,
        train_indices=list(range(n_train)))

    payload = {
        "train": projected[:n_train],
        "val": projected[n_train:n_train + n_val],
        "test": projected[n_train + n_val:],
        "n_components": args.n_components,
        "mean": torch.from_numpy(mean),
        "components": torch.from_numpy(components),
    }
    torch.save(payload, args.output)
    print(f"Saved {display_path(args.output)}")
    for split in ("train", "val", "test"):
        print(f"  {split:5s} {tuple(payload[split].shape)}")


if __name__ == "__main__":
    main()
