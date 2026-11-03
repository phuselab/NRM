"""DINOv2 stimulus embeddings for the SAT benchmark (§2.2)."""

import argparse
import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.normpath(_REPO_ROOT))

import torch  # noqa: E402

from datasets import display_path  # noqa: E402
from datasets.sat.loader import SATDataset  # noqa: E402

PERTURBATIONS = ["color-blur", "color-gray", "gray-noise"]


def output_path(perturbation, out_dir):
    # Behavioural files use hyphens, embedding files use underscores.
    return os.path.join(out_dir, f"sat_{perturbation.replace('-', '_')}_embeddings.pt")


def compute(perturbation, data_dir, out_dir, batch_size, overwrite=False):
    out = output_path(perturbation, out_dir)
    if os.path.exists(out) and not overwrite:
        print(f"{os.path.basename(out)} exists — skipping (use --overwrite)")
        return out

    ds = SATDataset(
        file_path=os.path.join(data_dir, f"{perturbation}.txt"),
        image_dir=os.path.join(data_dir, "data"),
        load_images=True,
    )
    print(f"{perturbation}: {len(ds.df)} trials, "
          f"{ds.df['image_index'].nunique()} unique stimuli")

    # Writes {'embeddings', 'dim', 'image_paths', 'stimuli_digest'} to `out`.
    # image_paths is the alignment key the loader uses when a later filter
    # changes the row count; stimuli_digest hashes the stimulus bytes so a
    # cache written against different images is recomputed, not returned.
    # The method loads an existing file on its own, so `overwrite` is passed
    # through as well.
    emb = ds.compute_dinov2_embeddings(save_path=out, batch_size=batch_size,
                                       overwrite=overwrite)
    print(f"  -> {display_path(out)}  shape={tuple(emb.shape)}")
    return out


def main():
    default_data = os.path.normpath(os.path.join(_REPO_ROOT, "datasets", "sat"))
    default_out = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--perturbation", choices=PERTURBATIONS,
                   help="which condition to embed")
    p.add_argument("--all", action="store_true", help="embed all three")
    p.add_argument("--data_dir", default=default_data)
    p.add_argument("--output_dir", default=default_out)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even if the output exists")
    args = p.parse_args()

    if not args.perturbation and not args.all:
        p.error("pass --perturbation <name> or --all")

    targets = PERTURBATIONS if args.all else [args.perturbation]
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    for t in targets:
        compute(t, args.data_dir, args.output_dir, args.batch_size, args.overwrite)


if __name__ == "__main__":
    main()
