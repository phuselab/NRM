"""Motion energy embeddings for RDM videos (Adelson & Bergen 1985, §2.2)."""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from datasets import display_path
from datasets.rdm.motion_energy import MotionEnergyEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory with .npy video files")
    parser.add_argument("--output", type=str, default="rdm_embeddings.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    embedder = MotionEnergyEmbedder(device=args.device)

    video_files = sorted([
        f for f in os.listdir(args.video_dir) if f.endswith(".npy")
    ])
    print(f"Found {len(video_files)} videos")

    all_embeddings = []
    video_indices = []

    for fname in tqdm(video_files, desc="Embedding"):
        video = np.load(os.path.join(args.video_dir, fname))
        emb = embedder.embed_video(video)
        all_embeddings.append(emb)
        idx = int(os.path.splitext(fname)[0].split("_")[-1])
        video_indices.append(idx)

    embeddings = torch.stack(all_embeddings)
    torch.save({
        "embeddings": embeddings,
        "video_indices": torch.tensor(video_indices),
    }, args.output)
    print(f"Saved {embeddings.shape} to {display_path(args.output)}")


if __name__ == "__main__":
    main()
