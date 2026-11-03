"""SC dataset: loader, DINOv2 embedding computation and train/val/test split (§3)."""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

from datasets import display_path


def _parse_prob(cat: str) -> float:
    parts = cat.split("_")
    category = parts[0]
    number = int(parts[1])
    if category == "natural":
        return number / 100.0
    else:
        return (100 - number) / 100.0


def _parse_choice(cat: str) -> int:
    return 0 if cat.startswith("natural") else 1


class SCDataset(Dataset):

    def __init__(self, csv_path: str, image_dir: str, load_images: bool = False):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.load_images = load_images
        self.df = self._load_and_preprocess()

    def _load_and_preprocess(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        before = len(df)
        df = df[df["bad_trial"] == 0].reset_index(drop=True)
        print(f"bad_trial removal: {before - len(df)} rows removed, {len(df)} remaining.")

        df["choice"] = df["cat"].apply(_parse_choice)
        df["prob"] = df["cat"].apply(_parse_prob)
        df["human_choice"] = (
            pd.to_numeric(df["response"], errors="coerce").fillna(0).astype(int)
        )
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0)
        df["rt"] = pd.to_numeric(df["rt"], errors="coerce")

        unique_imgs = sorted(df["img"].unique())
        img_to_idx = {img: i for i, img in enumerate(unique_imgs)}
        df["image_index"] = df["img"].map(img_to_idx)

        print(
            f"Dataset ready: {len(df)} trials, "
            f"{df['img'].nunique()} unique images, "
            f"{df['subj_idx'].nunique()} subjects."
        )
        return df

    @torch.no_grad()
    def compute_dinov2_embeddings(self, save_path: str | None = None,
                                  batch_size: int = 32):
        """384-d DINOv2 features for every trial; cached at ``save_path`` if given."""
        if save_path is not None and os.path.exists(save_path):
            print(f"Loading DINOv2 embeddings from {display_path(save_path)}...")
            data = torch.load(save_path, weights_only=False)
            saved_embeddings = data["embeddings"]
            if "image_paths" in data and len(saved_embeddings) != len(self.df):
                saved_paths = data["image_paths"]
                path_to_idx = {p: i for i, p in enumerate(saved_paths)}
                current_paths = self._get_full_image_paths()
                indices = [path_to_idx[p] for p in current_paths if p in path_to_idx]
                if len(indices) == len(current_paths):
                    self.embeddings = saved_embeddings[indices]
                    print(f"Embeddings aligned: {len(indices)} of {len(saved_embeddings)} rows.")
                else:
                    print(f"Warning: {len(current_paths) - len(indices)} paths not found. Recomputing...")
            else:
                self.embeddings = saved_embeddings
            if hasattr(self, "embeddings"):
                return self.embeddings

        print("Computing DINOv2 embeddings...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        model = AutoModel.from_pretrained("facebook/dinov2-small")
        model.eval().to(device)

        embeddings = []
        paths = self._get_full_image_paths()

        for i in tqdm(range(0, len(paths), batch_size)):
            batch_paths = paths[i:i + batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            pooled = patch_tokens.mean(dim=1)
            embeddings.append(pooled.cpu())

        self.embeddings = torch.cat(embeddings, dim=0)

        if save_path is not None:
            print(f"Saving embeddings to {display_path(save_path)}...")
            torch.save(
                {"embeddings": self.embeddings, "dim": self.embeddings.shape[1],
                 "image_paths": paths},
                save_path,
            )
        return self.embeddings

    def _get_full_image_paths(self) -> list[str]:
        return [os.path.join(self.image_dir, img) for img in self.df["img"].tolist()]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        sample = {
            "image_path": os.path.join(self.image_dir, str(row["img"])),
            "subj_idx": int(row["subj_idx"]),
            "img": str(row["img"]),
            "cat": str(row["cat"]),
            "human_choice": torch.tensor(int(row["human_choice"]), dtype=torch.long),
            "choice": torch.tensor(int(row["choice"]), dtype=torch.long),
            "rt": torch.tensor(
                float(row["rt"]) if pd.notna(row["rt"]) else 0.0,
                dtype=torch.float32,
            ),
            "image_index": torch.tensor(int(row["image_index"]), dtype=torch.long),
            "correct": torch.tensor(float(row["correct"]), dtype=torch.float32),
            "prob": float(row["prob"]),
        }

        if hasattr(self, "embeddings"):
            sample["embedding"] = self.embeddings[idx]

        return sample

    def _shallow_copy(self):
        new_ds = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            setattr(new_ds, k, v)
        return new_ds

    def train_val_test_split(self, val_size: float = 0.1, test_size: float = 0.2,
                             seed: int = 42):
        assert val_size + test_size < 1.0

        unique_imgs = np.array(sorted(self.df["img"].unique()))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_imgs)

        n_total = len(unique_imgs)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)

        test_imgs = set(unique_imgs[:n_test])
        val_imgs = set(unique_imgs[n_test:n_test + n_val])
        train_imgs = set(unique_imgs[n_test + n_val:])

        datasets = {}
        for name, img_set in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            mask = self.df["img"].isin(img_set).values
            idxs = np.where(mask)[0]
            ds = self._shallow_copy()
            ds.df = self.df.iloc[idxs].reset_index(drop=True)
            if hasattr(self, "embeddings"):
                ds.embeddings = self.embeddings[idxs]
            datasets[name] = ds
            print(f"  {name:5s}: {len(idxs)} trials ({len(img_set)} images)")

        return datasets["train"], datasets["val"], datasets["test"]

    def shuffle(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        idxs = rng.permutation(len(self.df))
        self.df = self.df.iloc[idxs].reset_index(drop=True)
        if hasattr(self, "embeddings"):
            self.embeddings = self.embeddings[idxs]


def sc_collate_fn(batch: list[dict]) -> dict:
    has_embeddings = "embedding" in batch[0]

    collated = {
        "rts": torch.stack([b["rt"] for b in batch]),
        "choices": torch.stack([b["choice"] for b in batch]),
        "human_choices": torch.stack([b["human_choice"] for b in batch]),
        "image_index": torch.stack([b["image_index"] for b in batch]),
        "corrects": torch.stack([b["correct"] for b in batch]),
        "probs": [b["prob"] for b in batch],
        "levels": [f"{b['prob']:.1f}" for b in batch],
        "cats": [b["cat"] for b in batch],
        "imgs": [b["img"] for b in batch],
        "subj_idxs": [b["subj_idx"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
    }

    if has_embeddings:
        collated["embeddings"] = torch.stack([b["embedding"] for b in batch])

    return collated


def project_embeddings(train_ds, val_ds, test_ds, n_components: int,
                       emb_type: str = "dinov2", cache_dir: str = ".",
                       cache_path: str | None = None):
    """
    Linear projection of embeddings to n_components dimensions.
    Fit on training split only; val and test are only transformed.
    Mutates .embeddings in-place on all three datasets.

    If ``cache_path`` (default ``cache_dir/embeddings_<emb_type>_<n>d.pt``)
    exists, the projected splits are loaded from it and nothing is fitted.
    """
    if cache_path is None:
        cache_path = os.path.join(cache_dir, f"embeddings_{emb_type}_{n_components}d.pt")

    if os.path.exists(cache_path):
        print(f"Loading projected embeddings from {display_path(cache_path)}...")
        data = torch.load(cache_path, weights_only=False)
        if data.get("n_components", n_components) != n_components:
            raise ValueError(
                f"{display_path(cache_path)} holds {data['n_components']}-d "
                f"embeddings, not {n_components}-d: delete it to recompute.")
        for split, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            if len(data[split]) != len(ds):
                raise ValueError(
                    f"{display_path(cache_path)}: {len(data[split])} {split} "
                    f"embeddings vs {len(ds)} trials: delete it to recompute.")
        train_ds.embeddings = data["train"]
        val_ds.embeddings = data["val"]
        test_ds.embeddings = data["test"]
        print(f"  Projected embeddings loaded (dim={train_ds.embeddings.shape[1]}).")
        return

    from embeddings.sc.compute_embeddings import project_embeddings as _svd_project

    print(f"Fitting linear projection (n_components={n_components}) on train split "
          f"({len(train_ds)} rows, input dim={train_ds.embeddings.shape[1]})...")

    train_indices = list(range(len(train_ds)))
    all_emb = torch.cat([train_ds.embeddings, val_ds.embeddings, test_ds.embeddings])
    n_train = len(train_ds)
    n_val = len(val_ds)

    projected, mean, components = _svd_project(
        all_emb, n_components=n_components, train_indices=train_indices)

    train_ds.embeddings = projected[:n_train]
    val_ds.embeddings = projected[n_train:n_train + n_val]
    test_ds.embeddings = projected[n_train + n_val:]

    print(f"Saving projected embeddings to {display_path(cache_path)}...")
    # The fitted basis travels with the splits so the same projection can be
    # reapplied to new images without refitting (and without re-deriving it
    # from a different split, which would change it).
    torch.save(
        {"train": train_ds.embeddings, "val": val_ds.embeddings,
         "test": test_ds.embeddings, "n_components": n_components,
         "mean": torch.from_numpy(mean), "components": torch.from_numpy(components)},
        cache_path,
    )
    print(f"  Projection complete (dim {all_emb.shape[1]} -> {n_components}).")
