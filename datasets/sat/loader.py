"""SAT dataset: loader, outlier removal and train/val/test split (§3)."""

import hashlib
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

from datasets import display_path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ============================================================
# CONSTANTS
# ============================================================

CATEGORIES = [
    "airplane", "bear", "bicycle", "bird", "boat", "bottle",
    "car", "cat", "chair", "clock", "dog", "elephant",
    "keyboard", "knife", "oven", "truck"
]

BLOCKS = ["cattut", "0", "400", "600", "800", "1000"]
TIMES  = [500, 900, 1100, 1300, 1500]

BLOCK_TO_FOLDER = {
    "cattut": "train",
    "0":    "0",
    "400":  "1",
    "600":  "2",
    "800":  "3",
    "1000": "4",
}

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============================================================
# DATASET
# ============================================================

class SATDataset(Dataset):
    """
    PyTorch Dataset for the SAT experiment (Subramanian et al., 2025).
    """

    def __init__(
        self,
        file_path: str,
        image_dir: str,
        transform=None,
        throw_out: set | None = None,
        load_images: bool = True,
    ):
        self.file_path   = file_path
        self.image_dir   = image_dir
        self.transform   = transform if transform is not None else DEFAULT_TRANSFORM
        self.throw_out   = set(str(s) for s in (throw_out or {"2580", "2581", "2619", "2615", "2591"}))
        self.load_images = load_images

        self.cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}

        self.df = self._load_and_preprocess()

        if self.load_images:
            before = len(self.df)
            self.df = self.df[
                self.df["image_path"].apply(
                    lambda p: bool(p) and os.path.exists(p)
                )
            ].reset_index(drop=True)
            removed = before - len(self.df)
            if before and not len(self.df):
                raise FileNotFoundError(
                    f"none of the {before} SAT stimulus images is under {image_dir}. "
                    "Build them with `python fetch_assets.py sat-images` "
                    "(needs ImageNet in datasets/sat/ILSVRC/), or point "
                    "SAT_IMAGE_DIR at an existing copy.")
            if removed:
                print(f"Image pre-filter: {removed} trials removed (missing file), "
                      f"{len(self.df)} remaining.")

    # ------------------------------------------------------------------
    # LOAD & PREPROCESS
    # ------------------------------------------------------------------

    def _extract_ground_truth(self, path, cat_to_idx):
            if not path or not isinstance(path, str):
                return -1
            cat_name = os.path.basename(path).replace('.JPEG', '').split('_')[-1]
            return cat_to_idx.get(cat_name, -1)

    def _extract_img_index(self, path):
        if not path or not isinstance(path, str):
            return -1
        fname = os.path.basename(path)
        try:
            return int(fname.split('_')[0])
        except (ValueError, IndexError):
            return -1

    def _load_and_preprocess(self) -> pd.DataFrame:
        raw_data = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    session = json.loads(line)
                    raw_data.append(
                        pd.DataFrame(session) if isinstance(session, list)
                        else pd.DataFrame([session])
                    )
                except json.JSONDecodeError:
                    continue

        print(f"File read: {len(raw_data)} subjects found.")
        all_trials = []

        for df in raw_data:
            srid = str(df["url"].iloc[0]["srid"]) if "url" in df.columns else "unknown"
            if srid in self.throw_out:
                continue

            order = df.iloc[500]["order"] if len(df) > 500 and "order" in df.columns else np.nan

            def get_start_idx(blockname):
                match = df.loc[df["sender"] == "Screen" + blockname]
                return match.index[0] if len(match) > 0 else None

            start_indices = {b: get_start_idx("CategorizationTutorial" if b == "cattut" else b)
                             for b in BLOCKS}

            for j, block in enumerate(BLOCKS):
                start_idx = start_indices[block]
                if start_idx is None:
                    continue

                n_samples = 50 * 3 if block == "cattut" else 210 * 3
                n_skip    = 0      if block == "cattut" else 10

                window   = df.iloc[start_idx: start_idx + n_samples + 1]
                block_df = window[window["sender"] == "Stimulus & Response"].iloc[n_skip:].copy()

                if block != "cattut":
                    beep_time = TIMES[j - 1]
                    block_df["time_limit"] = beep_time
                else:
                    block_df["time_limit"] = -1

                block_df["srid"]   = srid
                block_df["block"]  = block
                block_df["order"]  = order

                folder = BLOCK_TO_FOLDER[block]
                block_df["image_path"] = block_df["filename"].apply(
                    lambda fn: os.path.join(self.image_dir, folder, str(fn))
                    if pd.notna(fn) else None
                )

                all_trials.append(block_df)


        final_df = pd.concat(all_trials, ignore_index=True)

        final_df["choice"] = final_df["image_path"].apply(
            lambda x: self._extract_ground_truth(x, self.cat_to_idx)
        )

        final_df["image_index"] = final_df["image_path"].apply(self._extract_img_index)

        nan_mask = final_df["response"].isna()
        final_df.loc[nan_mask, "response"] = np.random.choice(CATEGORIES, size=nan_mask.sum())

        final_df["level_str"] = final_df["mode"].astype(str).str.strip()
        final_df["mode"]    = pd.to_numeric(final_df["mode"],    errors="coerce")
        final_df["correct"] = pd.to_numeric(final_df["correct"], errors="coerce").fillna(0)
        final_df["duration"]= pd.to_numeric(final_df["duration"],errors="coerce")

        print(f"Dataset ready: {len(final_df)} total trials.")
        return final_df

    # ------------------------------------------------------------------
    # DINOV2 EMBEDDINGS
    # ------------------------------------------------------------------

    def stimuli_digest(self):
        """MD5 over the *bytes* of every distinct stimulus in ``self.df``.

        A cache keyed only on path names cannot tell two generations of the
        same stimulus apart. ``datasets/sat/data`` shipped a mixture of two:
        for ``_noise_*`` a different draw of the Gaussian noise, for
        ``_blur_0_``/``_blur_3_`` a different image altogether. The embedding
        caches were loaded against the corrected images without complaint and
        the SAT numbers moved by 5 RMSE points with nothing to point at. The
        47,250 trials of a condition cover only ~600 distinct files, so
        hashing them all costs well under a second.

        Returns None when none of the stimuli is on disk.
        """
        paths = sorted(set(self.df["image_path"].tolist()))
        if not any(p and os.path.exists(p) for p in paths):
            return None
        h = hashlib.md5()
        for p in paths:
            h.update(os.path.basename(str(p)).encode())
            try:
                with open(p, "rb") as f:
                    h.update(f.read())
            except OSError:
                h.update(b"<missing>")
        return h.hexdigest()

    @torch.no_grad()
    def compute_dinov2_embeddings(self, save_path="embeddings.pt", batch_size=32,
                                  overwrite=False):
        """
        Compute DINOv2 embeddings for images in self.df.

        Loads from disk if the file already exists, *unless* ``overwrite`` is
        set or the cached stimulus digest does not match the images on disk
        (see :meth:`stimuli_digest`) — a stale cache is recomputed rather than
        returned.
        """

        digest = self.stimuli_digest()

        if digest is None:
            # No stimulus on disk: the cache is all there is, and it can only
            # be checked against the trial count.
            n_cached = (len(torch.load(save_path, weights_only=False)["embeddings"])
                        if os.path.exists(save_path) else None)
            if n_cached != len(self.df):
                raise FileNotFoundError(
                    f"no SAT stimulus image is under {self.image_dir} and "
                    + (f"{display_path(save_path)} does not exist"
                       if n_cached is None else
                       f"{display_path(save_path)} holds {n_cached} rows for "
                       f"{len(self.df)} trials")
                    + ". Build the images with `python fetch_assets.py sat-images` "
                    "(needs ImageNet in datasets/sat/ILSVRC/), or fetch the "
                    "embeddings with `python fetch_assets.py embeddings-sat`.")
            print(f"Loading DINOv2 embeddings from {display_path(save_path)} "
                  "(stimulus images not on disk: not checked against them)...")
            self.embeddings = torch.load(save_path, weights_only=False)["embeddings"]
            return self.embeddings

        if os.path.exists(save_path) and not overwrite:
            print(f"Loading DINOv2 embeddings from {display_path(save_path)}...")
            data = torch.load(save_path, weights_only=False)
            cached_digest = data.get("stimuli_digest")
            if cached_digest is not None and cached_digest != digest:
                print(f"  Stimulus digest mismatch (cached {cached_digest[:12]}, "
                      f"on disk {digest[:12]}): the images changed since this "
                      f"cache was written. Recomputing.")
            else:
                if cached_digest is None:
                    print("  [warn] cache predates the stimulus digest; cannot "
                          "verify it matches the images on disk.")
                saved_embeddings = data["embeddings"]
                if "image_paths" in data and len(saved_embeddings) != len(self.df):
                    saved_paths = data["image_paths"]
                    path_to_idx = {p: i for i, p in enumerate(saved_paths)}
                    current_paths = self.df["image_path"].tolist()
                    indices = [path_to_idx[p] for p in current_paths if p in path_to_idx]
                    if len(indices) == len(current_paths):
                        self.embeddings = saved_embeddings[indices]
                        print(f"Embeddings aligned: {len(indices)} of {len(saved_embeddings)} rows.")
                    else:
                        print(f"Warning: {len(current_paths) - len(indices)} paths not found. Recomputing...")
                        pass
                else:
                    self.embeddings = saved_embeddings
                if hasattr(self, 'embeddings'):
                    return self.embeddings

        print("Computing DINOv2 embeddings...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        model = AutoModel.from_pretrained("facebook/dinov2-small")
        model.eval().to(device)

        embeddings = []
        paths = self.df["image_path"].tolist()
        missing = sorted({p for p in paths if not (p and os.path.exists(p))})
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} SAT stimulus images are missing (first: {missing[0]}). "
                "Build them with `python fetch_assets.py sat-images`.")

        for i in tqdm(range(0, len(paths), batch_size)):
            batch_paths = paths[i:i+batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            pooled = patch_tokens.mean(dim=1)
            embeddings.append(pooled.cpu())

        self.embeddings = torch.cat(embeddings, dim=0)

        print(f"Saving embeddings to {display_path(save_path)}...")
        torch.save(
            {
                "embeddings": self.embeddings,
                "dim": self.embeddings.shape[1],
                "image_paths": paths,
                "stimuli_digest": digest,
            },
            save_path
        )

        return self.embeddings

    # ------------------------------------------------------------------
    # PYTORCH DATASET
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image = None
        if self.load_images:
            img_path = row.get("image_path")
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

        resp_str = row.get("response", "")
        resp_idx = self.cat_to_idx.get(resp_str, 0)

        sample = {
            "image":      image,
            "srid":       row["srid"],
            "block":      row["block"],
            "image_path": row.get("image_path", ""),
            "mode":       torch.tensor(int(row["mode"])      if pd.notna(row["mode"])  else -1,  dtype=torch.long),
            "level_str":  str(row.get("level_str", "unknown")),
            "time_limit": torch.tensor(int(row["time_limit"]),                                    dtype=torch.long),
            "rt":         torch.tensor(float(row["duration"]) if pd.notna(row["duration"]) else 0., dtype=torch.float32),
            "human_choice":   torch.tensor(resp_idx,                                                  dtype=torch.long),
            "choice":   torch.tensor(int(row["choice"]), dtype=torch.long),
            "image_index":    torch.tensor(int(row["image_index"]), dtype=torch.long),
            "correct":    torch.tensor(float(row["correct"]),                                     dtype=torch.float32),
            "order":      torch.tensor(float(row["order"])   if pd.notna(row["order"])  else -1., dtype=torch.float32),
        }

        if hasattr(self, "embeddings"):
            sample["embedding"] = self.embeddings[idx]

        return sample

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def _shallow_copy(self):
        new_ds = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            setattr(new_ds, k, v)
        return new_ds

    def remove_rt_outliers(self, outlier_range_ms=None):
        """
        Remove RT outliers outside a window around the beep time.
        The 'cattut' block is always kept (no beep constraints).

        Parameters
        ----------
        outlier_range_ms : list[int] or None
            Asymmetric window [a, b] in ms relative to time_limit.
            E.g. [-200, 100] keeps RT in [time_limit - 200, time_limit + 100].
            If None, uses [-100, +100].
        """
        if outlier_range_ms is None:
            lo_ms, hi_ms = -100, 100
        else:
            lo_ms, hi_ms = int(outlier_range_ms[0]), int(outlier_range_ms[1])

        before = len(self.df)

        mask_cattut = self.df['block'] == 'cattut'
        mask_exp    = self.df['block'] != 'cattut'
        valid_rt_exp = (
            (self.df['duration'] >= self.df['time_limit'] + lo_ms) &
            (self.df['duration'] <= self.df['time_limit'] + hi_ms)
        )

        final_mask = mask_cattut | (mask_exp & valid_rt_exp)

        if hasattr(self, 'embeddings'):
            mask_tensor = torch.from_numpy(final_mask.to_numpy(copy=True))
            self.embeddings = self.embeddings[mask_tensor]

        self.df = self.df[final_mask].reset_index(drop=True)
        after = len(self.df)

        print(f"RT outlier removal ([{lo_ms:+d}, {hi_ms:+d}] ms from beep): "
              f"{before - after} removed, {after} remaining.")

    def remove_rt_outliers_mad(self, rt_key='duration', mad_thresh=3.0,
                                min_rt=None, max_rt=None):
        """
        MAD-based RT outlier removal on log-transform, applied per beep.
        The 'cattut' block is always kept.

        Parameters
        ----------
        rt_key : str
            RT duration column in the DataFrame (default 'duration', in ms).
        mad_thresh : float
            Exclusion threshold in MAD units on log(RT) (default 3.0).
        min_rt, max_rt : float or None
            Optional hard bounds (same unit as rt_key, i.e. ms).
        """
        before = len(self.df)

        keep        = np.zeros(len(self.df), dtype=bool)
        mask_cattut = (self.df['block'] == 'cattut').values
        keep[mask_cattut] = True

        exp_mask = ~mask_cattut

        for beep_val in self.df.loc[exp_mask, 'time_limit'].unique():
            beep_mask = exp_mask & (self.df['time_limit'] == beep_val).values
            beep_idx  = np.where(beep_mask)[0]
            rt        = self.df[rt_key].values[beep_idx].astype(float)

            valid = np.ones(len(rt), dtype=bool)
            if min_rt is not None:
                valid &= rt >= min_rt
            if max_rt is not None:
                valid &= rt <= max_rt

            rt_v = rt[valid]
            if len(rt_v) < 2:
                keep[beep_idx[valid]] = True
                continue

            log_rt  = np.log(rt_v)
            median  = np.median(log_rt)
            mad_val = np.median(np.abs(log_rt - median))

            if mad_val == 0:
                keep[beep_idx[valid]] = True
                continue

            z  = np.abs(log_rt - median) / mad_val
            ok = z < mad_thresh
            keep[beep_idx[np.where(valid)[0][ok]]] = True

        if hasattr(self, 'embeddings'):
            mask_tensor = torch.from_numpy(keep)
            self.embeddings = self.embeddings[mask_tensor]

        self.df = self.df[keep].reset_index(drop=True)
        after = len(self.df)
        print(
            f"RT outlier removal (MAD per beep, thresh={mad_thresh}, "
            f"rt_key='{rt_key}'): {before - after} removed, {after} remaining."
        )

    def train_val_test_split(self, val_size=0.1, test_size=0.2, seed=42, shuffle=True):
        assert val_size + test_size < 1

        indices = np.arange(len(self.df))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        n_test = int(len(self.df) * test_size)
        n_val  = int(len(self.df) * val_size)

        splits = {
            "test":  indices[:n_test],
            "val":   indices[n_test: n_test + n_val],
            "train": indices[n_test + n_val:],
        }

        datasets = {}
        for name, idxs in splits.items():
            ds = self._shallow_copy()
            ds.df = self.df.iloc[idxs].reset_index(drop=True)

            if hasattr(self, 'embeddings'):
                ds.embeddings = self.embeddings[idxs]

            datasets[name] = ds
            print(f"  {name:5s}: {len(idxs)} trials")

        return datasets["train"], datasets["val"], datasets["test"]

    def shuffle(self, seed=None):
        indices = np.arange(len(self.df))
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

        self.df = self.df.iloc[indices].reset_index(drop=True)
        if hasattr(self, 'embeddings'):
            self.embeddings = self.embeddings[indices]


# ============================================================
# COLLATE FUNCTION
# ============================================================

def sat_collate_fn(batch: list[dict]) -> dict:
    has_images = batch[0]["image"] is not None
    has_embeddings = "embedding" in batch[0]

    collated = {
        "images":      torch.stack([b["image"] for b in batch]) if has_images else None,
        "modes":       torch.stack([b["mode"]       for b in batch]),
        "time_limits": torch.stack([b["time_limit"]/1000 for b in batch]),
        "rts":         torch.stack([b["rt"]/1000         for b in batch]),
        "human_choices":   torch.stack([b["human_choice"]   for b in batch]),
        "choices":     torch.stack([b["choice"]   for b in batch]),
        "image_index":   torch.stack([b["image_index"]  for b in batch]),
        "corrects":    torch.stack([b["correct"]    for b in batch]),
        "orders":      torch.stack([b["order"]      for b in batch]),
        "srids":       [b["srid"]       for b in batch],
        "blocks":      [b["block"]      for b in batch],
        "image_paths": [b["image_path"] for b in batch],
        "levels":      [b.get("level_str", "unknown") for b in batch],
    }

    if has_embeddings:
        collated["embeddings"] = torch.stack([b["embedding"] for b in batch])

    return collated


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    IMAGE_DIR = str(_REPO_ROOT / "datasets" / "sat" / "data")

    dataset = SATDataset(
        file_path=str(_REPO_ROOT / "datasets" / "sat" / "color-blur.txt"),
        image_dir=IMAGE_DIR,
    )

    dataset.compute_dinov2_embeddings(
        save_path=str(_REPO_ROOT / "embeddings" / "sat" / "sat_color_blur_embeddings.pt")
    )

    dataset.remove_rt_outliers()

    train_ds, val_ds, test_ds = dataset.train_val_test_split(val_size=0.1, test_size=0.2)

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=sat_collate_fn,
    )

    for batch in train_loader:
        if not batch:
            continue
        print("Images shape:    ", batch["images"].shape)
        if "embeddings" in batch:
            print("Embeddings shape:", batch["embeddings"].shape)
        print("RTs:             ", batch["rts"][:5])
        print("Corrects:        ", batch["corrects"][:5])
        break
